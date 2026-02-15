import hydra
from omegaconf import DictConfig, OmegaConf, ListConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
import os
import sys
import logging
import torch
import shutil  # 用于文件复制
from pytorch_lightning import seed_everything
from data.prepare_flow_file_pipeline import prepare_sampled_data_files

# ==========================================
# 🔧 [核心修复] 路径自动定位
# ==========================================
# 获取 train.py 所在的绝对路径
current_file_path = os.path.abspath(__file__)
# 回退 4 层找到项目根目录 code_TangYuJie
# (src/models/flow_bert_multiview/train.py -> src/models/flow_bert_multiview -> src/models -> src -> ROOT)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

# 设置 utils 路径
utils_path = os.path.join(PROJECT_ROOT, 'src', 'utils')
sys.path.insert(0, utils_path)

# 设置日志
from logging_config import setup_preset_logging

# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)

# 添加 src 到 Python 路径
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.flow_bert_multiview import FlowBertMultiview
from data.flow_bert_multiview_dataset import MultiviewFlowDataModule


def setup_environment(cfg: DictConfig):
    """根据硬件可用性自动设置训练环境"""
    force_single_gpu = cfg.get('force_single_gpu', False)

    if force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["PYTORCH_MPS_DISABLE"] = "1"
        cfg.data.num_workers = 0
        logger.info("✅ 强制使用单 GPU 训练")

    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        cuda_count = torch.cuda.device_count()
        logger.info(f"检测到 {cuda_count} 个CUDA设备可用")

        os.environ['PYTORCH_MPS_DISABLE'] = '1'
        accelerator = "gpu"
        if force_single_gpu:
            devices = 1
            strategy = "auto"
        else:
            devices = torch.cuda.device_count()
            strategy = "ddp_find_unused_parameters_true" if devices > 1 else "auto"
        logger.info("使用GPU进行训练")
    else:
        logger.info("未检测到GPU，使用CPU进行训练")
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    return accelerator, devices, strategy


def validate_config(cfg: DictConfig):
    """验证配置完整性 (简化版)"""
    if not hasattr(cfg.optimizer, 'default_total_steps'):
        cfg.optimizer.default_total_steps = 10000

    # 简单的必要性检查
    if not cfg.model.bert.model_name:
        raise ValueError("Config Error: model.bert.model_name is missing")


def resolve_dataset_paths(cfg):
    dataset_cfg = cfg.datasets
    cfg.data.flow_data_path = dataset_cfg.flow_data_path
    cfg.data.session_split.session_split_path = dataset_cfg.session_split_path
    logger.info(f"📦 使用数据集: {cfg.data.dataset}")


@hydra.main(config_path="./config", config_name="flow_bert_multiview_config", version_base=None)
def main(cfg: DictConfig):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    logger.info(f"📍 项目根目录定位: {PROJECT_ROOT}")

    try:
        seed_everything(cfg.data.random_state, workers=True)

        # 切换回项目根目录，确保相对路径正确
        if os.path.exists(PROJECT_ROOT):
            os.chdir(PROJECT_ROOT)

        accelerator, devices, strategy = setup_environment(cfg)
        validate_config(cfg)
        resolve_dataset_paths(cfg)

        prepare_sampled_data_files(cfg)

        datamodule = MultiviewFlowDataModule(cfg)
        datamodule.setup("fit")

        model = FlowBertMultiview(cfg, dataset=datamodule.train_dataset)

        # 漂移检测器初始化日志
        if hasattr(model, 'drift_detector') and model.drift_detector is not None:
            stats = model.drift_detector.get_statistics()
            logger.info(f"✅ 漂移检测器就绪: {stats}")

        logger_tb = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.name,
            version=cfg.logging.get('version')
        )

        # 1. 确定保存目录: code_TangYuJie/processed_data
        export_dir_name = cfg.training.get('model_export_dir', 'processed_data')

        # 如果是绝对路径则直接使用，否则拼接在 PROJECT_ROOT 下
        if os.path.isabs(export_dir_name):
            checkpoint_dir = export_dir_name
        else:
            checkpoint_dir = os.path.join(PROJECT_ROOT, export_dir_name)

        # 确保目录存在
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"📂 模型将强制保存至: {checkpoint_dir}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,  # ✅ 强制直接写入目标文件夹，不使用默认的 lightning_logs
            monitor=cfg.training.model_checkpoint.monitor,
            mode=cfg.training.model_checkpoint.mode,
            save_top_k=cfg.training.model_checkpoint.save_top_k,
            filename=cfg.training.model_checkpoint.filename,  # 例如 "best-{epoch}"
            save_last=True,  # ✅ 强制保存 last.ckpt，即使验证集指标未触发
            auto_insert_metric_name=False
        )

        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.get('min_delta', 0.001)
        )

        # 漂移检测回调
        drift_callbacks = []
        if hasattr(cfg.training, 'drift_monitor') and cfg.training.drift_monitor.enabled:
            try:
                from pytorch_lightning.callbacks import Callback
                class DriftMonitorCallback(Callback):
                    def __init__(self, check_interval=50):
                        super().__init__()
                        self.check_interval = check_interval
                        self.batch_count = 0

                    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
                        self.batch_count += 1
                        if self.batch_count % self.check_interval == 0:
                            if hasattr(pl_module, 'drift_detector') and pl_module.drift_detector is not None:
                                is_drift, B, info = pl_module.drift_detector.detect_drift()
                                # 仅做日志记录，不干预训练

                drift_callback = DriftMonitorCallback(
                    check_interval=cfg.training.drift_monitor.get('check_interval', 50)
                )
                drift_callbacks.append(drift_callback)
            except Exception as e:
                logger.warning(f"概念漂移回调创建失败: {e}")

        callbacks = [checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='step')]
        callbacks.extend(drift_callbacks)

        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.training.precision,
            # gradient_clip_val=cfg.training.gradient_clip_val,
            callbacks=callbacks,
            strategy=strategy,
            enable_progress_bar=True,
            logger=logger_tb,
            detect_anomaly=cfg.training.get('detect_anomaly', False)
        )

        logger.info("🚀 开始训练...")
        trainer.fit(model, datamodule=datamodule)
        logger.info("🏁 训练完成！")

        # ==========================================================
        # 🔧 [核心修复] 确保 best_model_path 有效并复制为标准名称
        # ==========================================================
        best_model_path = checkpoint_callback.best_model_path

        # 1. 检查 best_model 是否存在
        if not best_model_path or not os.path.exists(best_model_path):
            logger.warning("⚠️ 未找到 Best Model (可能验证集指标未触发)，尝试使用 Last Model...")
            best_model_path = checkpoint_callback.last_model_path

        # 2. 如果找到了有效模型 (best 或 last)
        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"✅ 捕获到模型路径: {best_model_path}")

            # 3. 统一复制为 'best_model.ckpt' 以供下游任务使用
            final_path = os.path.join(checkpoint_dir, "best_model.ckpt")

            # 避免自我复制
            if os.path.abspath(best_model_path) != os.path.abspath(final_path):
                shutil.copy(best_model_path, final_path)
                logger.info(f"📦 已将模型归档至: {final_path}")

            # 更新路径变量用于测试
            best_model_path = final_path
        else:
            logger.error("❌ 严重错误: 未生成任何模型文件！(best_model_path 和 last_model_path 均无效)")
            logger.error("请检查: 1. max_epochs 是否太小? 2. 数据集是否为空? 3. 磁盘空间是否已满?")
            return  # 退出，无法测试

        # ==========================================================
        # 测试阶段
        # ==========================================================
        logger.info(f"🧪 开始测试 (加载: {best_model_path})...")
        best_model = FlowBertMultiview.load_from_checkpoint(
            best_model_path,
            cfg=cfg,
            dataset=datamodule.train_dataset
        )

        trainer_test = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
        )
        trainer_test.test(best_model, datamodule=datamodule)

        if hasattr(best_model, 'drift_detector') and best_model.drift_detector is not None:
            best_model.on_test_end()

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()