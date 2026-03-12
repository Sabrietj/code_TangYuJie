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
from torch.utils.data import ConcatDataset, DataLoader  # 🟢 引入数据集拼接工具

from data.prepare_flow_file_pipeline import prepare_sampled_data_files

# ==========================================
# 🔧 [核心修复] 路径自动定位
# ==========================================
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

utils_path = os.path.join(PROJECT_ROOT, 'src', 'utils')
sys.path.insert(0, utils_path)

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.flow_bert_multiview import FlowBertMultiview
from data.flow_bert_multiview_dataset import MultiviewFlowDataModule


def setup_environment(cfg: DictConfig):
    """根据硬件可用性自动设置训练环境"""
    force_single_gpu = cfg.get('force_single_gpu', False)

    if force_single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 如果想用卡0可以改回0
        os.environ["PYTORCH_MPS_DISABLE"] = "1"
        # 如果资源够，你可以尝试把这里的 0 改大一点（比如 4），提升数据读取速度
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

        if os.path.exists(PROJECT_ROOT):
            os.chdir(PROJECT_ROOT)

        accelerator, devices, strategy = setup_environment(cfg)
        validate_config(cfg)
        resolve_dataset_paths(cfg)

        prepare_sampled_data_files(cfg)

        # 1. 初始化数据模块
        datamodule = MultiviewFlowDataModule(cfg)
        datamodule.setup("fit")

        # 2. 初始化模型
        model = FlowBertMultiview(cfg, dataset=datamodule.train_dataset)

        if hasattr(model, 'drift_detector') and model.drift_detector is not None:
            stats = model.drift_detector.get_statistics()
            logger.info(f"✅ 漂移检测器就绪: {stats}")

        logger_tb = TensorBoardLogger(
            save_dir=cfg.logging.save_dir,
            name=cfg.logging.name,
            version=cfg.logging.get('version')
        )

        export_dir_name = cfg.training.get('model_export_dir', 'processed_data')
        if os.path.isabs(export_dir_name):
            checkpoint_dir = export_dir_name
        else:
            checkpoint_dir = os.path.join(PROJECT_ROOT, export_dir_name)

        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.info(f"📂 模型将强制保存至: {checkpoint_dir}")

        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            monitor=cfg.training.model_checkpoint.monitor,
            mode=cfg.training.model_checkpoint.mode,
            save_top_k=cfg.training.model_checkpoint.save_top_k,
            filename=cfg.training.model_checkpoint.filename,
            save_last=True,
            auto_insert_metric_name=False
        )

        early_stopping = EarlyStopping(
            monitor=cfg.training.early_stopping.monitor,
            patience=cfg.training.early_stopping.patience,
            mode=cfg.training.early_stopping.mode,
            min_delta=cfg.training.early_stopping.get('min_delta', 0.001)
        )

        callbacks = [checkpoint_callback, early_stopping, LearningRateMonitor(logging_interval='step')]

        # 3. 开始训练
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.training.precision,
            callbacks=callbacks,
            strategy=strategy,
            enable_progress_bar=True,
            logger=logger_tb,
            detect_anomaly=cfg.training.get('detect_anomaly', False),
            num_sanity_val_steps=0  # 🟢 彻底关闭训练前那 512 个样本的健康检查打印！
        )

        logger.info("🚀 开始训练...")
        trainer.fit(model, datamodule=datamodule)
        logger.info("🏁 训练完成！")

        # ==========================================================
        # 提取最佳模型并归档
        # ==========================================================
        best_model_path = checkpoint_callback.best_model_path

        if not best_model_path or not os.path.exists(best_model_path):
            logger.warning("⚠️ 未找到 Best Model，尝试使用 Last Model...")
            best_model_path = checkpoint_callback.last_model_path

        if best_model_path and os.path.exists(best_model_path):
            logger.info(f"✅ 捕获到模型路径: {best_model_path}")
            final_path = os.path.join(checkpoint_dir, "best_model.ckpt")
            if os.path.abspath(best_model_path) != os.path.abspath(final_path):
                shutil.copy(best_model_path, final_path)
                logger.info(f"📦 已将模型归档至: {final_path}")
            best_model_path = final_path
        else:
            logger.error("❌ 严重错误: 未生成任何模型文件！")
            return

        logger.info("============================================================")

        # 1. 加载最佳模型
        best_model = FlowBertMultiview.load_from_checkpoint(
            best_model_path,
            cfg=cfg,
            dataset=datamodule.train_dataset
        )

        # 临时拦截模型内置打印条件，欺骗模型以为当前在跑 test，强制输出表格
        original_malicious = best_model._compute_and_log_is_malicious_epoch_metrics

        def force_print_malicious(*args, **kwargs):
            if 'stage' in kwargs: kwargs['stage'] = 'test'
            args_list = list(args)
            if len(args_list) > 0 and args_list[0] in ["val", "test"]: args_list[0] = "test"
            return original_malicious(*args_list, **kwargs)

        best_model._compute_and_log_is_malicious_epoch_metrics = force_print_malicious

        original_family = best_model._compute_and_log_attack_family_epoch_metrics

        def force_print_family(*args, **kwargs):
            if 'stage' in kwargs: kwargs['stage'] = 'test'
            args_list = list(args)
            if len(args_list) > 0 and args_list[0] in ["val", "test"]: args_list[0] = "test"
            return original_family(*args_list, **kwargs)

        best_model._compute_and_log_attack_family_epoch_metrics = force_print_family

        # 🚀 启动轻量级评估，【仅仅传入验证集】
        trainer_baseline = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0
        )

        # 仅仅使用 validation_dataloader！
        trainer_baseline.validate(best_model, dataloaders=datamodule.val_dataloader())

        logger.info("🎉 纯验证集 Baseline 成绩单均已输出完毕！本次运行全程未接触 Test 集。")
        logger.info("============================================================")

    except Exception as e:
        logger.error(f"❌ 训练失败: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()