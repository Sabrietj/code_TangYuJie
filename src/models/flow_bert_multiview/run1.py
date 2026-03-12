import hydra
from omegaconf import DictConfig
import torch
import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
import os
import sys
import logging

# ==========================================
# 🔧 路径自动定位
# ==========================================
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

# 设置 utils 路径并导入日志配置
utils_path = os.path.join(PROJECT_ROOT, 'src', 'utils')
sys.path.insert(0, utils_path)
from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

# 添加 src 到 Python 路径
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.flow_bert_multiview import FlowBertMultiview
from data.flow_bert_multiview_dataset import MultiviewFlowDataModule


def setup_environment():
    """简单的推理环境配置"""
    torch.set_float32_matmul_precision('high')
    if torch.cuda.is_available():
        os.environ['PYTORCH_MPS_DISABLE'] = '1'
        return "gpu", 1
    return "cpu", 1


@hydra.main(config_path="./config", config_name="flow_bert_multiview_config", version_base=None)
def main(cfg: DictConfig):
    if os.path.exists(PROJECT_ROOT):
        os.chdir(PROJECT_ROOT)

    accelerator, devices = setup_environment()

    # 强制解析数据集路径
    dataset_cfg = cfg.datasets
    cfg.data.flow_data_path = dataset_cfg.flow_data_path
    cfg.data.session_split.session_split_path = dataset_cfg.session_split_path

    # ==========================================
    # 1. 初始化数据模块并构建数据集
    # ==========================================
    logger.info("📦 正在准备数据...")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("fit")

    # ==========================================
    # 2. 检查并加载已训练好的模型
    # ==========================================
    # 优先使用 best_model，如果没有则尝试 last.ckpt
    ckpt_path = os.path.join(PROJECT_ROOT, "processed_data", "best_model.ckpt")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(PROJECT_ROOT, "processed_data", "last.ckpt")
        if not os.path.exists(ckpt_path):
            logger.error("❌ 找不到任何模型权重文件，请确保 processed_data 目录下存在模型。")
            return

    logger.info(f"🔍 成功定位模型权重: {ckpt_path}")
    logger.info("⚙️ 正在加载模型结构与权重...")

    model = FlowBertMultiview.load_from_checkpoint(
        checkpoint_path=ckpt_path,
        cfg=cfg,
        dataset=datamodule.train_dataset
    )
    model.eval()  # 切换为评估模式

    # ==========================================
    # 3. 拦截打印逻辑（黑魔法）：强制触发详细成绩单
    # ==========================================
    original_malicious = model._compute_and_log_is_malicious_epoch_metrics

    def force_print_malicious(*args, **kwargs):
        if 'stage' in kwargs: kwargs['stage'] = 'test'
        args_list = list(args)
        if len(args_list) > 0 and args_list[0] in ["val", "test"]: args_list[0] = "test"
        return original_malicious(*args_list, **kwargs)

    model._compute_and_log_is_malicious_epoch_metrics = force_print_malicious

    original_family = model._compute_and_log_attack_family_epoch_metrics

    def force_print_family(*args, **kwargs):
        if 'stage' in kwargs: kwargs['stage'] = 'test'
        args_list = list(args)
        if len(args_list) > 0 and args_list[0] in ["val", "test"]: args_list[0] = "test"
        return original_family(*args_list, **kwargs)

    model._compute_and_log_attack_family_epoch_metrics = force_print_family

    # ==========================================
    # 4. 合并 Train 和 Val 数据集并构造统一 DataLoader
    # ==========================================
    logger.info("============================================================")
    logger.info("🔄 正在将【训练集】和【验证集】无缝拼接，准备计算统一 Baseline...")

    combined_dataset = ConcatDataset([datamodule.train_dataset, datamodule.val_dataset])
    val_loader_ref = datamodule.val_dataloader()

    combined_loader = DataLoader(
        combined_dataset,
        batch_size=val_loader_ref.batch_size,
        num_workers=val_loader_ref.num_workers,
        collate_fn=val_loader_ref.collate_fn,
        pin_memory=val_loader_ref.pin_memory,
        shuffle=False  # 纯评估不需要打乱数据
    )

    total_samples = len(combined_dataset)
    logger.info(f"✅ 数据合并完成！总计包含 {total_samples} 条已知数据。")
    logger.info("============================================================")

    # ==========================================
    # 5. 初始化轻量级 Trainer 并出成绩
    # ==========================================
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0  # 关闭开头的 512 样本探针检查
    )

    logger.info("🚀 开始运行统一 Baseline 前向推理...")
    trainer.validate(model=model, dataloaders=combined_loader)

    logger.info("🎉 统一 Baseline 成绩单均已输出完毕！(本次运行未接触 Test 集)")


if __name__ == "__main__":
    main()