# src/models/session_graphmae/train_graphmae.py
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
import os
import sys
import logging
import torch

# ==========================================
# 🔧 路径自动定位
# ==========================================
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))

utils_path = os.path.join(PROJECT_ROOT, 'src', 'utils')
sys.path.insert(0, utils_path)
from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from models.session_graphmae.graphmae_model import SessionGraphMAE


# 注意：你需要自己实现一个 SessionGraphDataModule，用于加载 .bin 文件为 PyG DataLoader
# from data.session_graph_datamodule import SessionGraphDataModule

# 这是一个占位适配器：你需要在内部调用 FlowBertMultiview 来提取节点特征
class MultiviewNodeEncoderAdapter(torch.nn.Module):
    def __init__(self, pretrained_flow_bert_path):
        super().__init__()
        # 加载你之前训练保存好的最佳 flow_bert 模型
        # self.flow_bert = FlowBertMultiview.load_from_checkpoint(pretrained_flow_bert_path)
        # 冻结底层特征提取器以节省显存 (可选)
        # for param in self.flow_bert.parameters(): param.requires_grad = False

    def forward(self, pyg_batch):
        # 将 PyG batch 转化为 flow_bert 需要的格式，提取融合后的特征
        # return self.flow_bert.extract_node_features(pyg_batch)
        pass


@hydra.main(config_path="./config", config_name="graphmae_config", version_base=None)
def main(cfg: DictConfig):
    os.environ['HYDRA_FULL_ERROR'] = '1'
    pl.seed_everything(cfg.data.random_state, workers=True)
    os.chdir(PROJECT_ROOT)

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    devices = 1

    # 1. 准备图数据模块
    logger.info("📦 正在加载 Session Graph 数据...")
    # datamodule = SessionGraphDataModule(cfg)
    # datamodule.setup("fit")

    # 2. 准备节点编码器 (复用之前训练好的 FlowBert 提取能力)
    best_flow_bert_path = os.path.join(PROJECT_ROOT, "processed_data", "best_model.ckpt")
    node_encoder = MultiviewNodeEncoderAdapter(best_flow_bert_path)

    # 3. 初始化 GraphMAE 模型
    model = SessionGraphMAE(cfg, node_encoder)

    # 4. 配置日志与保存点
    checkpoint_dir = os.path.join(PROJECT_ROOT, "processed_data", "graphmae_checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        filename="best_graphmae",
        save_last=False
    )

    # 5. 开始多任务训练
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator=accelerator,
        devices=devices,
        callbacks=[checkpoint_callback],
        enable_progress_bar=True,
        # 🟢 彻底关闭训练前的 512 样本检测，保持日志纯净
        num_sanity_val_steps=0
    )

    logger.info("🚀 开始在 Session 拓扑图上训练 GraphMAE (重建 + 分类 多任务学习)...")
    # trainer.fit(model, datamodule=datamodule)
    logger.info("🏁 GraphMAE 训练彻底完成！")

    # ==========================================================
    # 🏅 终极阶段：纯验证集 Baseline 评估 (绝不碰 Test)
    # ==========================================================
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path and os.path.exists(best_model_path):
        logger.info("============================================================")
        logger.info("🔄 正在加载最佳 GraphMAE，准备输出纯净的 In-domain Baseline 成绩...")

        # 加载最佳模型
        # best_model = SessionGraphMAE.load_from_checkpoint(best_model_path, cfg=cfg, node_encoder=node_encoder)

        # 🟢 设置魔法标志，触发 validation_epoch_end 打印最底部的成绩表格
        # best_model.print_baseline_report = True

        trainer_baseline = pl.Trainer(
            accelerator=accelerator,
            devices=1,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0
        )

        # 仅仅使用 validation_dataloader！严格遵守防数据泄露的学术标准
        # trainer_baseline.validate(best_model, dataloaders=datamodule.val_dataloader())

        logger.info("🎉 纯验证集 GraphMAE Baseline 成绩单输出完毕！本次未接触 Test 集。")
        logger.info("============================================================")
    else:
        logger.warning("⚠️ 训练可能被跳过或未保存最佳模型。")


if __name__ == "__main__":
    main()