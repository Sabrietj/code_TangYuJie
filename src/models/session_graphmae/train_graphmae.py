import os
import sys
import torch
import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
import logging

current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)
from models.session_graphmae.graphmae_model import SessionGraphMAE


def main():
    pl.seed_everything(42, workers=True)

    dataset_dir = "/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
    pt_path = os.path.join(dataset_dir, "embedded_graphs.pt")

    if not os.path.exists(pt_path):
        logger.error(f"❌ 找不到特征图文件 {pt_path}，请先运行 extract_graph_embeddings.py")
        return

    logger.info(f"📦 成功定位并加载图数据集: {pt_path}")

    dataset = torch.load(pt_path, weights_only=False)

    train_dataset = [g for g in dataset if 'train' in getattr(g, 'split', 'train')]
    val_dataset = [g for g in dataset if 'valid' in getattr(g, 'split', '') or 'val' in getattr(g, 'split', '')]

    if not val_dataset:
        logger.warning("未找到 val 验证集数据，后备方案：切分 train 做验证")
        split_idx = int(len(train_dataset) * 0.8)
        val_dataset = train_dataset[split_idx:]
        train_dataset = train_dataset[:split_idx]

    logger.info(f"📊 图数据集分布：Train={len(train_dataset)} 张图, Val={len(val_dataset)} 张图")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = SessionGraphMAE(
        in_dim=768,
        hidden_dim=128,
        num_attack_families=6,
        mask_rate=0.5,
        lr=0.0001,
        enc_layers=4,
        dec_layers=4
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=dataset_dir,
        monitor="train_loss",
        save_top_k=1,
        filename="best_graphmae",
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[checkpoint_callback],
        num_sanity_val_steps=0,
        enable_progress_bar=False  # 🟢 彻底关掉恶心的刷屏进度条！
    )

    logger.info("🚀 开启 GraphMAE 极速训练模式...（由于禁用了进度条，请耐心等待，每个 Epoch 会打印一次日志）")
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    logger.info("🏁 训练完成！正在验证最佳模型输出成绩单...")
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"✅ 加载最佳图模型: {best_model_path}")
        best_model = SessionGraphMAE.load_from_checkpoint(best_model_path)

        best_model.print_final_report = True

        trainer.validate(best_model, dataloaders=val_loader)


if __name__ == "__main__":
    main()