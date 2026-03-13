import os
import sys
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
from tqdm import tqdm
from torch_geometric.data import Data
from torch.utils.data.dataloader import default_collate
from dgl.data.utils import load_graphs, load_info

# --- 路径定位 ---
current_file_path = os.path.abspath(__file__)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_file_path))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src', 'utils'))
sys.path.append(os.path.join(PROJECT_ROOT, 'src'))

from logging_config import setup_preset_logging

logger = setup_preset_logging(log_level=logging.INFO)

from models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule


@hydra.main(config_path="../flow_bert_multiview/config", config_name="flow_bert_multiview_config", version_base=None)
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_dir = "/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"

    # =========================================================================
    # 🟢 终极修复：完全对齐你 cic_ids_2017.yaml 里的真实文件名！
    # =========================================================================
    OmegaConf.set_struct(cfg, False)  # 允许修改 config

    # 补全缺失的 dataset 变量
    if "dataset" not in cfg:
        cfg.dataset = {}
    cfg.dataset.name = "CIC-IDS-2017"

    # 强制将报错的路径替换为真实的绝对路径
    cfg.data.flow_data_path = os.path.join(dataset_dir, "all_embedded_flow.csv")
    if "session_split" not in cfg.data:
        cfg.data.session_split = {}

    # 🚨 就是这里！修复了之前错误的 .pkl，改为真实的 all_split_session.csv
    cfg.data.session_split.session_split_path = os.path.join(dataset_dir, "all_split_session.csv")

    logger.info(f"🔒 已跳过 Hydra 解析，强制将流数据路径锁定为: {cfg.data.flow_data_path}")
    logger.info(f"🔒 已强制将 split 数据路径锁定为: {cfg.data.session_split.session_split_path}")
    # =========================================================================

    # 1. 启动完整的 DataModule
    logger.info("📦 正在初始化 DataModule 确保特征预处理逻辑与训练一致...")
    datamodule = MultiviewFlowDataModule(cfg)
    datamodule.setup("fit")
    datamodule.setup("test")

    # 构建全局 UID 映射字典
    uid2dataset = {}
    for ds in [datamodule.train_dataset, datamodule.validate_dataset, datamodule.test_dataset]:
        if ds is not None:
            for i, uid in enumerate(ds.flow_df['uid'].values):
                uid2dataset[uid] = (ds, i)
    logger.info(f"✅ 成功构建全局 UID 映射字典，共缓存 {len(uid2dataset)} 条流。")

    # 2. 加载训练好的 FlowBertMultiview 模型
    ckpt_path = os.path.join(PROJECT_ROOT, "processed_data", "best_model.ckpt")
    logger.info(f"⚙️ 加载最佳流级别特征提取器: {ckpt_path}")
    flow_bert = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=datamodule.train_dataset)
    flow_bert.to(device)
    flow_bert.eval()

    # 3. 加载 DGL 图和图信息
    bin_path = os.path.join(dataset_dir, "all_session_graph.bin")
    info_path = os.path.join(dataset_dir, "all_session_graph_info.pkl")

    if not os.path.exists(bin_path):
        logger.error(f"❌ 找不到原图文件: {bin_path}。请确认你是否重新跑了 build_session_graph")
        return

    dgl_graphs, _ = load_graphs(bin_path)
    info_dict = load_info(info_path)

    if 'node_uids' not in info_dict:
        logger.error("❌ info.pkl 中没有 node_uids！请务必修改 session_graph_builder.py 保存 node_uids，并重新运行构图！")
        return

    node_uids_list = info_dict['node_uids']
    splits = info_dict['split']

    pyg_graphs = []
    logger.info(f"🚀 开始转换并提取 {len(dgl_graphs)} 张图的 768维 表征向量...")

    with torch.no_grad():
        for i, (g, uids, split) in enumerate(tqdm(zip(dgl_graphs, node_uids_list, splits), total=len(dgl_graphs))):

            batch_items = []
            for uid in uids:
                if uid in uid2dataset:
                    ds, idx = uid2dataset[uid]
                    batch_items.append(ds[idx])

            if not batch_items:
                continue

            batch = default_collate(batch_items)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)

            # 调用接口提取融合特征
            X = flow_bert.extract_fusion_features(batch)

            # 提取图标签
            y_bin = batch['is_malicious_label'].max().view(1).float()
            y_multi = batch['attack_family_label'].max(dim=0)[0].unsqueeze(0).float()

            # 边转换
            src, dst = g.edges()
            edge_index = torch.stack([src, dst], dim=0).to(torch.long)

            # 构建 PyG Data
            pyg_data = Data(x=X.cpu(), edge_index=edge_index.cpu(), y_bin=y_bin.cpu(), y_multi=y_multi.cpu())
            pyg_data.split = str(split).strip().lower()
            pyg_graphs.append(pyg_data)

    # 4. 将提取好的图保存回你的 Dataset 专属文件夹中
    out_path = os.path.join(dataset_dir, "embedded_graphs.pt")
    torch.save(pyg_graphs, out_path)
    logger.info(f"🎉 转换大功告成！全量 {len(pyg_graphs)} 张 PyG 图已保存至: {out_path}")


if __name__ == "__main__":
    main()