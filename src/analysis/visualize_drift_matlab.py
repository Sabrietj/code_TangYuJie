import os
import sys
import torch
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler
from tqdm import tqdm
from omegaconf import OmegaConf
import glob


# ==========================================
# 1. 自动路径设置
# ==========================================
def setup_paths():
    current_script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_script_path)
    project_root = current_dir
    while not os.path.exists(os.path.join(project_root, 'src')):
        parent = os.path.dirname(project_root)
        if parent == project_root:
            project_root = os.path.abspath(os.path.join(current_dir, '..', '..'))
            break
        project_root = parent
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    print(f"[DEBUG] 项目根目录: {project_root}")

    config_dir = os.path.join(project_root, 'src', 'models', 'flow_bert_multiview', 'config')
    config_path = os.path.join(config_dir, 'flow_bert_multiview_config.yaml')
    dataset_config_path = os.path.join(config_dir, 'datasets', 'cic_ids_2017.yaml')

    return project_root, config_path, dataset_config_path


PROJECT_ROOT, CONFIG_PATH, DATASET_CONFIG_PATH = setup_paths()

try:
    from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
    from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataset
    from src.utils.logging_config import setup_preset_logging
except ImportError as e:
    print(f"[ERROR] 导入失败: {e}")
    sys.exit(1)

logger = setup_preset_logging()

# ==========================================
# 2. 用户配置 (请确保路径正确)
# ==========================================
DATA_ROOT = r"/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
CHECKPOINT_PATH = r"/tmp/pycharm_project_982/processed_data/best_model.ckpt"
SAMPLES_PER_DAY = 2000
DAY_MAPPING = {
    "Monday": "Day 1",
    "Tuesday": "Day 2",
    "Wednesday": "Day 3",
    "Thursday": "Day 4",
    "Friday": "Day 5"
}


# ==========================================
# 3. 数据处理逻辑
# ==========================================
def load_data_and_sample(data_root, samples_per_day):
    if not os.path.exists(data_root):
        logger.error(f"数据目录不存在: {data_root}")
        sys.exit(1)

    all_dfs = []
    subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        day_label = None
        for key in DAY_MAPPING:
            if key in folder_name:
                day_label = DAY_MAPPING[key]
                break
        if not day_label: continue

        search_pattern = os.path.join(folder, "*-flow.csv")
        csv_files = glob.glob(search_pattern)
        if not csv_files: csv_files = glob.glob(os.path.join(folder, "flow.csv"))
        if not csv_files: continue

        try:
            df = pd.read_csv(csv_files[0], low_memory=False)
            if len(df) > samples_per_day:
                df = df.sample(n=samples_per_day, random_state=42)
            df = df.assign(Day_Label=day_label)
            all_dfs.append(df)
        except Exception as e:
            logger.error(f"读取失败 {folder_name}: {e}")

    if not all_dfs:
        logger.error("❌ 未加载到任何数据！")
        sys.exit(1)

    combined_df = pd.concat(all_dfs, ignore_index=True)
    # 确保时间顺序 Day 1 -> Day 5
    combined_df.sort_values('Day_Label', inplace=True)
    logger.info(f"✅ 数据加载完成，总样本: {len(combined_df)}")
    return combined_df


def extract_embeddings(model, dataset, device):
    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    embeddings_list = []
    model.to(device)
    model.eval()

    logger.info("提取特征向量中...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device)
            outputs = model(batch)
            if 'multiview_embeddings' in outputs:
                emb = outputs['multiview_embeddings']
            else:
                emb = outputs['tabular_embeddings']
            embeddings_list.append(emb.cpu().numpy())
    return np.vstack(embeddings_list)


def export_data_for_matlab(features, labels, output_csv_path):
    """
    计算坐标并保存为CSV
    """
    # 1. 特征 L2 归一化 (优化聚类效果)
    logger.info("L2 归一化...")
    features_norm = normalize(features, norm='l2', axis=1)

    # 2. t-SNE 计算
    logger.info("计算 t-SNE (这可能需要几分钟)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_norm)

    # 3. 坐标归一化到 [0, 1]
    logger.info("坐标 [0,1] 归一化...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_results_scaled = scaler.fit_transform(tsne_results)

    # 4. 保存 CSV
    df_export = pd.DataFrame({
        'x': tsne_results_scaled[:, 0],
        'y': tsne_results_scaled[:, 1],
        'Day': labels
    })

    df_export.to_csv(output_csv_path, index=False)
    logger.info(f"✅ 数据已导出供 MATLAB 使用: {output_csv_path}")
    print(f"请将此文件下载到本地: {output_csv_path}")


def main():
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"配置文件缺失: {CONFIG_PATH}")
        return
    cfg = OmegaConf.load(CONFIG_PATH)

    # 禁用漂移检测模块
    if 'concept_drift' in cfg: cfg.concept_drift.enabled = False
    if 'training' in cfg and 'drift_monitor' in cfg.training: cfg.training.drift_monitor.enabled = False

    # 加载数据集配置
    if os.path.exists(DATASET_CONFIG_PATH):
        dataset_cfg = OmegaConf.load(DATASET_CONFIG_PATH)
        cfg.datasets = dataset_cfg
        if 'flow_data_path' in dataset_cfg: cfg.data.flow_data_path = dataset_cfg.flow_data_path
        if 'session_split_path' in dataset_cfg:
            if 'session_split' not in cfg.data: cfg.data.session_split = {}
            cfg.data.session_split.session_split_path = dataset_cfg.session_split_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 加载数据
    sampled_df = load_data_and_sample(DATA_ROOT, SAMPLES_PER_DAY)

    # 2. 补全列防止报错
    malicious_col = cfg.data.is_malicious_column
    if malicious_col not in sampled_df.columns: sampled_df[malicious_col] = 0
    sampled_df[malicious_col] = sampled_df[malicious_col].fillna(0).astype(int)
    multi_col = cfg.data.multiclass_label_column
    if multi_col not in sampled_df.columns: sampled_df[multi_col] = "BENIGN"

    # 3. 初始化数据集和模型
    dataset = MultiviewFlowDataset(sampled_df, cfg, is_training=True)
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"加载模型: {CHECKPOINT_PATH}")
        model = FlowBertMultiview.load_from_checkpoint(CHECKPOINT_PATH, cfg=cfg, dataset=dataset, strict=False)
    else:
        logger.warning("⚠️ 未找到模型文件!")
        return

    # 4. 提取与导出
    features = extract_embeddings(model, dataset, device)
    labels = sampled_df['Day_Label'].values[:len(features)]

    # 保存到项目根目录下，方便查找
    output_path = os.path.join(PROJECT_ROOT, "drift_data_for_matlab.csv")
    export_data_for_matlab(features, labels, output_path)


if __name__ == "__main__":
    main()