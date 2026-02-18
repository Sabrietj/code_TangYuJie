import os
import sys
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, MinMaxScaler
from tqdm import tqdm
from omegaconf import OmegaConf
import glob


# ==========================================
# 1. 自动路径修复逻辑
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
# 2. 用户配置区域
# ==========================================

# ⚠️ 请确保路径为您服务器上的真实路径
DATA_ROOT = r"/tmp/pycharm_project_982/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
CHECKPOINT_PATH = r"/tmp/pycharm_project_982/processed_data/best_model.ckpt"

# 采样数
SAMPLES_PER_DAY = 2000

# 标签映射
DAY_MAPPING = {
    "Monday": "Day 1",
    "Tuesday": "Day 2",
    "Wednesday": "Day 3",
    "Thursday": "Day 4",
    "Friday": "Day 5"
}


# ==========================================
# 3. 核心逻辑
# ==========================================

def load_data_and_sample(data_root, samples_per_day):
    if not os.path.exists(data_root):
        logger.error(f"数据目录不存在: {data_root}")
        sys.exit(1)

    all_dfs = []
    subfolders = [f.path for f in os.scandir(data_root) if f.is_dir()]
    logger.info(f"扫描到 {len(subfolders)} 个子文件夹")

    for folder in subfolders:
        folder_name = os.path.basename(folder)
        day_label = None
        for key in DAY_MAPPING:
            if key in folder_name:
                day_label = DAY_MAPPING[key]
                break

        if not day_label:
            continue

        search_pattern = os.path.join(folder, "*-flow.csv")
        csv_files = glob.glob(search_pattern)
        if not csv_files:
            csv_files = glob.glob(os.path.join(folder, "flow.csv"))

        if not csv_files:
            continue

        csv_path = csv_files[0]

        try:
            df = pd.read_csv(csv_path, low_memory=False)
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


def plot_tsne(features, labels, output_path):
    """
    绘制符合期刊审美的 t-SNE 图
    - 包含总标题
    - 图例位于右下角
    - 坐标归一化 [0,1]
    """
    # 1. 特征归一化 (L2)
    logger.info("进行 L2 特征归一化...")
    features_norm = normalize(features, norm='l2', axis=1)

    # 2. t-SNE 降维
    logger.info("计算 t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features_norm)

    # 3. 坐标归一化 [0, 1]
    logger.info("进行坐标 [0,1] 归一化...")
    scaler = MinMaxScaler(feature_range=(0, 1))
    tsne_results_scaled = scaler.fit_transform(tsne_results)

    # 4. 绘图数据准备
    plot_df = pd.DataFrame({
        'x': tsne_results_scaled[:, 0],
        'y': tsne_results_scaled[:, 1],
        'Day': labels
    })

    # 5. 设置绘图风格
    plt.figure(figsize=(10, 8))
    sns.set_style("white")
    sns.set_context("paper", font_scale=1.4)

    # 绘制散点图
    scatter = sns.scatterplot(
        x="x", y="y", hue="Day",
        palette="tab10",
        data=plot_df,
        alpha=0.85,
        s=40,
        edgecolor='w',
        linewidth=0.3
    )

    # 6. 坐标轴处理 (只保留边框和数值，去掉Label)
    plt.xlabel('')
    plt.ylabel('')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)

    # 保留四周边框
    sns.despine(top=False, right=False, left=False, bottom=False)
    #
    # # 7. 添加整张图的标题
    # plt.title("Visualization of Concept Drift (CIC-IDS-2017)", fontsize=18, weight='bold', pad=15)

    # 8. 图例优化 (右下角)
    plt.legend(
        loc='lower right',  # 右下角
        title="Time Window",
        frameon=True,  # 开启背景框
        facecolor='white',  # 白色背景
        framealpha=0.9,  # 90% 不透明，防止遮挡
        edgecolor='#cccccc',  # 浅灰色边框
        fontsize=11,
        title_fontsize=12,
        borderpad=0.8  # 内边距
    )

    # 9. 保存
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"✅ 期刊风格图片已保存: {output_path}")


def main():
    if not os.path.exists(CONFIG_PATH):
        logger.error(f"配置文件缺失: {CONFIG_PATH}")
        return
    cfg = OmegaConf.load(CONFIG_PATH)

    # 关闭漂移检测
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

    # 2. 预处理
    malicious_col = cfg.data.is_malicious_column
    if malicious_col not in sampled_df.columns: sampled_df[malicious_col] = 0
    sampled_df[malicious_col] = sampled_df[malicious_col].fillna(0).astype(int)

    multi_col = cfg.data.multiclass_label_column
    if multi_col not in sampled_df.columns: sampled_df[multi_col] = "BENIGN"

    # 3. 初始化数据集
    logger.info("初始化数据集...")
    dataset = MultiviewFlowDataset(sampled_df, cfg, is_training=True)

    # 4. 加载模型
    if os.path.exists(CHECKPOINT_PATH):
        logger.info(f"加载模型: {CHECKPOINT_PATH}")
        model = FlowBertMultiview.load_from_checkpoint(CHECKPOINT_PATH, cfg=cfg, dataset=dataset, strict=False)
    else:
        logger.warning("⚠️ 未找到模型文件，使用随机初始化模型演示！")
        model = FlowBertMultiview(cfg, dataset)

    # 5. 提取特征
    features = extract_embeddings(model, dataset, device)
    labels = sampled_df['Day_Label'].values[:len(features)]

    # 6. 绘图
    plot_tsne(features, labels, "/tmp/pycharm_project_982/concept_drift_final.png")


if __name__ == "__main__":
    main()