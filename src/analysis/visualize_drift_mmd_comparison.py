import os
import sys
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from omegaconf import OmegaConf
import glob


# ==========================================
# 1. 自动路径修复与环境配置
# ==========================================
def setup_paths():
    current_script_path = os.path.abspath(__file__)
    project_root = os.path.abspath(os.path.join(os.path.dirname(current_script_path), '..', '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    config_dir = os.path.join(project_root, 'src', 'models', 'flow_bert_multiview', 'config')
    config_path = os.path.join(config_dir, 'flow_bert_multiview_config.yaml')
    dataset_config_path = os.path.join(config_dir, 'datasets', 'cic_ids_2017.yaml')
    return project_root, config_path, dataset_config_path


PROJECT_ROOT, CONFIG_PATH, DATASET_CONFIG_PATH = setup_paths()

from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector, KSWINDetector


# ==========================================
# 2. 工具函数 (路径解析与检查点寻找)
# ==========================================
def resolve_path(raw_path, dataset_name, project_root):
    if raw_path is None: return ""
    if "${dataset.name}" in raw_path:
        raw_path = raw_path.replace("${dataset.name}", dataset_name)
    if raw_path.startswith("."):
        raw_path = os.path.normpath(os.path.join(project_root, raw_path))
    return raw_path


def find_valid_checkpoint(project_root):
    processed_dir = os.path.join(project_root, "processed_data")
    checkpoints_dir = os.path.join(project_root, "checkpoints")
    candidates = [
        os.path.join(processed_dir, "best_model.ckpt"),
        os.path.join(processed_dir, "last.ckpt"),
        os.path.join(checkpoints_dir, "best_model.ckpt"),
        os.path.join(checkpoints_dir, "last.ckpt"),
    ]
    if os.path.exists(processed_dir):
        extra_ckpts = glob.glob(os.path.join(processed_dir, "*.ckpt"))
        extra_ckpts.sort(key=os.path.getmtime, reverse=True)
        candidates.extend(extra_ckpts)

    for ckpt in candidates:
        if os.path.exists(ckpt): return ckpt
    return None


# ==========================================
# 3. MMD (最大均值差异) 计算函数
# ==========================================
def compute_mmd(x, y, kernel='rbf'):
    """计算两个张量集合的 MMD 距离 (使用高斯 RBF 核)"""
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t() + ry - 2. * zz

    XX, YY, XY = (torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device),
                  torch.zeros(xx.shape).to(x.device))

    bandwidth_range = [10, 15, 20, 50]
    for a in bandwidth_range:
        XX += torch.exp(-0.5 * dxx / a)
        YY += torch.exp(-0.5 * dyy / a)
        XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY).item()


# ==========================================
# 4. 核心实验与绘图逻辑
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 步骤 1: 加载配置与模型 ---
    cfg = OmegaConf.load(CONFIG_PATH)
    OmegaConf.set_struct(cfg, False)  # 解除结构限制，允许动态添加属性

    # 手动解析 dataset.name 的插值问题
    if os.path.exists(DATASET_CONFIG_PATH):
        dataset_cfg = OmegaConf.load(DATASET_CONFIG_PATH)
        cfg.datasets = dataset_cfg  # 修复: 这里必须是 datasets (复数)
        cfg.data.dataset = dataset_cfg.name
        if 'flow_data_path' in dataset_cfg:
            cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name, PROJECT_ROOT)
        if 'session_split_path' in dataset_cfg:
            if 'session_split' not in cfg.data: cfg.data.session_split = {}
            cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name,
                                                                     PROJECT_ROOT)
        if 'class_weights' in dataset_cfg:
            if 'loss' not in cfg: cfg.loss = {}
            cfg.loss.class_weights = dataset_cfg.class_weights

    # 初始化 DataLoader
    cfg.data.split_mode = "flow"
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")
    dataloader = dm.test_dataloader()

    # 尝试加载最佳检查点，提取高质量特征
    ckpt_path = find_valid_checkpoint(PROJECT_ROOT)
    if ckpt_path:
        print(f"✅ 成功加载检查点: {ckpt_path}")
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dm.train_dataset, strict=False)
    else:
        print("⚠️ 未找到检查点，使用随机初始化模型提取特征")
        model = FlowBertMultiview(cfg, dm.train_dataset)

    model.to(device)
    model.eval()

    # --- 步骤 2: 初始化三种检测器 ---
    det_config = {'seed': 2026, 'threshold': 0.01, 'max_level': 6, 'delta': 0.002, 'alpha': 0.005}
    detectors = {
        'BNDM': BNDMDetector(det_config),
        'ADWIN': ADWINDetector(det_config),
        'KSWIN': KSWINDetector(det_config)
    }

    alarm_points = {'BNDM': [], 'ADWIN': [], 'KSWIN': []}
    mmd_values = []

    # MMD 窗口设置
    ref_window = []
    cur_window = []
    window_size = 50  # 每个窗口包含 50 个 batch 的特征

    global_batch_idx = 0

    # --- 步骤 3: 模拟流式数据与检测 ---
    print("🚀 正在提取特征并计算 MMD 分布距离...")
    with torch.no_grad():
        for batch in tqdm(dataloader, total=min(len(dataloader), 500)):
            if global_batch_idx > 500: break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            features = outputs['multiview_embeddings']  # [Batch_size, Embed_dim]

            # --- A. 记录用于计算 MMD 的数据 ---
            batch_mean_feat = features.mean(dim=0)
            if len(ref_window) < window_size:
                ref_window.append(batch_mean_feat)
            else:
                cur_window.append(batch_mean_feat)
                if len(cur_window) == window_size:
                    ref_tensor = torch.stack(ref_window)
                    cur_tensor = torch.stack(cur_window)
                    mmd_val = compute_mmd(ref_tensor, cur_tensor)
                    mmd_values.append((global_batch_idx, mmd_val))

                    # 华丽滑动
                    ref_window = cur_window.copy()
                    cur_window = []

            # --- B. 数据进入检测器 ---
            for i in range(features.shape[0]):
                feat_input = features[i].unsqueeze(0)

                for name, det in detectors.items():
                    val = det.preprocess(feat_input)
                    if det.update(val):
                        alarm_points[name].append(global_batch_idx)
                        det.reset()

            global_batch_idx += 1

    # --- 步骤 4: 寻找差异最大区间并截取绘图 ---
    if not mmd_values:
        print("数据量不足以计算 MMD 窗口，请调小 window_size 或检查数据。")
        return

    mmd_df = pd.DataFrame(mmd_values, columns=['Batch', 'MMD'])
    mmd_df['MMD_Smooth'] = mmd_df['MMD'].rolling(window=3, min_periods=1).mean()

    # 找到 MMD 最大的点，放大特写
    peak_batch = mmd_df.loc[mmd_df['MMD_Smooth'].idxmax(), 'Batch']
    zoom_start = max(0, peak_batch - 100)
    zoom_end = peak_batch + 150

    zoom_mmd = mmd_df[(mmd_df['Batch'] >= zoom_start) & (mmd_df['Batch'] <= zoom_end)]

    # --- 步骤 5: 学术绘图 ---
    plt.figure(figsize=(12, 6))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # 绘制 MMD 曲线与填充面积
    plt.plot(zoom_mmd['Batch'], zoom_mmd['MMD_Smooth'], label='Distribution Shift (MMD)',
             color='#2c3e50', linewidth=2.5, zorder=1)
    plt.fill_between(zoom_mmd['Batch'], zoom_mmd['MMD_Smooth'], color='#34495e', alpha=0.15)

    # 绘制竖线
    colors = {'BNDM': '#e74c3c', 'ADWIN': '#2980b9', 'KSWIN': '#27ae60'}
    linestyles = {'BNDM': '-', 'ADWIN': '--', 'KSWIN': '-.'}
    legend_added = {'BNDM': False, 'ADWIN': False, 'KSWIN': False}

    for name in detectors.keys():
        points_in_zoom = [p for p in alarm_points[name] if zoom_start <= p <= zoom_end]
        for p in points_in_zoom:
            plt.axvline(x=p, color=colors[name], linestyle=linestyles[name],
                        linewidth=2.5, zorder=2,
                        label=f'{name} Detection' if not legend_added[name] else "")
            legend_added[name] = True

            plt.text(p + 2, zoom_mmd['MMD_Smooth'].max() * 0.9, name,
                     color=colors[name], fontsize=10, rotation=90, fontweight='bold')

    plt.title('Comparison of Concept Drift Detection Latency during Severe Distribution Shift',
              fontsize=16, fontweight='bold', pad=15)
    plt.xlabel('Data Stream (Batch Index)', fontsize=14, fontweight='bold')
    plt.ylabel('Distribution Difference (MMD)', fontsize=14, fontweight='bold')

    plt.xlim(zoom_start, zoom_end)
    plt.ylim(0, zoom_mmd['MMD_Smooth'].max() * 1.15)

    plt.legend(loc='upper left', frameon=True, shadow=True, fancybox=True)
    sns.despine()

    output_path = os.path.join(PROJECT_ROOT, "concept_drift_mmd_comparison.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 漂移对比图已保存至: {output_path}")


if __name__ == "__main__":
    main()