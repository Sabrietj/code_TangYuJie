import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
from omegaconf import OmegaConf

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector
from src.concept_drift_detect.run_experiment import MappingWrapper, resolve_path, find_valid_checkpoint


def calculate_mad(data):
    if len(data) == 0: return 0, 0
    median = np.median(data)
    abs_deviation = np.abs(data - median)
    mad = np.median(abs_deviation)
    return median, mad


def analyze_dynamic_threshold(model, dataloader, initial_threshold_log, device='cuda'):
    """
    模拟动态监测过程：检测到漂移 -> 触发 Reset -> 继续监测。
    """
    model.eval()
    model.to(device)

    # 初始化检测器
    config = {
        'seed': 2026,
        'threshold': math.exp(initial_threshold_log),  # 转换为概率阈值
        'max_level': 6,
        'window_size': 1000,
        'alpha_scale': 0.1
    }
    detector = BNDMDetector(config)

    all_log_bfs = []
    drift_points = []

    print(f"正在全量数据上运行动态监测 (Log BF Threshold: {initial_threshold_log})...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Monitoring"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            outputs = model(batch)
            features = outputs['multiview_embeddings']

            for i in range(features.shape[0]):
                val = detector.preprocess(features[i].unsqueeze(0))

                # 1. 记录当前 Log BF (在更新前获取状态)
                if detector.is_initialized:
                    bf = detector._get_total_bf()
                    all_log_bfs.append(bf)
                else:
                    # 初始化阶段没有 BF，填 0 或 NaN
                    all_log_bfs.append(0.0)

                # 2. 执行更新
                is_drift = detector.update(val)

                # 3. 🔴 关键修复：检测到漂移后，必须显式重置！
                if is_drift:
                    drift_points.append(len(all_log_bfs) - 1)
                    detector.reset()  # <--- 加上这一行，让 BF 回升

    return np.array(all_log_bfs), drift_points


def plot_dynamic_analysis(log_bfs, drift_points, initial_th, save_path="dynamic_drift_analysis.png"):
    if len(log_bfs) == 0:
        print("未收集到数据。")
        return

    # 过滤掉初始化阶段的 0 值，只统计有效 BF
    valid_bfs = log_bfs[log_bfs != 0]
    if len(valid_bfs) == 0: valid_bfs = log_bfs

    median, mad = calculate_mad(valid_bfs)

    plt.figure(figsize=(15, 8))

    # 绘制 Log BF 曲线
    plt.plot(log_bfs, label='Log Bayes Factor', color='blue', linewidth=0.6, alpha=0.7)

    # 绘制检测到漂移并“重置”的时刻 (红线)
    for i, pt in enumerate(drift_points):
        plt.axvline(x=pt, color='red', linestyle=':', linewidth=1.0, alpha=0.5,
                    label='Drift Reset' if i == 0 else "")

    # 绘制统计线
    plt.axhline(median, color='green', linestyle='-', label=f'Median: {median:.2f}')
    plt.axhline(initial_th, color='black', linestyle='--', linewidth=2, label=f'Threshold: {initial_th}')

    plt.title(f'Log Bayes Factor Dynamics (Threshold={initial_th})\nRed lines indicate detector RESET')
    plt.xlabel('Sample Sequence')
    plt.ylabel('Log Bayes Factor')
    plt.legend(loc='lower left')
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ 分析图表已保存至: {save_path}")
    print(f"统计结果: 共触发 {len(drift_points)} 次重置。")
    print(f"有效数据统计 - Median: {median:.4f}, MAD: {mad:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, default="flow_bert_multiview_config")
    parser.add_argument("--dataset_name", type=str, default="cic_ids_2017")
    # 🔴 建议默认阈值设低一点，比如 -200，以观察正常震荡
    parser.add_argument("--initial_log_th", type=float, default=-200.0, help="Log BF 阈值")
    args = parser.parse_args()

    # 1. 配置加载与修正
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    config_path = os.path.join(project_root, "src/models/flow_bert_multiview/config")

    cfg = OmegaConf.load(os.path.join(config_path, f"{args.config_name}.yaml"))
    dataset_cfg = OmegaConf.load(os.path.join(config_path, f"datasets/{args.dataset_name}.yaml"))

    # 解决 InterpolationKeyError
    if 'dataset' not in cfg: cfg.dataset = OmegaConf.create()
    cfg.dataset.name = dataset_cfg.name
    cfg.dataset[dataset_cfg.name] = dataset_cfg

    if 'datasets' not in cfg: cfg.datasets = OmegaConf.create()
    cfg.datasets[dataset_cfg.name] = dataset_cfg
    cfg.datasets = OmegaConf.merge(cfg.datasets, dataset_cfg)
    cfg.data = OmegaConf.merge(cfg.data, dataset_cfg)

    # 手动解析路径
    cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
    cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)

    cfg.data.split_mode = "flow"
    if 'sampling' not in cfg.data: cfg.data.sampling = {}
    cfg.data.sampling.random = False

    # 2. 数据准备
    print("正在初始化 DataModule...")
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")

    target_loader = dm.val_dataloader()
    if not target_loader or len(target_loader) == 0:
        target_loader = dm.test_dataloader()

    # 3. 模型加载
    train_ds = dm.train_dataset
    dataset_wrapper = MappingWrapper(train_ds.categorical_val2idx_mappings,
                                     train_ds.categorical_columns_effective)

    ckpt_path = find_valid_checkpoint(project_root)
    print(f"正在加载模型: {ckpt_path}")
    model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)

    # 4. 执行动态分析
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_bfs, drift_points = analyze_dynamic_threshold(model, target_loader, args.initial_log_th, device)

    # 5. 绘图
    plot_dynamic_analysis(log_bfs, drift_points, args.initial_log_th,
                          save_path=os.path.join(project_root, "dynamic_drift_analysis.png"))


if __name__ == "__main__":
    main()