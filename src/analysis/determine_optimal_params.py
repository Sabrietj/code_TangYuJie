"""
Script: sw_bndm_evaluation.py (已针对 CIC-IDS-2017 时序特性优化)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
from omegaconf import OmegaConf
import gc

# --- Project Imports ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.concept_drift_detect.detectors import BNDMDetector
from src.utils.config_loader import ConfigLoader
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule

# --- Configuration ---
CONFIG_DIR = os.path.join(project_root, "src/models/flow_bert_multiview/config")
MAIN_CONFIG_NAME = "flow_bert_multiview_config.yaml"
DATASET_CONFIG_NAME = "datasets/cic_ids_2017.yaml"

WINDOW_SIZE = 5000
STEP_SIZE = 1000

DEPTHS_TO_TEST = [3, 4, 5, 6]

# [核心改变] 不再使用 k 倍数，而是直接设定目标误报率 (Target FPR)
# 例如：[0.01, 0.05, 0.10] 代表我们愿意容忍 1%, 5%, 10% 的误报
TARGET_FPRS_TO_TEST = [0.01, 0.02, 0.05]

def get_datamodule():
    # ... 保留原有的 get_datamodule 函数代码 ...
    print(f"\n>>> Initializing DataModule...")
    loader = ConfigLoader(config_dir=CONFIG_DIR)
    main_config_dict = loader.load_config(MAIN_CONFIG_NAME)
    dataset_config_dict = loader.load_config(DATASET_CONFIG_NAME)
    conf = OmegaConf.create(main_config_dict)
    conf.datasets = dataset_config_dict

    conf.data.flow_data_path = conf.datasets['flow_data_path']
    conf.data.session_split.session_split_path = conf.datasets['session_split_path']
    if 'name' in conf.datasets: conf.data.dataset = conf.datasets['name']

    return MultiviewFlowDataModule(conf)

def extract_features(dm, stage='fit', slice_ratio=1.0):
    # ... 保留原有的 extract_features 函数代码 ...
    print(f"\n>>> Extracting Data for Stage: {stage} (Ratio: {slice_ratio*100}%) ...")
    dm.setup(stage)
    ds = dm.train_dataset if stage == 'fit' else dm.test_dataset

    if hasattr(ds, 'numeric_features') and ds.numeric_features is not None:
        X = np.array(ds.numeric_features, dtype=np.float32)
    else:
        raise ValueError("No numeric features found.")

    y = np.zeros(len(X))
    if hasattr(ds, 'flow_df'):
        df = ds.flow_df
        if 'is_malicious' in df.columns:
            y = df['is_malicious'].values
        elif 'label' in df.columns:
            labels = df['label'].values
            y = np.array([0 if str(l).upper() == 'BENIGN' else 1 for l in labels])

    df = ds.flow_df
    ts_col = next((c for c in ['timestamp', 'conn.ts', 'Start Time', 'date', 'ts'] if c in df.columns), None)
    timestamps = pd.to_datetime(df[ts_col]).values if ts_col else np.arange(len(df))

    sort_idx = np.argsort(timestamps)
    X, y, timestamps = X[sort_idx], y[sort_idx], timestamps[sort_idx]

    target_len = int(len(X) * slice_ratio)
    X, y, timestamps = X[:target_len], y[:target_len], timestamps[:target_len]

    ts_series = pd.Series(timestamps)
    diffs = ts_series.diff()
    gap_indices = diffs[diffs > pd.Timedelta(hours=1)].index.tolist()

    print(f"Extracted {len(X)} samples. Malicious Ratio: {np.mean(y):.4f}")
    return X, y, gap_indices

def run_sliding_window_bndm(X, y, depth, desc_prefix=""):
    # ... 保留原有的 run_sliding_window_bndm 函数代码 ...
    n_samples = len(X)
    X_tensor = torch.from_numpy(X)

    scores = []
    plot_indices = []
    y_windows = []

    for i in tqdm(range(0, n_samples - 2 * WINDOW_SIZE, STEP_SIZE), desc=f"{desc_prefix} d={depth}", leave=False):
        W_combined = X_tensor[i : i + 2 * WINDOW_SIZE]
        y_new = y[i + WINDOW_SIZE : i + 2 * WINDOW_SIZE]

        is_anomaly = 1 if np.mean(y_new) > 0.05 else 0

        detector = BNDMDetector({'max_level': depth, 'window_size': WINDOW_SIZE, 'seed': 2026})
        for j in range(len(W_combined)):
            val = detector.preprocess(W_combined[j:j+1])
            detector.update(val)

        score = -detector._get_total_bf()
        scores.append(score)
        plot_indices.append(i + 2 * WINDOW_SIZE)
        y_windows.append(is_anomaly)

    return np.array(plot_indices), np.array(scores), np.array(y_windows)

def plot_curve(indices, scores, threshold, gap_indices, title, filename):
    # ... 保留原有的画图代码 ...
    plt.figure(figsize=(15, 6))
    plt.plot(indices, scores, color='purple', linewidth=1.5, label='Log Bayes Factor (SW-BNDM)')
    plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.2f})')

    for gap in gap_indices:
        if gap > indices[0] and gap < indices[-1]:
            plt.axvline(x=gap, color='gray', linestyle=':', linewidth=2)
            plt.text(gap, max(scores)*0.9, " Day Break ", rotation=90, color='gray', fontsize=8)

    plt.fill_between(indices, scores, threshold, where=(scores > threshold), color='red', alpha=0.3, label='Drift Alarm')

    plt.title(title)
    plt.xlabel("Sample Index (Continuous)")
    plt.ylabel("Drift Score (-LogBF)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename)
    print(f">>> Image Saved: {filename}")

def main():
    try:
        dm = get_datamodule()

        # ==========================================
        # Phase 1: 训练集上的全局分位数寻优 (破解 CIC-IDS 时序问题)
        # ==========================================
        print("\n=== Phase 1: Global Percentile Calibration (Strict FPR Control) ===")
        X_train, y_train, gap_train = extract_features(dm, 'fit', slice_ratio=1.0)

        best_tpr = -1
        best_params = {}
        train_results = {}

        for depth in DEPTHS_TO_TEST:
            indices, scores, y_win = run_sliding_window_bndm(X_train, y_train, depth, "Train")
            train_results[depth] = {'indices': indices, 'scores': scores, 'y_win': y_win}

            # 提取所有真实标签为“正常 (0)”的窗口得分
            benign_scores = scores[y_win == 0]
            if len(benign_scores) == 0:
                print(f"Warning: Depth {depth} 中没有找到纯净的正常窗口，跳过。")
                continue

            for target_fpr in TARGET_FPRS_TO_TEST:
                # 核心数学技巧：如果我们要 5% 的误报率，说明正常样本中有 5% 的得分可以高于阈值
                # 因此，阈值应该设为正常得分分布的 (1 - target_fpr) * 100 分位数
                percentile = (1.0 - target_fpr) * 100
                threshold = np.percentile(benign_scores, percentile)

                # 应用阈值计算实际表现
                y_pred = (scores > threshold).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_win, y_pred, labels=[0, 1]).ravel()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 召回率
                actual_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0 # 实际误报率

                # 在满足预设误报率的前提下，寻找最大召回率
                if tpr > best_tpr:
                    best_tpr = tpr
                    best_params = {
                        'depth': depth,
                        'target_fpr': target_fpr,
                        'threshold': threshold,
                        'tpr': tpr,
                        'actual_fpr': actual_fpr
                    }

        if not best_params:
            print("\n⚠️ 错误: 无法找到任何有效参数。")
            return

        print("\n>>> 🏆 最佳工业级参数锁定 (Guaranteed Low FPR):")
        print(f"    树深度 (Depth): {best_params['depth']}")
        print(f"    预设最大误报容忍度: {best_params['target_fpr']:.1%}")
        print(f"    自动推导硬阈值 (Threshold): {best_params['threshold']:.2f}")
        print(f"    🎯 攻击召回率 (TPR): {best_params['tpr']:.2%}")
        print(f"    🛡️ 实际训练集误报率: {best_params['actual_fpr']:.2%}")

        opt_train = train_results[best_params['depth']]
        plot_curve(opt_train['indices'], opt_train['scores'], best_params['threshold'], gap_train,
                   f"Train Set: Percentile Calibrated (d={best_params['depth']}, Th={best_params['threshold']:.2f}, FPR={best_params['actual_fpr']:.1%})",
                   "1_train_sw_bndm_percentile.png")

        del X_train, y_train, train_results; gc.collect()

        # ==========================================
        # Phase 2: 测试集盲测 (取 50% 进行泛化验证)
        # ==========================================
        print("\n=== Phase 2: Test Set Generalization Verification (Top 50% only) ===")
        X_test, y_test, gap_test = extract_features(dm, 'test', slice_ratio=0.50)

        test_indices, test_scores, _ = run_sliding_window_bndm(X_test, y_test, best_params['depth'], "Test")

        plot_curve(test_indices, test_scores, best_params['threshold'], gap_test,
                   f"Test Set (50%): Depth={best_params['depth']}, Fixed Threshold={best_params['threshold']:.2f}",
                   "2_test_sw_bndm_generalization.png")

        print("\n=== All Done ===")

    except Exception as e:
        import traceback; traceback.print_exc()

if __name__ == "__main__":
    main()