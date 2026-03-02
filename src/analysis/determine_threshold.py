import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import math
from tqdm import tqdm
from omegaconf import OmegaConf

# 添加项目根目录到 Path，确保可以导入 src 模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector
from src.concept_drift_detect.run_experiment import MappingWrapper, resolve_path, find_valid_checkpoint


def calculate_mad(data):
    """计算绝对中位差 (Median Absolute Deviation)"""
    if len(data) == 0:
        return 0, 0
    median = np.median(data)
    abs_deviation = np.abs(data - median)
    mad = np.median(abs_deviation)
    return median, mad


def analyze_threshold(model, dataloader, device='cuda'):
    """
    运行 BNDM 检测器，记录 Log Bayes Factor 的变化，不触发重置。
    """
    model.eval()
    if torch.cuda.is_available():
        model.to(device)
    else:
        device = 'cpu'
        model.to(device)

    # 初始化 BNDM 检测器
    config = {
        'seed': 2026,
        'threshold': 1e-10,
        'max_level': 6,
        'window_size': 1000,
        'alpha_scale': 0.1
    }
    detector = BNDMDetector(config)

    # 🔴 核心技巧：将阈值设为负无穷，确保永远不会触发 detector.reset()
    detector.threshold = -float('inf')

    log_bfs = []
    processed_count = 0

    print("正在分析 Log Bayes Factor 分布...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Analyzing", unit="batch"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 1. 模型推理获取特征
            try:
                outputs = model(batch)
                features = outputs['multiview_embeddings']  # (Batch, Dim)
            except Exception as e:
                print(f"❌ 模型推理出错 (可能是输入特征缺失): {e}")
                continue

            # 2. 逐样本更新检测器
            batch_size = features.shape[0]
            for i in range(batch_size):
                feat = features[i].unsqueeze(0)

                # 预处理 (投影 + 归一化)
                val = detector.preprocess(feat)

                # 更新检测器
                _ = detector.update(val)
                processed_count += 1

                # 记录当前的 Log Bayes Factor
                # 注意：我们需要等待 warm-up 阶段过后数据才稳定 (参考窗口填满后)
                if detector.is_initialized and processed_count > config['window_size']:
                    bf = detector._get_total_bf()
                    if not math.isnan(bf) and not math.isinf(bf):
                        log_bfs.append(bf)

    return np.array(log_bfs)


def plot_analysis(log_bfs, save_path="threshold_analysis.png"):
    """绘制 Log BF 趋势图和直方图，并标记建议阈值"""
    if len(log_bfs) == 0:
        print("❌ 没有收集到足够的 Log BF 数据（可能是数据量太少，未通过 Warm-up 阶段）。")
        return

    # 计算统计量
    median, mad = calculate_mad(log_bfs)

    # 根据 MAD 原则计算阈值
    k_values = [3, 5, 10]
    thresholds = {k: median - k * mad for k in k_values}

    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # 子图 1: 时序变化趋势
    ax1.plot(log_bfs, label='Log Bayes Factor', color='blue', alpha=0.6, linewidth=0.5)
    ax1.axhline(median, color='green', linestyle='--', label=f'Median ({median:.2f})')

    colors = ['orange', 'red', 'purple']
    for i, k in enumerate(k_values):
        th = thresholds[k]
        ax1.axhline(th, color=colors[i], linestyle='--', label=f'Threshold (k={k}): {th:.2f}')

    ax1.set_title('Log Bayes Factor Trend over Time (Stable Stream)')
    ax1.set_xlabel('Sample Index (after warm-up)')
    ax1.set_ylabel('Log Bayes Factor')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 子图 2: 分布直方图
    ax2.hist(log_bfs, bins=100, color='skyblue', edgecolor='black', alpha=0.7, density=True)
    ax2.axvline(median, color='green', linestyle='--', label='Median')
    for i, k in enumerate(k_values):
        th = thresholds[k]
        ax2.axvline(th, color=colors[i], linestyle='--', label=f'k={k}')

    ax2.set_title('Distribution of Log Bayes Factor')
    ax2.set_xlabel('Log Bayes Factor')
    ax2.set_ylabel('Density')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"\n✅ 分析图表已保存至: {save_path}")

    print("\n" + "=" * 60)
    print("📊 推荐阈值设置参考 (基于 Median - k * MAD):")
    print(f"Median Log BF: {median:.4f}")
    print(f"MAD: {mad:.4f}")
    print("-" * 40)
    for k in k_values:
        val = thresholds[k]
        # 配置文件中 BNDMDetector 会对 threshold 取 log
        # self.threshold = math.log(config.get('threshold', 0.05))
        # 因此，如果我们希望 Log BF 的阈值是 val，那么 Config 中的值应该是 exp(val)
        prob_val = np.exp(val)
        print(f"k = {k:2d} | 建议 Log Threshold: {val:.4f} | 对应 Config Threshold (填入yaml): {prob_val:.4e}")
    print("=" * 60)
    print("💡 提示: ")
    print("1. 较小的 k (如 k=3) -> 更敏感 (More Drifts)")
    print("2. 较大的 k (如 k=10) -> 更稳健 (Less False Alarms)")
    print("3. 请将 'Config Threshold' 的值复制到 run_experiment.py 或配置文件中。")


def main():
    parser = argparse.ArgumentParser(description="Determine BNDM Drift Threshold using MAD")
    parser.add_argument("--config_name", type=str, default="flow_bert_multiview_config", help="Config name")
    parser.add_argument("--dataset_name", type=str, default="cic_ids_2017", help="Dataset name")
    args = parser.parse_args()

    # 1. 加载配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))

    config_path = os.path.join(project_root, "src/models/flow_bert_multiview/config")

    # 加载主配置
    cfg = OmegaConf.load(os.path.join(config_path, f"{args.config_name}.yaml"))

    # 加载数据集配置
    dataset_yaml_path = os.path.join(config_path, f"datasets/{args.dataset_name}.yaml")
    if os.path.exists(dataset_yaml_path):
        print(f"Loading dataset config from: {dataset_yaml_path}")
        dataset_cfg = OmegaConf.load(dataset_yaml_path)

        # =========================================================================
        # 🔴 核心修正：构建 cfg.datasets.labels 结构
        # =========================================================================
        if 'datasets' not in cfg:
            cfg.datasets = OmegaConf.create()

        # 将整个 dataset_cfg 挂载到 cfg.datasets 下
        # 这样 cfg.datasets.labels 就能被访问到了
        # 同时保留 dataset_cfg 中的其他字段 (如 flow_data_path)
        cfg.datasets = OmegaConf.merge(cfg.datasets, dataset_cfg)

        # 为了兼容性，也可以将 dataset_cfg 的内容直接 merge 到 data 下 (旧逻辑)
        if 'data' not in cfg:
            cfg.data = OmegaConf.create()
        cfg.data = OmegaConf.merge(cfg.data, dataset_cfg)

        # 解析路径
        if 'flow_data_path' in dataset_cfg:
            cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
        if 'session_split_path' in dataset_cfg:
            cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)

        cfg.data.dataset = dataset_cfg.name

    else:
        print(f"Dataset config not found: {dataset_yaml_path}")
        return

    # =========================================================================
    # 🔴 关键逻辑 1: 确保使用包含特征的 all_embedded_flow.csv
    # =========================================================================
    original_path = cfg.data.flow_data_path
    if "all_flow.csv" in original_path:
        # 尝试切换到 embedded 文件
        target_path = original_path.replace("all_flow.csv", "all_embedded_flow.csv")
        if os.path.exists(target_path):
            print(f"🔴 [Auto-Correction] 检测到 all_flow.csv (缺特征)，自动切换为: {target_path}")
            cfg.data.flow_data_path = target_path
        else:
            print(f"⚠️ [Warning] 无法找到 all_embedded_flow.csv，继续使用: {original_path}")
            print("   可能会因缺少 ssl.server_name*_freq 等列而报错！")

    # =========================================================================
    # 🔴 关键逻辑 2: 强制按顺序读取 (不随机打乱)
    # =========================================================================
    print("🔴 [Config] 强制设置 split_mode = 'flow' (时序模式)")
    cfg.data.split_mode = "flow"

    print("🔴 [Config] 强制禁用随机采样 (random = False)")
    if 'sampling' not in cfg.data:
        cfg.data.sampling = {}
    cfg.data.sampling.random = False

    # 2. 初始化数据模块
    print("正在初始化数据加载器...")
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")

    # 使用验证集进行阈值确定 (位于训练集之后，适合测试)
    target_loader = dm.val_dataloader()
    if not target_loader:
        print("验证集加载失败，尝试使用测试集...")
        target_loader = dm.test_dataloader()

    # 获取映射 (用于模型初始化)
    if hasattr(dm, 'train_dataset') and dm.train_dataset:
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)
    else:
        # 兜底初始化
        _ = dm.train_dataloader()
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)

    dataset_wrapper = MappingWrapper(mappings, effective_columns)

    # 3. 加载模型
    ckpt_path = find_valid_checkpoint(project_root)
    if not ckpt_path:
        print("❌ 错误: 未找到有效的模型检查点 (Checkpoint)！请先运行训练。")
        return

    print(f"正在加载模型: {ckpt_path}")
    try:
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
    except Exception as e:
        print(f"❌ 加载模型失败: {e}")
        return

    # 4. 运行分析
    log_bfs = analyze_threshold(model, target_loader)

    # 5. 绘图和输出
    plot_analysis(log_bfs, save_path=os.path.join(project_root, "drift_threshold_analysis.png"))


if __name__ == "__main__":
    main()