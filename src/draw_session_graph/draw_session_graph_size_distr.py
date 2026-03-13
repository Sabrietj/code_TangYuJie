# -*- coding: utf-8 -*-
import os, sys, json, argparse
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs

# 添加 ../utils 到路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
import config_manager as ConfigManager

import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetAnalyzer:
    def __init__(self, bin_path=None):
        self.bin_path = bin_path
        self.graphs = []
        if bin_path and os.path.exists(bin_path):
            self.graphs, _ = load_graphs(bin_path)

    def get_node_count_distribution(self):
        node_counts = [g.num_nodes() for g in self.graphs]
        unique, counts = np.unique(node_counts, return_counts=True)
        return dict(zip(unique, counts)), len(self.graphs)

    def get_edge_count_distribution(self):
        edge_counts = [g.num_edges() for g in self.graphs]
        unique, counts = np.unique(edge_counts, return_counts=True)
        return dict(zip(unique, counts)), len(self.graphs)

    @staticmethod
    def parse_distribution_string(distribution_str):
        lines = distribution_str.strip().split('\n')
        counts_dict = {}
        total_graphs = 0
        for line in lines:
            if line.startswith("Total Graphs:"):
                total_graphs = int(line.split(":")[1])
            elif line and line.split()[0].isdigit():
                parts = line.split()
                counts_dict[int(parts[0])] = int(parts[1])
        return counts_dict, total_graphs

    def load_or_compute_distribution(self, json_file, dataset_name, use_cache=True, data_type="nodes"):
        if use_cache and os.path.exists(json_file):
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            key = f"{dataset_name}_{data_type}"
            if key in data:
                return self.parse_distribution_string(data[key])

        if data_type == "nodes":
            count_dict, total_graphs = self.get_node_count_distribution()
        else:
            count_dict, total_graphs = self.get_edge_count_distribution()
        return count_dict, total_graphs


def plot_combined_hist_and_cdf(dataset_name, node_counts, edge_counts, total_graphs, max_x=200, bin_width=5):
    """绘制完全复刻版的双Y轴图表：直方图（频率）+ 折线图（累计分布函数 CDF），完全不过滤数据"""

    # 设置中文字体，优先使用常见中文字体，防止出现方块
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Songti SC', 'STFangsong', 'Microsoft YaHei', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(10, 6))

    node_data = []
    edge_data = []

    # 展开所有数据，不做任何过滤
    for k, v in node_counts.items():
        if k <= max_x:
            node_data.extend([k] * v)

    for k, v in edge_counts.items():
        if k <= max_x:
            edge_data.extend([k] * v)

    if not node_data: node_data = [0]
    if not edge_data: edge_data = [0]

    # 定义 X 轴的区间划分 (Bins)，例如 0-5, 5-10...
    bins = np.arange(0, max_x + bin_width, bin_width)

    # 计算权重，使其转换为百分比 (%)，严格使用真实的 total_graphs
    node_weights = np.ones(len(node_data)) / total_graphs * 100
    edge_weights = np.ones(len(edge_data)) / total_graphs * 100

    # ========== 左侧 Y 轴 (ax1) - 频率直方图 ==========
    # 传入列表的列表，实现并排显示
    ax1.hist([node_data, edge_data], bins=bins, weights=[node_weights, edge_weights],
             color=['#cedb6e', '#d2cdc9'], edgecolor='#555555', alpha=0.85,
             label=['节点直方图', '边直方图'], zorder=2)

    ax1.set_xlabel('节点/边数量', fontsize=12)
    ax1.set_ylabel('频率 (%)', fontsize=12)
    ax1.set_xlim(-2, max_x)

    # 因为不过滤数据，第一根柱子必然极高，左轴上限固定到 105% 保证能装下所有数据
    ax1.set_ylim(0, 105)
    ax1.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)

    # ========== 右侧 Y 轴 (ax2) - 累计分布函数 CDF ==========
    ax2 = ax1.twinx()

    node_hist, _ = np.histogram(node_data, bins=bins)
    edge_hist, _ = np.histogram(edge_data, bins=bins)

    node_cdf = np.cumsum(node_hist) / total_graphs * 100
    edge_cdf = np.cumsum(edge_hist) / total_graphs * 100

    # CDF 的 X 坐标取每个 bin 的右边缘
    cdf_x = bins[1:]

    ax2.plot(cdf_x, node_cdf, color='#5bb960', linestyle='--', marker='.', markersize=8, alpha=0.9,
             label='节点累计分布函数')
    ax2.plot(cdf_x, edge_cdf, color='#5899da', linestyle='--', marker='+', markersize=8, alpha=0.9,
             label='边累计分布函数')

    ax2.set_ylabel('累计分布 (%)', fontsize=12)
    ax2.set_ylim(15, 105)

    ax2.legend(loc='center right', frameon=True, edgecolor='black', fancybox=False)

    # 边框加黑加粗，复刻参考图风格
    for ax in [ax1, ax2]:
        for spine in ax.spines.values():
            spine.set_linewidth(1.5)
            spine.set_color('black')

    plt.tight_layout()
    return plt


def main():
    parser = argparse.ArgumentParser(description='Draw session graph node and edge number distribution.')
    parser.add_argument('--plot-style', '-p', choices=['loglog_dot', 'normal_bar', 'both', 'combined_cdf'],
                        default='combined_cdf')
    parser.add_argument('--maxshow', type=int, default=200, help='Max count for X axis')
    parser.add_argument("--graph_file_name", type=str, default="all_session_graph")

    args = parser.parse_args()

    dataset_dir = ConfigManager.read_plot_data_path_config()
    concurrent_flow_iat_threshold = ConfigManager.read_concurrent_flow_iat_threshold()
    sequential_flow_iat_threshold = ConfigManager.read_sequential_flow_iat_threshold()

    dataset_name = os.path.basename(dataset_dir.rstrip("/\\"))
    graph_file_path = os.path.join(dataset_dir, f"{args.graph_file_name}.bin")

    analyzer = DatasetAnalyzer(graph_file_path)
    plot_data_path = ConfigManager.read_plot_data_path_config()
    json_file = os.path.join(plot_data_path,
                             f'session_graph_size_distr_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.json')

    figs_dir = os.path.join('.', 'figs', dataset_name)
    os.makedirs(figs_dir, exist_ok=True)

    if args.plot_style == 'combined_cdf':
        node_counts, total_graphs_n = analyzer.load_or_compute_distribution(json_file, dataset_name, True, "nodes")
        edge_counts, total_graphs_e = analyzer.load_or_compute_distribution(json_file, dataset_name, True, "edges")

        plt_combined = plot_combined_hist_and_cdf(
            dataset_name, node_counts, edge_counts, total_graphs_n,
            max_x=args.maxshow, bin_width=5
        )

        fig_path = os.path.join(figs_dir,
                                f'session_graph_combined_cdf_{concurrent_flow_iat_threshold}_{sequential_flow_iat_threshold}.pdf')
        plt_combined.savefig(fig_path, dpi=300, bbox_inches='tight')
        logger.info(f"图表已成功保存至: {fig_path}")


if __name__ == "__main__":
    main()