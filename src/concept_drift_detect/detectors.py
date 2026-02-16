import torch
import numpy as np
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional

# 尝试导入 river 用于对比实验
try:
    from river import drift

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


# ==========================================
# 核心数学工具
# ==========================================
def log_beta(a, b):
    """计算 log(Beta(a, b))"""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# 🔴 [Removed] log_binomial 函数已移除，因为 BNDM 处理序列似然不需要组合数


# ==========================================
# 检测器基类
# ==========================================
class BaseDriftDetector(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.total_samples = 0

        # 预处理：随机投影矩阵
        self.projection_matrix = None
        self.running_mean = 0.0
        self.running_var = 1.0

        # 是否冻结统计量 (用于固定参考系)
        self.stats_frozen = False

    def preprocess(self, features: torch.Tensor) -> float:
        """
        特征降维 + 在线归一化
        Args:
            features: Tensor of shape (1, Dim) or (Dim,)
        Returns:
            Scalar float value (projected and normalized)
        """
        device = features.device

        # 确保输入是 (1, Dim)
        if features.dim() == 1:
            features = features.unsqueeze(0)

        # 获取特征维度
        feature_dim = features.shape[-1]

        # 初始化投影矩阵 (Dim, 1)
        if self.projection_matrix is None:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.get('seed', 2026))

            # 创建 (Feature_Dim, 1) 的矩阵
            self.projection_matrix = torch.randn(feature_dim, 1, generator=g_cpu).to(device)
            self.projection_matrix = self.projection_matrix / torch.norm(self.projection_matrix)
        elif self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        # 投影: (1, Dim) x (Dim, 1) -> (1, 1)
        val = torch.matmul(features, self.projection_matrix).item()

        # 仅在未冻结时更新统计量，防止漂移被在线归一化掩盖
        if not self.stats_frozen:
            self.total_samples += 1
            delta = val - self.running_mean
            self.running_mean += delta / self.total_samples
            delta2 = val - self.running_mean
            self.running_var += delta * delta2

        if self.total_samples < 2:
            return 0.0

        std = math.sqrt(self.running_var / (self.total_samples - 1)) + 1e-8
        return (val - self.running_mean) / std

    def freeze_stats(self):
        """冻结归一化参数 (Reference Window 确定后调用)"""
        self.stats_frozen = True

    @abstractmethod
    def update(self, val: float) -> bool:
        pass

    @abstractmethod
    def reset(self):
        pass


# ==========================================
# BNDM (Polya Tree) 实现
# ==========================================
class PTNode:
    def __init__(self, level, alpha):
        self.level = level
        self.alpha = alpha
        self.n_ref_L = 0
        self.n_ref_R = 0
        self.n_cur_L = 0
        self.n_cur_R = 0
        self.cached_log_B = 0.0
        self.left = None
        self.right = None

    def compute_log_bayes_factor(self):
        """
        计算 Log Bayes Factor.
        注意：此处比较的是序列数据的生成概率，因此不包含二项式组合系数。
        """
        n_ref = self.n_ref_L + self.n_ref_R
        n_cur = self.n_cur_L + self.n_cur_R

        # 边界情况
        if n_ref == 0 and n_cur == 0:
            self.cached_log_B = 0.0
            return 0.0

        n_tot_L = self.n_ref_L + self.n_cur_L
        n_tot_R = self.n_ref_R + self.n_cur_R

        # H0: 同分布 (Combined)
        # Log Probability = Log Beta(alpha + n_L, alpha + n_R) - Log Beta(alpha, alpha)
        log_ev_H0 = log_beta(self.alpha + n_tot_L, self.alpha + n_tot_R) - \
                    log_beta(self.alpha, self.alpha)

        # H1: 不同分布 (Separate)
        # Log Probability = (Log Beta_Ref + Log Beta_Cur)
        log_ev_H1_ref = log_beta(self.alpha + self.n_ref_L, self.alpha + self.n_ref_R) - \
                        log_beta(self.alpha, self.alpha)

        log_ev_H1_cur = log_beta(self.alpha + self.n_cur_L, self.alpha + self.n_cur_R) - \
                        log_beta(self.alpha, self.alpha)

        # Log Bayes Factor = Log P(H0) - Log P(H1)
        # 负值越小，越倾向于 H1 (Drift)
        self.cached_log_B = log_ev_H0 - (log_ev_H1_ref + log_ev_H1_cur)
        return self.cached_log_B


class BNDMDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_level = config.get('max_level', 5)
        self.alpha_scale = config.get('alpha_scale', 0.1)
        self.window_size = config.get('window_size', 1000)
        # BNDM 使用 log Bayes Factor
        # 如果 threshold = 0.05, math.log(0.05) ≈ -3.0
        self.threshold = math.log(config.get('threshold', 0.05))

        self.ref_window = deque(maxlen=self.window_size)
        self.cur_window = deque(maxlen=self.window_size)
        self.root = self._build_tree(0)

        from scipy.stats import norm
        self.norm_cdf = norm.cdf
        self.is_initialized = False

    def _build_tree(self, level):
        # 按照 BNDM 论文建议，alpha 随深度增加，以保持不同层级的影响力平衡
        alpha = self.alpha_scale * ((level + 1) ** 2)
        node = PTNode(level, alpha)
        if level < self.max_level:
            node.left = self._build_tree(level + 1)
            node.right = self._build_tree(level + 1)
        return node

    def _update_tree(self, val, window_type, delta=1):
        # 将标准化后的值映射到 [0, 1] 区间
        cdf = self.norm_cdf(val)
        node = self.root
        low, high = 0.0, 1.0

        for _ in range(self.max_level):
            mid = (low + high) / 2
            if cdf < mid:
                if window_type == 'ref':
                    node.n_ref_L += delta
                else:
                    node.n_cur_L += delta

                # 实时更新当前节点 BF
                node.compute_log_bayes_factor()
                node = node.left
                high = mid
            else:
                if window_type == 'ref':
                    node.n_ref_R += delta
                else:
                    node.n_cur_R += delta

                # 实时更新当前节点 BF
                node.compute_log_bayes_factor()
                node = node.right
                low = mid

    def _get_total_bf(self):
        total = 0.0
        q = [self.root]
        while q:
            node = q.pop(0)
            total += node.cached_log_B
            if node.left: q.append(node.left)
            if node.right: q.append(node.right)
        return total

    def update(self, val: float) -> bool:
        # 1. 初始化阶段 (填充 Reference Window)
        if not self.is_initialized:
            self.cur_window.append(val)
            if len(self.cur_window) >= self.window_size:
                # 将收集到的数据作为 Reference
                for v in self.cur_window:
                    self.ref_window.append(v)
                    self._update_tree(v, 'ref', 1)

                # 初始化完成后，冻结归一化统计量
                # 这确保后续的漂移数据在坐标系中呈现偏移，而不是被归一化消除
                self.freeze_stats()

                # 清空当前窗口，准备开始监测
                self.cur_window.clear()
                self.is_initialized = True
            return False

        # 2. 滑动窗口维护
        if len(self.cur_window) == self.window_size:
            old_val = self.cur_window.popleft()
            self._update_tree(old_val, 'cur', -1)

        self.cur_window.append(val)
        self._update_tree(val, 'cur', 1)

        # Warm-up: 如果当前窗口样本太少，统计量不稳定，跳过检测
        if len(self.cur_window) < 50:
            return False

        log_bf = self._get_total_bf()
        return log_bf < self.threshold

    def reset(self):
        self.root = self._build_tree(0)
        self.ref_window.clear()

        # 将当前的滑动窗口作为新的参考窗口
        for v in self.cur_window:
            self.ref_window.append(v)
            self._update_tree(v, 'ref', 1)

        # 保持统计量冻结状态 (使用当前的坐标系继续监测)
        self.freeze_stats()

        self.cur_window.clear()


class ADWINDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not RIVER_AVAILABLE:
            raise ImportError("River library required for ADWIN")
        self.adwin = drift.ADWIN(delta=config.get('delta', 0.002))

    def update(self, val: float) -> bool:
        self.adwin.update(val)
        return self.adwin.drift_detected

    def reset(self):
        self.adwin = drift.ADWIN(delta=self.config.get('delta', 0.002))