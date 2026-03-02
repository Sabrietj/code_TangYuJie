import torch
import numpy as np
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Dict, List, Optional

try:
    from river import drift

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False


# ==========================================
# 核心数学工具：只保留 log_beta，彻底删除 log_binomial 污染
# ==========================================
def log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


# ==========================================
# 检测器基类
# ==========================================
class BaseDriftDetector(ABC):
    def __init__(self, config: Dict):
        self.config = config
        self.total_samples = 0

        self.projection_matrix = None
        self.running_mean = 0.0
        self.running_var = 1.0
        self.stats_frozen = False

    def preprocess(self, features: torch.Tensor) -> float:
        device = features.device
        if features.dim() == 1:
            features = features.unsqueeze(0)

        feature_dim = features.shape[-1]

        if self.projection_matrix is None:
            g_cpu = torch.Generator()
            g_cpu.manual_seed(self.config.get('seed', 2026))
            self.projection_matrix = torch.randn(feature_dim, 1, generator=g_cpu).to(device)
            self.projection_matrix = self.projection_matrix / torch.norm(self.projection_matrix)
        elif self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        val = torch.matmul(features, self.projection_matrix).item()

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
        self.n_ref_L, self.n_ref_R = 0, 0
        self.n_cur_L, self.n_cur_R = 0, 0
        self.cached_log_B = 0.0
        self.left, self.right = None, None

    def compute_log_bayes_factor(self):
        n_ref = self.n_ref_L + self.n_ref_R
        n_cur = self.n_cur_L + self.n_cur_R

        if n_ref == 0 and n_cur == 0:
            self.cached_log_B = 0.0
            return 0.0

        n_tot_L = self.n_ref_L + self.n_cur_L
        n_tot_R = self.n_ref_R + self.n_cur_R

        # 🚀 【核心修复】：纯净版的 Polya Tree 边际似然，无多项式偏移
        log_ev_H0 = log_beta(self.alpha + n_tot_L, self.alpha + n_tot_R) - log_beta(self.alpha, self.alpha)
        log_ev_H1_ref = log_beta(self.alpha + self.n_ref_L, self.alpha + self.n_ref_R) - log_beta(self.alpha,
                                                                                                  self.alpha)
        log_ev_H1_cur = log_beta(self.alpha + self.n_cur_L, self.alpha + self.n_cur_R) - log_beta(self.alpha,
                                                                                                  self.alpha)

        self.cached_log_B = log_ev_H0 - (log_ev_H1_ref + log_ev_H1_cur)
        return self.cached_log_B


class BNDMDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.max_level = config.get('max_level', 6)
        self.alpha_scale = config.get('alpha_scale', 0.1)
        self.window_size = config.get('window_size', 1000)

        # 将配置的 0.01 转化为正确的自然对数阈值 (-4.6)
        threshold_val = config.get('threshold', 0.01)
        if threshold_val <= 0: threshold_val = 0.01
        self.threshold = math.log(threshold_val)

        self.ref_window = deque(maxlen=self.window_size)
        self.cur_window = deque(maxlen=self.window_size)
        self.root = self._build_tree(0)

        from scipy.stats import norm
        self.norm_cdf = norm.cdf
        self.is_initialized = False

    def _build_tree(self, level):
        alpha = self.alpha_scale * ((level + 1) ** 2)
        node = PTNode(level, alpha)
        if level < self.max_level:
            node.left = self._build_tree(level + 1)
            node.right = self._build_tree(level + 1)
        return node

    def _update_tree(self, val, window_type, delta=1):
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
                node.compute_log_bayes_factor()
                node = node.left
                high = mid
            else:
                if window_type == 'ref':
                    node.n_ref_R += delta
                else:
                    node.n_cur_R += delta
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
        if not self.is_initialized:
            self.cur_window.append(val)
            if len(self.cur_window) >= self.window_size:
                for v in self.cur_window:
                    self.ref_window.append(v)
                    self._update_tree(v, 'ref', 1)
                self.freeze_stats()
                self.cur_window.clear()
                self.is_initialized = True
            return False

        if len(self.cur_window) == self.window_size:
            old_val = self.cur_window.popleft()
            self._update_tree(old_val, 'cur', -1)

        self.cur_window.append(val)
        self._update_tree(val, 'cur', 1)

        if len(self.cur_window) < 50:
            return False

        log_bf = self._get_total_bf()
        return log_bf < self.threshold

    def reset(self):
        self.root = self._build_tree(0)
        self.ref_window.clear()

        for v in self.cur_window:
            self.ref_window.append(v)
            self._update_tree(v, 'ref', 1)

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