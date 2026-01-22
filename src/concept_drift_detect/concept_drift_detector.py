import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import math
from abc import ABC, abstractmethod
from scipy.special import betaln
from scipy.stats import norm

# 尝试导入 river，如果不存在则报错或降级
try:
    from river import drift

    RIVER_AVAILABLE = True
except ImportError:
    RIVER_AVAILABLE = False

logger = logging.getLogger(__name__)


# ==========================================
# 1. 抽象基类 (Interface)
# ==========================================

class BaseDriftDetector(ABC):
    """概念漂移检测器抽象基类"""

    def __init__(self, config: Dict):
        self.config = config
        self.drift_count = 0
        self.total_samples = 0
        self.is_initialized = False

        # 特征预处理状态
        self.running_mean = 0.0
        self.running_var = 1.0
        self.projection_matrix = None

    def preprocess(self, features: torch.Tensor) -> List[float]:
        """
        通用预处理：随机投影 -> 在线归一化 -> 转标量列表
        """
        batch_size = features.shape[0]
        feature_dim = features.shape[1]
        device = features.device

        # 初始化投影矩阵
        if self.projection_matrix is None:
            self.projection_matrix = torch.randn(feature_dim, 1).to(device)
            self.projection_matrix = self.projection_matrix / torch.norm(self.projection_matrix)
        elif self.projection_matrix.device != device:
            self.projection_matrix = self.projection_matrix.to(device)

        # 1. 投影
        projected = torch.matmul(features, self.projection_matrix).squeeze(-1)  # [B]

        z_vals = []
        for val in projected:
            val_item = val.item()
            self.total_samples += 1

            # 2. 在线更新 Mean/Var
            delta = val_item - self.running_mean
            self.running_mean += delta / self.total_samples
            delta2 = val_item - self.running_mean
            self.running_var += delta * delta2

            # 3. 归一化 (Z-Score)
            if self.total_samples < 2:
                z = 0.0
            else:
                std = math.sqrt(self.running_var / (self.total_samples - 1)) + 1e-8
                z = (val_item - self.running_mean) / std

            z_vals.append(z)

        return z_vals

    @abstractmethod
    def update(self, val: float) -> bool:
        """更新单个样本，返回是否漂移"""
        pass

    @abstractmethod
    def get_info(self) -> Dict:
        """返回当前状态信息（如 BayesFactor 或 Window Size）"""
        pass

    @abstractmethod
    def get_drift_evidence(self) -> str:
        """获取漂移证据描述（用于日志）"""
        pass

    @abstractmethod
    def reset(self):
        """重置检测器状态"""
        pass


class PTNode:
    """Polya Tree 节点 (保持不变)"""

    def __init__(self, level: int, code: str, alpha_val: float):
        self.level = level
        self.code = code
        self.alpha_0 = alpha_val
        self.alpha_1 = alpha_val
        self.n_ref_0 = 0
        self.n_ref_1 = 0
        self.n_cur_0 = 0
        self.n_cur_1 = 0
        self.cached_log_B_s = 0.0
        self.left: Optional['PTNode'] = None
        self.right: Optional['PTNode'] = None

    def update_counts(self, side: int, window_type: str, delta: int = 1):
        if window_type == 'ref':
            if side == 0:
                self.n_ref_0 += delta
            else:
                self.n_ref_1 += delta
        else:
            if side == 0:
                self.n_cur_0 += delta
            else:
                self.n_cur_1 += delta

    def compute_log_local_bayes_factor(self) -> float:
        N_tot_0 = self.n_ref_0 + self.n_cur_0
        N_tot_1 = self.n_ref_1 + self.n_cur_1
        a0 = self.alpha_0
        a1 = self.alpha_1
        log_prob_H0 = betaln(a0 + N_tot_0, a1 + N_tot_1) - betaln(a0, a1)
        log_prob_H1_ref = betaln(a0 + self.n_ref_0, a1 + self.n_ref_1) - betaln(a0, a1)
        log_prob_H1_cur = betaln(a0 + self.n_cur_0, a1 + self.n_cur_1) - betaln(a0, a1)
        self.cached_log_B_s = log_prob_H0 - (log_prob_H1_ref + log_prob_H1_cur)
        return self.cached_log_B_s

    def get_counts_info(self) -> str:
        return f"Ref[L:{self.n_ref_0}, R:{self.n_ref_1}] vs Cur[L:{self.n_cur_0}, R:{self.n_cur_1}]"


class PolyaTree:
    """Polya Tree 结构 (保持不变)"""

    def __init__(self, max_level: int = 4, c: float = 1.0):
        self.max_level = max_level
        self.c = c
        self.root = self._build_tree(0, "")

    def _build_tree(self, level: int, code: str) -> PTNode:
        alpha = self.c * ((level + 1) ** 2)
        node = PTNode(level, code, alpha)
        if level < self.max_level:
            node.left = self._build_tree(level + 1, code + "0")
            node.right = self._build_tree(level + 1, code + "1")
        return node

    def _get_direction(self, val: float, level: int, code: str) -> int:
        val_int = 0
        if len(code) > 0: val_int = int(code, 2)
        total_intervals = 2 ** level
        step = 1.0 / total_intervals
        start_p = val_int * step
        mid_p = start_p + step / 2.0
        cut_point = norm.ppf(mid_p)
        return 0 if val < cut_point else 1

    def update(self, val: float, window_type: str, delta: int = 1):
        curr = self.root
        curr_code = ""
        for level in range(self.max_level + 1):
            if curr is None: break
            direction = self._get_direction(val, level, curr_code)
            curr.update_counts(direction, window_type, delta)
            curr.compute_log_local_bayes_factor()
            if direction == 0:
                curr = curr.left
                curr_code += "0"
            else:
                curr = curr.right
                curr_code += "1"

    def compute_total_bayes_factor(self) -> float:
        total_log_B = 0.0
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            total_log_B += node.cached_log_B_s
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        return total_log_B

    def diagnose_drift(self, top_k: int = 3) -> List[Dict]:
        all_nodes = []
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            total_count = node.n_ref_0 + node.n_ref_1 + node.n_cur_0 + node.n_cur_1
            if total_count > 0: all_nodes.append(node)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        sorted_nodes = sorted(all_nodes, key=lambda x: x.cached_log_B_s)
        diagnosis = []
        for node in sorted_nodes[:top_k]:
            diagnosis.append({
                "level": node.level,
                "code": node.code if node.code else "ROOT",
                "log_B_s": node.cached_log_B_s,
                "ref_counts": (node.n_ref_0, node.n_ref_1),
                "cur_counts": (node.n_cur_0, node.n_cur_1),
                "description": node.get_counts_info()
            })
        return diagnosis

    def reset_counts(self):
        queue = deque([self.root])
        while queue:
            node = queue.popleft()
            node.n_ref_0 = 0;
            node.n_ref_1 = 0;
            node.n_cur_0 = 0;
            node.n_cur_1 = 0
            node.cached_log_B_s = 0.0
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)


# --- BNDM Detector ---
class BNDMDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.params = config.get('detectors', {}).get('bndm', {})

        self.pt = PolyaTree(
            max_level=self.params.get('max_tree_level', 4),
            c=self.params.get('polya_c', 1.0)
        )
        self.window_size = self.params.get('window_size', 500)
        self.threshold = self.params.get('threshold', 0.15)
        self.min_samples = self.params.get('min_samples', 100)

        self.ref_window = deque(maxlen=self.window_size)
        self.cur_window = deque(maxlen=self.window_size)
        self.last_B = 1.0

    def update(self, val: float) -> bool:
        # 初始化阶段
        if not self.is_initialized:
            self.cur_window.append(val)
            if len(self.cur_window) >= self.min_samples:
                # 初始化完成：Cur -> Ref
                self.pt.reset_counts()
                self.ref_window.clear()
                for v in self.cur_window:
                    self.ref_window.append(v)
                    self.pt.update(v, 'ref', 1)
                    self.pt.update(v, 'cur', 1)
                self.is_initialized = True
            return False

        # 运行阶段
        if len(self.cur_window) == self.window_size:
            old_val = self.cur_window.popleft()
            self.pt.update(old_val, 'cur', -1)

        self.cur_window.append(val)
        self.pt.update(val, 'cur', 1)

        # 计算 Bayes Factor
        log_B = self.pt.compute_total_bayes_factor()
        try:
            self.last_B = math.exp(log_B)
        except OverflowError:
            self.last_B = float('inf') if log_B > 0 else 0.0

        is_drift = log_B < math.log(self.threshold)

        if is_drift:
            self.drift_count += 1

        return is_drift

    def reset(self):
        # BNDM 的 reset 通常是把 Cur 设为 Ref
        self.pt.reset_counts()
        self.ref_window.clear()
        data_list = list(self.cur_window)
        for val in data_list:
            self.ref_window.append(val)
            self.pt.update(val, 'ref', 1)
            self.pt.update(val, 'cur', 1)
        # 注意：这里 cur_window 不清空，而是继续滑动，或者根据策略清空

    def get_info(self) -> Dict:
        return {
            "bayes_factor": self.last_B,
            "ref_size": len(self.ref_window),
            "cur_size": len(self.cur_window)
        }

    def get_drift_evidence(self) -> str:
        return f"Bayes Factor {self.last_B:.6e} < Threshold {self.threshold}"


# --- ADWIN Detector ---
class ADWINDetector(BaseDriftDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        if not RIVER_AVAILABLE:
            raise ImportError("River library not installed. Cannot use ADWIN.")

        self.params = config.get('detectors', {}).get('adwin', {})
        self.adwin = drift.ADWIN(
            delta=self.params.get('delta', 0.002),
            clock=self.params.get('clock', 32)
        )
        self.width = 0
        self.variance = 0.0
        self.is_initialized = True  # ADWIN 不需要显式初始化阶段

    def update(self, val: float) -> bool:
        self.adwin.update(val)
        self.width = self.adwin.width
        self.variance = self.adwin.variance

        if self.adwin.drift_detected:
            self.drift_count += 1
            return True
        return False

    def reset(self):
        # ADWIN 自动处理窗口重置，手动 reset 意味着完全重来
        self.adwin = drift.ADWIN(
            delta=self.params.get('delta', 0.002),
            clock=self.params.get('clock', 32)
        )

    def get_info(self) -> Dict:
        return {
            "width": self.width,
            "variance": self.variance
        }

    def get_drift_evidence(self) -> str:
        return f"ADWIN Width Shrink detected. Variance: {self.variance:.4f}"


# ==========================================
# 3. 适应策略管理器 (Adaptation Manager)
# ==========================================

class DriftAdaptationManager:
    """管理当漂移发生时的适应策略"""

    def __init__(self, config: Dict, pl_module):
        self.config = config.get('adaptation', {})
        self.pl_module = pl_module
        self.logger = logging.getLogger("DriftAdaptation")

    def adapt(self, detector: BaseDriftDetector):
        """执行配置的所有适应策略"""
        actions_taken = []

        # 1. 学习率衰减
        if self.config.get('lr_decay', {}).get('enabled', False):
            factor = self.config['lr_decay'].get('factor', 0.5)
            min_lr = float(self.config['lr_decay'].get('min_lr', 1e-7))

            optimizers = self.pl_module.trainer.optimizers
            for opt in optimizers:
                for param_group in opt.param_groups:
                    old_lr = param_group['lr']
                    new_lr = max(old_lr * factor, min_lr)
                    param_group['lr'] = new_lr

            actions_taken.append(f"LR Decay ({factor}x)")

        # 2. 优化器重置 (清除动量)
        if self.config.get('optimizer_reset', {}).get('enabled', False):
            optimizers = self.pl_module.trainer.optimizers
            for opt in optimizers:
                if hasattr(opt, 'state'):
                    opt.state.clear()  # 清空 state (如 Adam 的 m, v)
            actions_taken.append("Optimizer Momentum Reset")

        # 3. 检测器重置 (Window Reset)
        if self.config.get('window_reset', {}).get('enabled', False):
            detector.reset()
            actions_taken.append("Detector Window Reset")

        self.logger.info(f"🛡️ 漂移适应执行: {', '.join(actions_taken)}")


# ==========================================
# 4. 工厂类
# ==========================================

class DriftDetectorFactory:
    @staticmethod
    def create(config: Dict, algorithm: str) -> BaseDriftDetector:
        if algorithm == "bndm":
            return BNDMDetector(config)
        elif algorithm == "adwin":
            return ADWINDetector(config)
        # elif algorithm == "ks": return KSDetector(config)
        else:
            raise ValueError(f"Unknown drift algorithm: {algorithm}")


# ==========================================
# 5. 检测器管理器 (处理主/影模式)
# ==========================================

class ConceptDriftManager:
    """整合所有逻辑的入口"""

    def __init__(self, cfg, pl_module):
        self.drift_cfg = cfg.concept_drift
        self.adapter = DriftAdaptationManager(self.drift_cfg, pl_module)

        # 主检测器
        self.main_detector = DriftDetectorFactory.create(self.drift_cfg, self.drift_cfg.algorithm)
        self.main_name = self.drift_cfg.algorithm

        # 影子检测器 (用于对比)
        self.shadow_detectors = {}
        if self.drift_cfg.get('shadow_mode', {}).get('enabled', False):
            for name in self.drift_cfg.shadow_mode.algorithms:
                if name != self.main_name:
                    self.shadow_detectors[name] = DriftDetectorFactory.create(self.drift_cfg, name)

        self.history = []

    def process_batch(self, features: torch.Tensor, global_step: int, current_epoch: int) -> Dict:
        """处理一个Batch的数据"""
        # 1. 预处理 (使用主检测器的预处理逻辑，保证一致性)
        z_vals = self.main_detector.preprocess(features)

        # 2. 更新主检测器
        main_drift = False
        # 批量更新（虽然 BNDM 支持流式，但在 Batch Loop 中循环调用）
        # 为了性能，可以抽样。这里全量更新。
        for z in z_vals:
            if self.main_detector.update(z):
                main_drift = True
                # 注意：有些检测器（如ADWIN）是每个点都可能触发，
                # 这里我们标记该 Batch 发生了漂移

        # 3. 更新影子检测器
        shadow_drifts = {}
        for name, det in self.shadow_detectors.items():
            triggered = False
            for z in z_vals:
                if det.update(z):
                    triggered = True
            shadow_drifts[name] = triggered

        # 4. 触发适应 (仅由主检测器决定)
        if main_drift:
            self.adapter.adapt(self.main_detector)
            # 也可以选择是否重置影子检测器，通常不重置以观察它们是否随后触发

        # 5. 记录历史
        if main_drift or any(shadow_drifts.values()):
            record = {
                "step": global_step,
                "epoch": current_epoch,
                "main_algo": self.main_name,
                "main_triggered": main_drift,
                "main_info": self.main_detector.get_info(),
                "shadow_status": shadow_drifts
            }
            self.history.append(record)

        return {
            "main_drift": main_drift,
            "info": self.main_detector.get_info()
        }

    def generate_report(self):
        """生成对比分析报告"""
        report = []
        report.append("=" * 60)
        report.append("📊 概念漂移检测器对比报告")
        report.append(f"主算法 (Active): {self.main_name.upper()} | 触发次数: {self.main_detector.drift_count}")

        for name, det in self.shadow_detectors.items():
            report.append(f"影算法 (Passive): {name.upper()} | 触发次数: {det.drift_count}")

        report.append("-" * 60)
        report.append(f"{'Step':<10} | {'Epoch':<6} | {self.main_name.upper():<10} | {'Shadow Algos'}")

        for rec in self.history:
            main_mark = "🔴 DRIFT" if rec['main_triggered'] else "   -"
            shadow_marks = [f"{k}:{'🔵' if v else '-'}" for k, v in rec['shadow_status'].items()]
            report.append(f"{rec['step']:<10} | {rec['epoch']:<6} | {main_mark:<10} | {', '.join(shadow_marks)}")

        report.append("=" * 60)
        return "\n".join(report)