from typing import List, Dict, Optional, Tuple, Any
import ast
import logging
import os, sys

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
# 导入配置管理模块
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.DEBUG)    

verbose = False


def normalize_label(label: str) -> str:
    if label is None:
        return ""
    return str(label).strip().lower()        
class SessionParser:
    def __init__(self, flow_node_builder, session_label_id_map=None):
        self.flow_node_builder = flow_node_builder
        self.session_label_id_map = session_label_id_map or {}
    
    def extract_flow_uid_list(self, session_row) -> List[str]:
        """从会话行中提取流UID列表"""
        if 'flow_uid_list' not in session_row:
            return []
        
        try:
            return ast.literal_eval(session_row['flow_uid_list'])
        except (ValueError, SyntaxError):
            return []
        
    # 辅助函数：基于配置的标签映射进行前缀匹配
    def match_configured_label(self, raw_label: str) -> Optional[str]:
        """
        Map a flow-level attack label to a configured session-level attack category.
        Only called when the flow is already known to be malicious.
        """
        raw_label = raw_label.strip().lower()

        for configured_label in self.session_label_id_map.keys():
            configured_lower = configured_label.lower()

            # 精确匹配
            if raw_label == configured_lower:
                return configured_label

            # 前缀 / 子串匹配（保留你的需求）
            if configured_lower in raw_label:
                return configured_label

        return None

    def aggregate_session_label(self, flow_uid_list):
        label_name = self.aggregate_session_label_without_label_id(flow_uid_list)

        # ⭐【关键】统一做 label 规范化（大小写 / 空格）
        label_name = normalize_label(label_name)

        is_malicious = (label_name != "benign")

        # mixed / unknown 直接跳过
        if label_name == "mixed":
            return label_name, -1, is_malicious

        label_id = self.session_label_id_map.get(label_name)
        if label_id is None:
            logger.error(
                f"Session label '{label_name}' not found in session_label_id_map. "
                f"Available labels: {list(self.session_label_id_map.keys())}"
            )
            return label_name, -1, is_malicious

        return label_name, label_id, is_malicious

    def aggregate_session_label_without_label_id(self, flow_uid_list):
        """
        Session label aggregation rule:
        1. All BENIGN                -> BENIGN
        2. BENIGN + single attack    -> that attack
        3. BENIGN + multiple attacks -> mixed
        4. Single attack only        -> that attack
        5. Multiple attacks only     -> mixed
        """

        attack_labels = set()

        for flow_uid in flow_uid_list:
            flow_record = self.flow_node_builder.get_flow_record(flow_uid)
            if flow_record is None:
                continue

            raw_label = str(flow_record.get("label", "")).strip().lower()

            # 阶段 1：严格 benign 判断
            if raw_label in ("benign", "background", "unknown", ""):
                continue

            # 阶段 2：攻击类型归一（允许前缀）
            attack_type = self.match_configured_label(raw_label)
            if attack_type is None:
                return "mixed"   # 无法归类的攻击，保守处理

            attack_labels.add(attack_type)


        # ---------- 聚合规则 ----------
        if len(attack_labels) == 0:
            return "benign"

        if len(attack_labels) == 1:
            return next(iter(attack_labels))

        return "mixed"
