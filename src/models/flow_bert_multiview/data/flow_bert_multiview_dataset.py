import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
import os,sys
import ast
from typing import List, Dict, Any, Tuple
import re
from tqdm import tqdm  # 添加tqdm导入
from collections import Counter

# 导入配置管理器和相关模块
try:
    # 添加../../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'utils')
    sys.path.insert(0, utils_path)    
    import config_manager as ConfigManager
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.INFO)

except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

class MultiviewFlowDataset(Dataset):
    """多视图流量数据集"""
    
    def __init__(self, df: pd.DataFrame, cfg: DictConfig, is_training: bool = True, 
                 train_categorical_mappings: Dict = None,
                 train_categorical_columns_effective: List[str] = None):
        self.flow_df = df.reset_index(drop=True)
        self.cfg = cfg
        self.labels_cfg = cfg.datasets.labels
        self.is_training = is_training
        self.max_seq_length = cfg.data.max_seq_length 
        self.mtu_normalize = getattr(self.cfg.data.sequence_features, "mtu_normalize", 1500)
        
        # 尝试从配置中读取session_label_id_map的长度
        self.session_label_id_map = dict(ConfigManager.read_session_label_id_map(self.cfg.data.dataset))                    

        # 检查各个可选特征是否启用
        self.categorical_features_enabled = hasattr(cfg.data.tabular_features, "categorical_features") \
            and cfg.data.tabular_features.categorical_features is not None

        if self.categorical_features_enabled:
            self.categorical_columns = cfg.data.tabular_features.categorical_features
        else:
            self.categorical_columns = []

        # 🔴🔴🔴 关键修改：非训练集使用训练集的映射
        if self.categorical_features_enabled and not is_training and train_categorical_mappings is not None:
            self.categorical_val2idx_mappings = train_categorical_mappings
            if train_categorical_columns_effective is None:
                raise ValueError("非训练集必须提供 train_categorical_columns_effective 参数")
            
            if len(train_categorical_columns_effective) == 0:
                logger.warning("⚠️ train_categorical_columns_effective 为空列表")

            self.train_categorical_columns_effective = train_categorical_columns_effective
            self.use_train_mappings = True
            logger.info(f"✅ 使用训练集的类别映射（映射大小: {len(train_categorical_mappings)}）")
        else:
            self.categorical_val2idx_mappings = {}
            self.train_categorical_columns_effective = []
            self.use_train_mappings = False

        self.text_features_enabled = False
        if hasattr(cfg.data, 'text_features') and cfg.data.text_features is not None:
            if hasattr(cfg.data.text_features, 'enabled') and cfg.data.text_features.enabled:
                self.text_features_enabled = True
                
        self.domain_embedding_enabled = False
        if hasattr(cfg.data, 'domain_name_embedding_features') and hasattr(cfg.data.domain_name_embedding_features, 'enabled') and hasattr(cfg.data.domain_name_embedding_features, 'column_list'):
            if cfg.data.domain_name_embedding_features.enabled and len(cfg.data.domain_name_embedding_features.column_list) > 0:
                self.domain_embedding_enabled = True
        
        self.sequence_features_enabled = False
        if hasattr(cfg.data, 'sequence_features') and cfg.data.sequence_features is not None:
            if hasattr(cfg.data.sequence_features, 'enabled') and cfg.data.sequence_features.enabled:
                self.sequence_features_enabled = True
                
        # 打印序列特征是否启用
        if self.sequence_features_enabled:
            logger.info("启用序列特征视图")
        else:
            logger.info("序列特征未配置，跳过该视图")

        # 打印文本特征是否启用
        if self.text_features_enabled:
            logger.info("启用文本特征视图")
        else:
            logger.info("文本特征未配置，跳过该视图")

        # 打印域名嵌入特征是否启用
        if self.domain_embedding_enabled:
            logger.info(f"启用域名嵌入特征，共 {len(cfg.data.domain_name_embedding_features.column_list)} 个特征列")
            self.prob_list_length = len(self.session_label_id_map)
            # logger.info(f"从配置中读取到域名嵌入的概率列表长度: {self.prob_list_length}, label_id_map = {label_id_map}")
        else:
            logger.info("域名嵌入特征未配置或为空，跳过该视图")

        # 预处理数据
        self._preprocess_data()

        # ===============================
        # 善意/恶意行为的二分类标签配置
        # ===============================
        self.is_malicious_column = cfg.data.is_malicious_column
        if self.is_malicious_column in self.flow_df.columns:
            self.is_malicious_labels = self.flow_df[self.is_malicious_column].astype(int).values
        else:
            raise KeyError(f"找不到 is_malicious 标签列 {self.is_malicious_column}，请检查 CSV 文件 和配置文件 is_malicious_column")

        # ===============================
        # 多分类标签配置
        # ===============================
        self.multiclass_label_column = self.cfg.data.multiclass_label_column
        if not self.multiclass_label_column in self.flow_df.columns:
            raise KeyError(f"找不到多分类标签列 {self.multiclass_label_column}，请检查 CSV 和配置文件 multiclass_label_column")

        # attack_family 类别（去除首尾空白，保持原始大小写）
        self.attack_family_classes = (
            [c.strip() for c in self.labels_cfg.attack_family.classes]
            if "attack_family" in self.labels_cfg
            else None
        )

        # attack_type 类别（去除首尾空白，保持原始大小写）
        self.attack_type_classes = (
            [c.strip() for c in self.labels_cfg.attack_type.classes]
            if "attack_type" in self.labels_cfg
            else None
        )

        # attack_type -> attack_family 映射（同时 strip key 和 value）
        self.attack_type_parent_mapping = (
            {k.strip(): v.strip() for k, v in self.labels_cfg.attack_type.parent_mapping.items()}
            if "attack_type" in self.labels_cfg
            and hasattr(self.labels_cfg.attack_type, "parent_mapping")
            else None
        )

        # 检查attack_type -> attack_family 映射的配置错误
        if self.attack_type_parent_mapping is not None:
            for t, f in self.attack_type_parent_mapping.items():
                assert self.attack_type_classes is None or t in self.attack_type_classes, \
                    f"attack_type '{t}' not in labels.attack_type.classes"
                assert self.attack_family_classes is None or f in self.attack_family_classes, \
                    f"attack_family '{f}' not in labels.attack_family.classes"

        if self.attack_family_classes is not None:
            logger.info(
                f"[Dataset] attack_family_classes ({len(self.attack_family_classes)}): "
                f"{self.attack_family_classes}"
            )

        if self.attack_type_classes is not None:
            logger.info(
                f"[Dataset] attack_type_classes ({len(self.attack_type_classes)}): "
                f"{self.attack_type_classes}"
            )

    def _preprocess_data(self):
        """预处理数据"""
        if hasattr(self, '_preprocessed') and self._preprocessed:
            logger.info("数据已经预处理过，跳过...")
            return
        
        self._preprocessed = True  # 标记为已处理
        logger.info("预处理多视图数据...")
    
        # 计算实际启用的视图数量
        self.num_optional_views = 1  # 表格数据特征：数值特征（必选） + 域名嵌入特征（可选）
        if self.text_features_enabled:
            self.num_optional_views += 1 # 文本视图可选
        if self.sequence_features_enabled:
            self.num_optional_views += 1 # 数据包序列视图可选

        # 使用tqdm添加进度条；注意：除了optional_views，还有必选的类别型特征+数值型特征
        with tqdm(total=self.num_optional_views+2, desc="数据预处理进度", position=0, leave=True) as pbar:
            # 处理数据包序列特征（可选）
            if self.sequence_features_enabled:
                self._process_sequence_features()
                pbar.update(1)
                pbar.set_description("数据预处理阶段：IP数据包序列特征处理完成")
            else:
                # 创建空的序列特征占位符
                if not hasattr(self, 'sequences') or not self.sequences:
                    self.sequences = [{
                        'directions': [0.0] * self.max_seq_length,
                        'payload_sizes': [0.0] * self.max_seq_length,
                        'iat_times': [0.0] * self.max_seq_length,
                        'packet_numbers': [0.0] * self.max_seq_length,
                        'avg_payload_sizes': [0.0] * self.max_seq_length,
                        'durations': [0.0] * self.max_seq_length,                        
                        'sequence_mask': [0] * self.max_seq_length,
                        'original_length': 0
                    } for _ in range(len(self.flow_df))]
                pbar.set_description("数据预处理阶段：跳过序列特征处理")
            
            # 处理文本特征
            if self.text_features_enabled:
                self._process_text_features()
                pbar.update(1)
                pbar.set_description("数据预处理阶段：文本特征处理完成")
            else:
                # 创建空的文本特征占位符
                if not hasattr(self, 'text_features') or not self.text_features:
                    self.text_features = [{} for _ in range(len(self.flow_df))]
                pbar.set_description("数据预处理阶段：跳过文本特征处理")               
            
            # 处理域名嵌入特征（概率列表）
            if self.domain_embedding_enabled:
                self._process_domain_embedding_features()
                pbar.update(1)
                pbar.set_description("数据预处理阶段：域名嵌入特征处理完成")
            else:
                # 创建空的域名嵌入特征占位符
                if not hasattr(self, 'domain_embedding_features') or not self.domain_embedding_features:
                    self.domain_embedding_features = [[] for _ in range(len(self.flow_df))]
                pbar.set_description("数据预处理阶段：跳过域名嵌入特征处理")                              

            # 处理类别型特征（必选）            
            if self.categorical_features_enabled:
                self._process_categorical_features()
                pbar.update(1)
                pbar.set_description("数据预处理阶段：类别特征处理完成")

            # 处理数值型特征（必选）
            self._process_numeric_features()
            pbar.update(1)
            pbar.set_description("数据预处理阶段：逐流数值特征处理完成")

        logger.info("数据预处理完成")
    
    def _process_sequence_features(self):
        """处理数据包序列特征"""
        self.sequences = []
        
        # 添加详细的进度条
        with tqdm(total=len(self.flow_df), desc="处理数据包序列特征", position=0, leave=True) as pbar:
            for idx, row in self.flow_df.iterrows():
                sequence_data = self._parse_sequence_row(row)
                self.sequences.append(sequence_data)
                pbar.update(1)
    
        # 统计空序列数量（用于日志记录）
        empty_count = sum(1 for seq in self.sequences if seq['original_length'] == 0)
        if empty_count > 0:
            logger.info(f"序列特征处理完成，发现 {empty_count} 个空序列（已用零填充）")
        else:
            logger.info("序列特征处理完成，所有序列数据有效")

    
    def _parse_sequence_row(self, row: pd.Series) -> Dict[str, Any]:
        """解析单行的序列数据，返回融合了方向信息的特征"""
        try:
            # -------------------------------
            # 获取 flow uid，打印调试信息用
            # -------------------------------
            # 尽量兼容常见 UID 字段名
            flow_uid = (
                row.get("uid") 
                or row.get("flow_uid") 
                or row.get("flowID") 
                or row.get("Flow ID") 
                or "UNKNOWN_UID"
            )

            # 解析方向向量
            packet_directions_str = row.get(self.cfg.data.sequence_features.packet_direction, '[]')
            packet_directions = self._safe_parse_list(packet_directions_str)
            
            # 解析载荷大小向量（0-1500字节范围）
            packet_payload_sizes_str = row.get(self.cfg.data.sequence_features.packet_payload, '[]')
            packet_payload_sizes = self._safe_parse_list(packet_payload_sizes_str)
            
            # 解析时间间隔向量（毫秒单位，可能达到百万级）
            packet_iat_times_str = row.get(self.cfg.data.sequence_features.packet_iat, '[]')
            packet_iat_times = self._safe_parse_list(packet_iat_times_str)

            # 解析数据包时间戳序列
            packet_timestamps_str = row.get(self.cfg.data.sequence_features.packet_timestamp, '[]')
            packet_timestamps = self._safe_parse_list(packet_timestamps_str)

            # 解析各bulk中第一个数据包的位置的序列
            # bulk_first_packet_index_vector_str = row.get(self.cfg.data.sequence_features.bulk_first_packet_index_vector, '[]')
            # bulk_first_packet_indices = self._safe_parse_list(bulk_first_packet_index_vector_str)

            # 解析各bulk的长度序列
            bulk_length_vector_str = row.get(self.cfg.data.sequence_features.bulk_length_vector, '[]')
            bulk_lengths = self._safe_parse_list(bulk_length_vector_str)

            # 解析批量数据包传输的索引序列
            bulk_packet_index_str = row.get(self.cfg.data.sequence_features.bulk_packet_index, '[]')
            bulk_packet_indices = self._safe_parse_list(bulk_packet_index_str)

            # 如果有 bulk 信息，就会返回 None；否则，会构造一个消息序列，里面包含单包和多包消息的信息
            msg_seq = self._extract_normalized_directed_msg_seq(flow_uid, 
                                                                packet_directions, 
                                                                packet_payload_sizes, 
                                                                packet_timestamps, 
                                                                bulk_lengths,
                                                                bulk_packet_indices)
            
            if msg_seq is not None:
                self._assert_directed_seq_output(msg_seq, self.max_seq_length, f"flow_uid={flow_uid} msg_seq")
                return msg_seq
            
            # 如果网络流只有零载荷包，有可能会出现以下情况：
            # * 该网络流只有 SYN → ACK → FIN 的数据包时序序列，比如SYN洪水攻击；
            # * 或者在发现扫描类攻击（大量 0 payload 探测）；
            # * 或者，心跳 / keep-alive 模式。
            # 由于零载荷包（如 ACK / 控制包）不作为 message，
            # 所以在没有 bulk 信息时，需要退回到单包序列处理。
            pkt_seq = self._extract_normalized_directed_pkt_seq(flow_uid, 
                                                                packet_directions, 
                                                                packet_payload_sizes, 
                                                                packet_iat_times)
            self._assert_directed_seq_output(pkt_seq, self.max_seq_length, f"flow_uid={flow_uid} pkt_seq")
            return pkt_seq
                
        except Exception as e:
            logger.warning(f"解析序列数据失败（flow_uid={flow_uid}）: {e}")
            return {
                'directions': [0.0] * self.max_seq_length,
                'payload_sizes': [0.0] * self.max_seq_length,
                'iat_times': [0.0] * self.max_seq_length,
                'packet_numbers': [0.0] * self.max_seq_length,
                'avg_payload_sizes': [0.0] * self.max_seq_length,
                'durations': [0.0] * self.max_seq_length,
                'sequence_mask': [0] * self.max_seq_length,
                'original_length': 0
            }
            
    def _parse_direction_value(self, direction_val):
        """统一解析方向值"""
        if direction_val is None:
            return 1
            
        if isinstance(direction_val, str):
            direction_str = str(direction_val).lower().strip()
            if direction_str in ['true', '1', 'forward', 'fwd']:
                return 1
            elif direction_str in ['false', '0', 'backward', 'bwd']:
                return 0
            else:
                # 尝试数值转换
                try:
                    num_val = float(direction_val)
                    return 1 if num_val > 0 else 0
                except:
                    return 1  # 默认值
        
        elif isinstance(direction_val, bool):
            return 1 if direction_val else 0
        
        elif isinstance(direction_val, (int, float)):
            return 1 if direction_val > 0 else 0
        
        else:
            return 1  # 默认值

    @staticmethod
    def _safe_log_scale_normalize(value: float, eps=1e-6, scale=1.0) -> float:
        """
        对以「秒」为单位的时间间隔进行稳健的 log 缩放，用于抑制超长时间间隔对模型训练的不稳定影响。
        """
        if abs(value) < eps:
            return 0.0
        
        sign = 1.0 if value > 0 else -1.0
        x = abs(value) + eps

        return sign * np.log1p(x / scale)

    def _assert_directed_seq_output(self, out, max_len, name):
        assert isinstance(out, dict), f"{name}: output is not dict"

        expected_keys = {
            'directions',
            'payload_sizes',
            'iat_times',
            'packet_numbers',
            'avg_payload_sizes',
            'durations',
            'sequence_mask',
            'original_length',
        }
        assert set(out.keys()) == expected_keys, \
            f"{name}: keys mismatch, got {set(out.keys())}"

        # ---------- 类型检查 ----------
        assert isinstance(out['directions'], list), \
            f"{name}: directions is not list"
        assert isinstance(out['payload_sizes'], list), \
            f"{name}: payload_sizes is not list"
        assert isinstance(out['iat_times'], list), \
            f"{name}: iat_times is not list"
        assert isinstance(out['packet_numbers'], list), \
            f"{name}: packet_numbers is not list"
        assert isinstance(out['avg_payload_sizes'], list), \
            f"{name}: avg_payload_sizes is not list"
        assert isinstance(out['durations'], list), \
            f"{name}: durations is not list"
        assert isinstance(out['sequence_mask'], list), \
            f"{name}: sequence_mask is not list"
        assert isinstance(out['original_length'], int), \
            f"{name}: original_length is not int"

        # ---------- 长度检查 ----------
        assert len(out['directions']) == max_len, \
            f"{name}: directions length {len(out['directions'])} != {max_len}"
        assert len(out['payload_sizes']) == max_len, \
            f"{name}: payload_sizes length {len(out['payload_sizes'])} != {max_len}"
        assert len(out['iat_times']) == max_len, \
            f"{name}: iat_times length {len(out['iat_times'])} != {max_len}"
        assert len(out['packet_numbers']) == max_len, \
            f"{name}: packet_numbers length {len(out['packet_numbers'])} != {max_len}"
        assert len(out['avg_payload_sizes']) == max_len, \
            f"{name}: avg_payload_sizes length {len(out['avg_payload_sizes'])} != {max_len}"
        assert len(out['durations']) == max_len, \
            f"{name}: durations length {len(out['durations'])} != {max_len}"
        assert len(out['sequence_mask']) == max_len, \
            f"{name}: sequence_mask length {len(out['sequence_mask'])} != {max_len}"

        # ---------- 元素类型与取值检查 ----------
        for v in out['directions']:
            assert isinstance(v, (int, float)), \
                f"{name}: directions contains non-numeric value {v}"

        for v in out['payload_sizes']:
            assert isinstance(v, (int, float)), \
                f"{name}: payload_sizes contains non-numeric value {v}"

        for v in out['iat_times']:
            assert isinstance(v, (int, float)), \
                f"{name}: iat_times contains non-numeric value {v}"
        
        for v in out['packet_numbers']:
            assert isinstance(v, (int, float)), \
                f"{name}: packet_numbers contains non-numeric value {v}"

        for v in out['avg_payload_sizes']:
            assert isinstance(v, (int, float)), \
                f"{name}: avg_payload_sizes contains non-numeric value {v}"

        for v in out['durations']:
            assert isinstance(v, (int, float)), \
                f"{name}: durations contains non-numeric value {v}"
            
        for v in out['sequence_mask']:
            assert isinstance(v, int), \
                f"{name}: sequence_mask contains non-int value {v}"
            assert v in (0, 1), \
                f"{name}: sequence_mask contains invalid value {v}"

        # ---------- 语义一致性 ----------
        valid_len = sum(out['sequence_mask'])
        assert out['original_length'] == valid_len, \
            f"{name}: original_length {out['original_length']} != mask sum {valid_len}"
        

    def _extract_normalized_directed_pkt_seq(self, flow_uid, packet_directions, packet_payload_sizes, packet_iat_times):
        # 对数据包方向做归一化成为0或者1
        normalized_pkt_directions = [self._parse_direction_value(x) for x in packet_directions]  # 0 或 1

        # 对数据包载荷大小做归一化
        mtu_float = float(self.mtu_normalize) # 默认1500字节，没有继续采用 pkt_payload_size / mtu 的归一化方式
        normalized_pkt_payload_sizes = [x / mtu_float for x in packet_payload_sizes]  # 归一化到0-1
        # normalized_pkt_payload_sizes = [self._safe_log_scale_normalize(x) for x in packet_payload_sizes]  # 归一化到0-1

        # Zeek Flowmeter插件里面packet_iat_vector的时间单位是microseconds，需要log归一化，否则模型训练会不稳定。
        # 具体参考插件代码700行左右，对iat_vector的操作。
        # https://gitee.com/seu-csqjxiao/zeek-flowmeter/blob/seu-devel/scripts/flowmeter.zeek
        #         # add the flow IAT, after converting it to microseconds, to the flow IAT vector
        #          # iat_vector[c$uid]["flow"] += |iat$flow| * 1000000.0;
        if packet_iat_times:
            # 处理时间间隔，添加第一个包的时间间隔为0
            packet_iat_times = [0.0] + packet_iat_times  # 第一个包的时间间隔为0
            
            # 使用统一的_safe_log_scale_time函数处理时间间隔
            processed_pkt_iat_times = []
            for x in packet_iat_times:
                x = x / 1000000.0  # 转换为秒
                # 为避免超长的数据包时间间隔导致训练不稳定，进行log缩放
                processed_x = self._safe_log_scale_normalize(x)
                processed_pkt_iat_times.append(processed_x)
            normalized_pkt_iat_times = processed_pkt_iat_times
        else:
            normalized_pkt_iat_times = [0.0]

        # 计算每个消息的数据包数量列表
        pkt_packet_numbers = [1] * len(normalized_pkt_directions)

        # 计算平均载荷大小序列
        normalized_pkt_avg_payload_sizes = normalized_pkt_payload_sizes.copy()

        # 计算每个单包消息的持续时间序列，这里简单设为0
        durations = [0] * len(normalized_pkt_directions)

        # -----------------------------------------------------------------------------------------
        # 空序列检测 - 返回空序列而不是None。注意必须在packet_iat_times做了补前导零之后，才能检测
        # -----------------------------------------------------------------------------------------
        if len(normalized_pkt_directions) == 0 or len(normalized_pkt_payload_sizes) == 0 or len(normalized_pkt_iat_times) == 0:
            logger.error(f"_extract_normalized_directed_pkt_seq()：检测到空序列数据，将跳过此样本 flow_uid={flow_uid}, "
                        f"directions={len(normalized_pkt_directions)}, payload={len(normalized_pkt_payload_sizes)}, iat={len(normalized_pkt_iat_times)}")
                        # 返回全零的空序列，而不是None
            return {
                'directions': [0.0] * self.max_seq_length,
                'iat_times': [0.0] * self.max_seq_length,
                'payload_sizes': [0.0] * self.max_seq_length,
                'packet_numbers': [0.0] * self.max_seq_length,
                'avg_payload_sizes': [0.0] * self.max_seq_length,
                'durations': [0.0] * self.max_seq_length,
                'sequence_mask': [0] * self.max_seq_length,
                'original_length': 0
            }

        # 统一截断到 max_seq_length
        pkt_seq_len = len(normalized_pkt_directions)
        assert pkt_seq_len == len(normalized_pkt_payload_sizes) == len(normalized_pkt_iat_times) == len(pkt_packet_numbers) == len(normalized_pkt_avg_payload_sizes) == len(durations), \
            f"序列长度不一致: flow uid = {flow_uid}, directions={len(normalized_pkt_directions)}, payload={len(normalized_pkt_payload_sizes)}, iat={len(normalized_pkt_iat_times)}, packet_numbers={len(pkt_packet_numbers)}, avg_payload_sizes={len(normalized_pkt_avg_payload_sizes)}, durations={len(durations)}"

        if pkt_seq_len > self.max_seq_length:
            pkt_sequence_mask = [1] * self.max_seq_length

            # 超长截断
            normalized_pkt_directions = normalized_pkt_directions[:self.max_seq_length]
            normalized_pkt_payload_sizes = normalized_pkt_payload_sizes[:self.max_seq_length]
            normalized_pkt_iat_times = normalized_pkt_iat_times[:self.max_seq_length]
            pkt_packet_numbers = pkt_packet_numbers[:self.max_seq_length]
            normalized_pkt_avg_payload_sizes = normalized_pkt_avg_payload_sizes[:self.max_seq_length]
            durations = durations[:self.max_seq_length]

        else:
            pkt_sequence_mask = [1] * pkt_seq_len

            # 短序列补齐
            if pkt_seq_len < self.max_seq_length:
                pad_len = self.max_seq_length - pkt_seq_len
                normalized_pkt_directions.extend([0.0] * pad_len)
                normalized_pkt_payload_sizes.extend([0.0] * pad_len)
                normalized_pkt_iat_times.extend([0.0] * pad_len)
                pkt_packet_numbers.extend([0.0] * pad_len)
                normalized_pkt_avg_payload_sizes.extend([0.0] * pad_len)                
                durations.extend([0.0] * pad_len)
                pkt_sequence_mask.extend([0] * pad_len)

        return {
                'directions': normalized_pkt_directions,
                'iat_times': normalized_pkt_iat_times,
                'payload_sizes': normalized_pkt_payload_sizes,
                'packet_numbers': pkt_packet_numbers,
                'avg_payload_sizes': normalized_pkt_avg_payload_sizes,
                'durations': durations,
                'sequence_mask': pkt_sequence_mask,
                'original_length': pkt_seq_len,
                }

    @staticmethod
    def build_pkt_to_bulk_idx_map(
        bulk_lengths,
        bulk_packet_indices,
    ):
        """
        构建 packet_index -> bulk_idx 的映射字典。

        说明：
        - bulk_length_vector 记录的是 data_size > 0 的数据包数量，
        packet index 在 flow 中可能是不连续的；
        - bulk_packet_indices 已按 bulk 顺序拼接，仅包含 data_size > 0 的包；
        - 因此，通过 bulk_length_vector 对 bulk_packet_indices 顺序切分，
        可以准确恢复每个 bulk 内的 packet index 集合。
        """
        pkt_to_bulk_idx = {}

        offset = 0
        for bulk_idx, bulk_len in enumerate(bulk_lengths):
            # 取属于该 bulk 的 data packets（不要求 index 连续）
            bulk_pkts = bulk_packet_indices[offset : offset + bulk_len]

            for pkt_idx in bulk_pkts:
                pkt_to_bulk_idx[pkt_idx] = bulk_idx

            offset += bulk_len

        return pkt_to_bulk_idx
    
    def _extract_normalized_directed_msg_seq(self, flow_uid, 
                                             packet_directions, packet_payload_sizes, packet_timestamps, 
                                             bulk_lengths, bulk_packet_indices):
        '''
        把packet序列变换成single packet 或者 bulk 的信息序列
        '''

        # -------------------------------
        # 空序列检测 - 返回空序列而不是None
        # -------------------------------
        if len(packet_directions) == 0 or len(packet_payload_sizes) == 0 or len(packet_timestamps) == 0:
            logger.error(f"_extract_normalized_directed_msg_seq()：检测到空序列数据，将跳过此样本 flow_uid={flow_uid}, directions={len(packet_directions)}, "
                        f"payload={len(packet_payload_sizes)}, timestamps={len(packet_timestamps)}")
                        # 返回全零的空序列，而不是None
            return {
                'directions': [0.0] * self.max_seq_length,
                'payload_sizes': [0.0] * self.max_seq_length,                
                'iat_times': [0.0] * self.max_seq_length,
                'packet_numbers': [0.0] * self.max_seq_length,
                'avg_payload_sizes': [0.0] * self.max_seq_length,
                'durations': [0.0] * self.max_seq_length,
                'sequence_mask': [0] * self.max_seq_length,
                'original_length': 0
            }

        # bulk_lengths 与 bulk_packet_indices 长度一致
        if sum(bulk_lengths) != len(bulk_packet_indices):
            logger.warning(f"bulk_lengths 与 bulk_packet_indices 长度不一致，flow_uid={flow_uid}. "
                           f"将忽略 bulk 信息，按单包处理。")
            return None

        pkt_to_bulk_idx = self.build_pkt_to_bulk_idx_map(bulk_lengths, bulk_packet_indices)

        msg_directions = [] 
        msg_payload_sizes = []
        msg_iat_times = []
        msg_packet_numbers = []
        msg_avg_payload_sizes = []
        msg_durations = []

        prev_msg_timestamp = 0
        current_bulk_direction = None
        current_bulk_bytes = []
        current_bulk_timestamps = []
        current_bulk_idx = -1
        
        for pkt_idx in range(len(packet_payload_sizes)):
            pkt_direction = packet_directions[pkt_idx] if pkt_idx < len(packet_directions) else 0
            pkt_payload_size = packet_payload_sizes[pkt_idx] if pkt_idx < len(packet_payload_sizes) else 0            
            pkt_timestamp = packet_timestamps[pkt_idx] if pkt_idx < len(packet_timestamps) else 0
            
            # if pkt_payload_size == 0:
            #     # 这里可以考虑忽略零payload bytes的载荷包，因为：
            #     # 零载荷包（如 ACK / 控制包）不作为 message，
            #     # 避免高频控制包干扰 message-level 行为建模。
            #     # 也可以选择将其作为单包消息处理。
            #     continue

            if pkt_payload_size == 0:
                # Flowmeter插件中，零载荷包不参与 bulk 传输的划分
                # 因此，直接将其视为不属于任何 bulk，但作为单包消息处理
                bulk_idx_of_current_pkt = None  
            else:
                bulk_idx_of_current_pkt = pkt_to_bulk_idx.get(pkt_idx, None)

            if bulk_idx_of_current_pkt is None or bulk_idx_of_current_pkt != current_bulk_idx:
                # 如果当前包不属于任何 bulk 或者其所属于的bulk_idx不同于当前bulk_idx，那么先结束当前的 bulk（如果有）
                if current_bulk_idx >= 0:
                    # bulk 发生切换（或进入 / 离开 bulk）
                    # 创建一个 multiple-packet message entry
                    prev_msg_timestamp = self._create_multiple_pkt_msg_return_msg_timestamp(
                                            current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                            msg_directions, msg_payload_sizes, msg_iat_times, msg_packet_numbers, msg_avg_payload_sizes, msg_durations)
                    # 通过重置 current bulk info，结束当前的 bulk
                    current_bulk_direction = None
                    current_bulk_bytes = []
                    current_bulk_timestamps = []
                    current_bulk_idx = -1

            if bulk_idx_of_current_pkt is None: 
                # Zeek Flowmeter插件创建一个Bulk时，要求bulk内至少包含5个数据包。查看https://gitee.com/seu-csqjxiao/zeek-flowmeter
                # FlowMeter::bulk_min_length: The minimal number of data packets which have to be in 
                #                             a bulk transmission for it to be considered a bulk transmission. 
                #                             The default value is 5 packets.
                # FlowMeter::bulk_timeout: The maximal allowed inter-arrival time between two data packets 
                #                          so they are considered to be part of the same bulk transmission. 
                #                          The default value is 1s.
                # 因此，如果当前包不属于任何 bulk，则创建单包消息。
                prev_msg_timestamp = self._create_single_pkt_msg_return_msg_timestamp(
                                        pkt_direction, pkt_timestamp, pkt_payload_size, prev_msg_timestamp, 
                                        msg_directions, msg_payload_sizes, msg_iat_times, msg_packet_numbers, msg_avg_payload_sizes, msg_durations)
            else:
                if current_bulk_idx == -1:
                    # 开始一个新的 bulk，其方向由第一个包决定，该bulk后续的包必须方向一致
                    current_bulk_idx = bulk_idx_of_current_pkt
                    current_bulk_direction = pkt_direction

                assert pkt_direction == current_bulk_direction, \
                    f"检测到 bulk 内方向不一致，flow_uid={flow_uid}, bulk_idx={current_bulk_idx}"
                current_bulk_bytes.append(pkt_payload_size)
                current_bulk_timestamps.append(pkt_timestamp)

        
        if current_bulk_direction is not None:
            # 已经扫描完数据包序列。如果当前bulk还没有结束，创建一个 multiple-packet message
            prev_msg_timestamp = self._create_multiple_pkt_msg_return_msg_timestamp(
                                        current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                        msg_directions, msg_payload_sizes, msg_iat_times, msg_packet_numbers, msg_avg_payload_sizes, msg_durations)
            # 通过重置 current bulk info，结束当前的 bulk
            current_bulk_direction = None
            current_bulk_bytes = []
            current_bulk_timestamps = []
            current_bulk_idx = -1

        # -------------------------------
        # 统一截断到 max_seq_length
        # -------------------------------
        msg_seq_len = len(msg_payload_sizes)
        # message 的 original_length 可能为 0，如果：
        # 全是 payload_size == 0 的包，或者 bulk 被忽略
        # 这种情况下，返回 None，表示没有有效的消息序列，直接回退 packet-level sequence 处理
        if msg_seq_len == 0:
            return None 

        msg_sequence_mask = [1] * msg_seq_len
        if msg_seq_len > self.max_seq_length:
            msg_seq_len = self.max_seq_length

            msg_directions = msg_directions[:self.max_seq_length]
            msg_payload_sizes = msg_payload_sizes[:self.max_seq_length]
            msg_iat_times = msg_iat_times[:self.max_seq_length]
            msg_packet_numbers = msg_packet_numbers[:self.max_seq_length]
            msg_avg_payload_sizes = msg_avg_payload_sizes[:self.max_seq_length]
            msg_durations = msg_durations[:self.max_seq_length]
            msg_sequence_mask = [1] * self.max_seq_length

        else:
            pad_len = self.max_seq_length - msg_seq_len
            if pad_len > 0:
                msg_directions.extend([0.0] * pad_len)
                msg_payload_sizes.extend([0.0] * pad_len)      
                msg_iat_times.extend([0.0] * pad_len)
                msg_packet_numbers.extend([0.0] * pad_len)
                msg_avg_payload_sizes.extend([0.0] * pad_len)
                msg_durations.extend([0.0] * pad_len)
                msg_sequence_mask.extend([0] * pad_len)

        return {
                'directions': msg_directions,
                'payload_sizes': msg_payload_sizes,                
                'iat_times': msg_iat_times,
                'packet_numbers': msg_packet_numbers,
                'avg_payload_sizes': msg_avg_payload_sizes,
                'durations': msg_durations,
                'sequence_mask': msg_sequence_mask,
                'original_length': msg_seq_len,
                }

    def _create_single_pkt_msg_return_msg_timestamp(self, pkt_direction, pkt_timestamp, pkt_payload_size, prev_msg_timestamp, 
                                                    msg_directions, msg_payload_sizes, msg_iat_times, msg_packet_numbers, msg_avg_payload_sizes, msg_durations):
        msg_direction_value = self._parse_direction_value(pkt_direction)
        msg_directions.append(msg_direction_value)

        msg_payload_size = pkt_payload_size / float(self.mtu_normalize)  # 归一化到0-1
        # msg_payload_size = self._safe_log_scale_normalize(pkt_payload_size)
        msg_payload_sizes.append(msg_payload_size)

        # Zeek Flowmeter插件里面，pkt_timestamp的时间单位是秒，无需转换。
        # packet_timestamp_vector 以Unix时间戳格式记录flow中各数据包的到达时间。
        # 查看 https://gitee.com/seu-csqjxiao/zeek-flowmeter/tree/seu-devel
        if prev_msg_timestamp == 0:
            # 第一个消息，时间间隔为0
            msg_iat_time = 0.0
        else:
            # message 到达时间定义为该消息中第一个 packet 的时间戳
            msg_iat_time = pkt_timestamp - prev_msg_timestamp
            msg_iat_time = self._safe_log_scale_normalize(msg_iat_time)
        
        msg_iat_times.append(msg_iat_time)

        msg_packet_numbers.append(1)  # 单包消息，包数为1

        msg_avg_payload_sizes.append(msg_payload_size)  # 单包消息，平均载荷大小等于载荷大小

        msg_durations.append(0)  # 单包消息，持续时间为0秒

        return pkt_timestamp
        
    def _create_multiple_pkt_msg_return_msg_timestamp(self, current_bulk_direction, current_bulk_bytes, current_bulk_timestamps, prev_msg_timestamp, 
                                                      msg_directions, msg_payload_sizes, msg_iat_times, msg_packet_numbers, msg_avg_payload_sizes, msg_durations):
        msg_direction_value = self._parse_direction_value(current_bulk_direction)
        msg_directions.append(msg_direction_value)

        msg_payload_size = sum(current_bulk_bytes) / float(self.mtu_normalize)  # 归一化到0-1
        # msg_payload_size = self._safe_log_scale_normalize(sum(current_bulk_bytes))
        msg_payload_sizes.append(msg_payload_size)

        # Zeek Flowmeter插件里面，pkt_timestamp的时间单位是秒，无需转换。
        # packet_timestamp_vector 以Unix时间戳格式记录flow中各数据包的到达时间。
        # 查看 https://gitee.com/seu-csqjxiao/zeek-flowmeter/tree/seu-devel
        if prev_msg_timestamp == 0:
            # 第一个消息，时间间隔为0
            msg_iat_time = 0.0
        else:
            # message 的到达时间定义为 bulk 中第一个数据包的时间戳
            msg_iat_time = current_bulk_timestamps[0] - prev_msg_timestamp
            msg_iat_time = self._safe_log_scale_normalize(msg_iat_time)
        
        msg_iat_times.append(msg_iat_time)

        msg_packet_numbers.append(len(current_bulk_bytes))  # 多包消息，包数为 bulk 内包数

        # 多包消息，平均载荷大小
        avg_payload_size = sum(current_bulk_bytes) / len(current_bulk_bytes) / float(self.mtu_normalize)
        # avg_payload_size = self._safe_log_scale_normalize(sum(current_bulk_bytes) / len(current_bulk_bytes))
        msg_avg_payload_sizes.append(avg_payload_size)  

        # 多包消息，持续时间设定为 bulk 内最后一个包的时间戳 - 第一个包的时间戳，时间单位是秒
        msg_duration = current_bulk_timestamps[-1] - current_bulk_timestamps[0]
        msg_durations.append(msg_duration)

        # 返回该消息的时间戳（即 bulk 中最后一个包的时间戳）
        return current_bulk_timestamps[-1] 

    def _process_categorical_features(self):
        """
        处理类别型特征（Label Encoding → Embedding 输入）：

        每一个类别字段（categorical column）都会经历以下处理流程：

        ------------------------------------------------------------
        ① 若列存在：
            - 将该列的类别值映射为整数 ID（Label Encoding）
            - ID 从 1 开始，0 用作 OOV（Out-Of-Vocabulary）
            - 按频率排序，仅保留 Top-K 类别，减少稀疏性
            - 将映射后的整数加入 categorical_data（一个列表）

        ② 若列缺失（例如 flow_df 中没有这个列）：
            - 打印 warning
            - 跳过该列（不会加入 categorical_data）
            - 这意味着 **该列不会出现在最终 categorical_features 矩阵中**

        ⚠️ 因此：
            categorical_features 的最终形状为：
                [num_flows, num_effective_categorical_columns]

        其中：
            num_effective_categorical_columns ≤ len(self.categorical_columns)

        （只统计实际存在、成功处理的列）

        这是预期行为：如果某些特征在实际数据集中不存在，就不会强制加入 0 列。
        这样可以保持 embedding 的维度与有效数据一致，避免全 0 噪声列。

        ------------------------------------------------------------
        最终输出：
            self.categorical_val2idx_mappings: dict
                { column_name → { category_value → index_id } }

            self.categorical_features: LongTensor of shape
                [N, C]
                N = 流数量（flow count）
                C = 实际存在的类别特征列数

            每列存储一个整数类别 ID（从 0 开始，0 为 OOV），
            之后会送入 nn.Embedding 作为输入。
        """
        
        use_train_mappings = hasattr(self, 'use_train_mappings') and self.use_train_mappings
        if use_train_mappings:
            logger.info("使用训练集的类别映射和有效类别列名，跳过独立映射计算")
            self.categorical_columns_effective = self.train_categorical_columns_effective
        else:
            self.categorical_val2idx_mappings = {}      # 每列的 {类别值 → index} 映射
            self.categorical_columns_effective = [] # 获得存在的有效categorical_columns的序列，确保是数据层和模型层的类别列名顺序完全一致
                    
        categorical_data = []               # 存储每列的整数 ID 序列
        
        # 针对不同字段的类别数量上限（保持与 config 的字段统计一致）
        TOP_K_MAP = {
            "ssl_cipher": 50,       # cipher 非常多，限制为 top 50
            "conn_history": 20,     # 会话状态序列 ShADd 非常稀疏
            "ssl_version": 4,       # TLS1.0~1.3
            "ssl_curve": 5,
            "ssl_next_protocol": 10,
            "dns_qtype": 30,
            "dns_rcode_name": 20,
        }
        default_top_k = 20
        
        for col in self.categorical_columns:
             # ⭐名称对齐，将列名中非法的"."字符替换成"_"，否则后续模型会报错。
            clean_col = col.replace(".", "_") 

            # ——————————————————————————
            # 0. 列不存在 → 跳过。注意：
            # categorical_features 的列数 = 实际存在的类别列数量。
            # 若某列在 flow_df 中缺失，则跳过该列，不会强制加入空列。
            # ——————————————————————————
            if col not in self.flow_df.columns:
                logger.warning(f"类别特征缺失列 {col}，跳过")
                continue

            # ——————————————————————————
            # 1. 填补空值并转string（统一形式）
            # ——————————————————————————
            series = (
                self.flow_df[col]   # 读取时依旧用原始 col
                .fillna("OOV")      # 所有缺失值归入“未知类别”
                .astype(str)
                .str.strip()
            )

            # ——————————————————————————
            # 2. 统计频率 → 选择 Top-K 类别
            #    这是 Embedding 处理高基数类别的通用方式
            # ——————————————————————————
            counter = Counter(series)

            # 最常见的前 K 个类别
            most_common = counter.most_common(TOP_K_MAP.get(clean_col, default_top_k))

            # 仅保留类别名称
            keep_values = [v for v, _ in most_common]
            # 确保至少保留 1 个真实分类
            if len(keep_values) == 0:
                keep_values = ["OOV"]
                
            # ——————————————————————————
            # 3. 建立类别 → index 映射
            #    index 从 1 开始（0 用作 OOV）
            # ——————————————————————————            
            if use_train_mappings:
                mapping = self.categorical_val2idx_mappings[clean_col] # 直接查询clean_col的val2idx
            else:
                mapping = {v: (i + 1) for i, v in enumerate(keep_values)} # 建立映射
                mapping["OOV"] = 0
                self.categorical_val2idx_mappings[clean_col] = mapping

            # ——————————————————————————
            # 4. 将每个值映射为整数 ID
            # ——————————————————————————
            mapped = [mapping.get(v, 0) for v in series]
            categorical_data.append(mapped)
            
            if not use_train_mappings:
                self.categorical_columns_effective.append(clean_col)
            
        # ——————————————————————————
        # 5. 将所有列堆叠成矩阵：N × C
        #    N = flow 数量
        #    C = 类别型特征列数量
        # ——————————————————————————
        if categorical_data:
            # zip(*list_of_lists) = 按列拼为按行
            self.categorical_features = torch.tensor(
                list(zip(*categorical_data)), dtype=torch.long
            )
        else:
            # 没有类别型特征
            self.categorical_features = torch.zeros(
                (len(self.flow_df), 0), dtype=torch.long
            )

        # 将全局categorical_columns的列名清洗标准化，将列名中非法的"."字符替换成"_"，否则后续模型会报错。
        # self.categorical_columns = [col.replace(".", "_") for col in self.categorical_columns] # 最好是别做这种替换，导致这个函数没法重入
        logger.info(f"[Dataset] 所有类别列（{len(self.categorical_columns)}列）：{self.categorical_columns}")
        logger.info(f"[Dataset] 列名清洗后的有效类别列（{len(self.categorical_columns_effective)}列）: {self.categorical_columns_effective}")
        # 校验类别特征列数是否一致
        if self.categorical_features.numel() > 0:  # 避免空 tensor 情况
            matrix_num_cols = self.categorical_features.shape[1]
        else:
            matrix_num_cols = 0

        if len(self.categorical_columns_effective) != matrix_num_cols:
            logger.warning(
                f"[Dataset] ⚠类别特征列数不匹配：有效列 {len(self.categorical_columns_effective)} "
                f"!= 矩阵列数 {matrix_num_cols}，请检查 categorical_features 构造逻辑"
            )
        
    def _process_text_features(self):
        """处理文本特征"""
        self.text_features = []
        
        text_columns = [
            self.cfg.data.text_features.dns_query,
            self.cfg.data.text_features.dns_answers,
            self.cfg.data.text_features.ssl_server_name,
            self.cfg.data.text_features.cert0_subject,
            self.cfg.data.text_features.cert0_issuer, 
            self.cfg.data.text_features.cert0_san_dns,
            self.cfg.data.text_features.cert1_subject,
            self.cfg.data.text_features.cert1_issuer,
            self.cfg.data.text_features.cert1_san_dns,
            self.cfg.data.text_features.cert2_subject,
            self.cfg.data.text_features.cert2_issuer,
            self.cfg.data.text_features.cert2_san_dns,
        ]
        
        def safe_to_str(v):
            """更安全的字符串转换函数"""
            if v is None or pd.isna(v):
                return ""
            
            if isinstance(v, list):
                # 递归处理列表中的每个元素
                return " ".join([safe_to_str(x) for x in v if x is not None and not pd.isna(x)])

            if isinstance(v, dict):
                # 处理字典结构
                parts = []
                for k, val in v.items():
                    if val is not None and not pd.isna(val):
                        parts.append(f"{k}_{safe_to_str(val)}")
                return " ".join(parts) if parts else ""

            if isinstance(v, str):
                # 对于字符串，直接返回（不再尝试解析）
                return v.strip()

            # 其他类型直接转换为字符串
            return str(v) if v is not None else ""   
        
        # 添加详细的进度条
        with tqdm(total=len(self.flow_df), desc="处理文本特征", position=0, leave=True) as pbar:
            for idx, row in self.flow_df.iterrows():
                text_data = {}

                for col in text_columns:
                    if col in row.index:
                        value = row[col]
                        text_data[col] = safe_to_str(value) if pd.notna(value) else ""
                    else:
                        text_data[col] = ""

                self.text_features.append(text_data)
                pbar.update(1)
    
    def _process_numeric_features(self):
        """处理数值特征，只负责：
        (1) 检查数值列
        (2) 训练集统计 numeric_stats
        (3) 调用 apply_numeric_stats() 做归一化
        """        
        # 只使用 flow_features，排除 domain_name_embedding_features
        flow_columns = self.cfg.data.tabular_features.numeric_features.flow_features
        x509_columns = self.cfg.data.tabular_features.numeric_features.x509_features
        dns_columns = self.cfg.data.tabular_features.numeric_features.dns_features

        numeric_columns = flow_columns + x509_columns + dns_columns

        # 添加列存在性检查
        available_numeric_columns = [col for col in numeric_columns if col in self.flow_df.columns]
        
        # ---------- Step 1: 保证列为 float ----------
        for col in available_numeric_columns:
            if self.flow_df[col].dtype == 'object':
                logger.warning(f"数值列 '{col}' 是对象类型，正在转换为数值类型...")
                # 强制转换为数值类型，无法转换的设为NaN
                self.flow_df[col] = pd.to_numeric(self.flow_df[col], errors='coerce')
                
                # 检查转换后的NaN比例
                nan_ratio = self.flow_df[col].isna().mean()
                if nan_ratio > 0.5:
                    logger.warning(f"列 '{col}' 有 {nan_ratio:.1%} 的值无法转换为数值")

        # ---------- Step 2: 训练阶段计算 numeric_stats ----------
        if self.is_training:
            self.numeric_stats = {}
            
            for col in available_numeric_columns:
                # 过滤掉异常值和NaN
                col_data = self.flow_df[col].dropna()
                
                if not col_data.empty:
                    if col_data.dtype.kind in 'iufc':  # 整数、无符号整数、浮点数、复数
                        # 移除极端异常值
                        q1 = col_data.quantile(0.25)
                        q3 = col_data.quantile(0.75)
                        iqr = q3 - q1
                        lower_bound = q1 - 1.5 * iqr
                        upper_bound = q3 + 1.5 * iqr
                        
                        filtered_data = col_data[(col_data >= lower_bound) & (col_data <= upper_bound)]
                        
                        if not filtered_data.empty:
                            self.numeric_stats[col] = {
                                'mean': filtered_data.mean(),
                                'std': max(filtered_data.std(), 1e-6)
                            }
                        else:
                            self.numeric_stats[col] = {
                                'mean': col_data.mean() if not col_data.empty else 0,
                                'std': max(col_data.std() if not col_data.empty else 1, 1e-6)
                            }
                    else:
                        logger.warning(f"列 '{col}' 不是数值类型，使用默认统计信息")
                        self.numeric_stats[col] = {'mean': 0, 'std': 1}
                else:
                    logger.warning(f"列 '{col}' 没有有效数值数据，使用默认统计信息")
                    self.numeric_stats[col] = {'mean': 0, 'std': 1}
                    
            # 训练集立即归一化
            self.apply_numeric_stats()
                
        else:
            # 验证/测试：等待注入训练统计
            self.numeric_stats = None
            self.numeric_features = None # 后面会从available_numeric_columns真正读取数据内容到self.numeric_features，现在先置空
            logger.info("验证集/测试集将等待训练集统计信息注入后再归一化。")
        
    def apply_numeric_stats(self):
        """根据训练集 numeric_stats 对 numeric_features 做归一化（带进度条）"""

        if not hasattr(self, "numeric_stats"):
            raise RuntimeError(
                "numeric_stats 未设置！请先从训练集注入统计信息。"
            )
            
        # ---------- 若 numeric_stats 全默认 {0,1}，则跳过 ----------
        if self.numeric_stats is not None:
            all_default = all(
                stats.get('mean',0)==0 and stats.get('std',1)==1
                for stats in self.numeric_stats.values()
            )
            if all_default:
                logger.info("数值特征使用默认 {0,1} 统计信息，跳过归一化。")
                # 直接使用原始特征，不做任何变换
                all_numeric_columns = (
                    self.cfg.data.tabular_features.numeric_features.flow_features +
                    self.cfg.data.tabular_features.numeric_features.x509_features +
                    self.cfg.data.tabular_features.numeric_features.dns_features
                )
                self.numeric_features = self.flow_df[all_numeric_columns].fillna(0.0).values.tolist()
                return
            
        # 只使用 numeric_features，排除 domain_name_embedding_features
        all_numeric_columns = (
            self.cfg.data.tabular_features.numeric_features.flow_features +
            self.cfg.data.tabular_features.numeric_features.x509_features +
            self.cfg.data.tabular_features.numeric_features.dns_features
        )
        # 列存在性检查
        available_numeric_columns = [col for col in all_numeric_columns if col in self.flow_df.columns]
        
        new_numeric_features = []

        with tqdm(total=len(self.flow_df), desc="应用数值特征归一化", position=0, leave=True) as pbar:
            for idx, row in self.flow_df.iterrows():
                numeric_row = []

                for col in available_numeric_columns:
                    value = row[col]

                    # 获取训练时统计信息
                    stats = self.numeric_stats.get(col, {'mean': 0, 'std': 1})
                    mean = stats['mean']
                    std = max(stats['std'], 1e-6)

                    # 转换 + 归一化
                    if pd.isna(value):
                        normalized = 0.0
                    else:
                        try:
                            v = float(value)
                            normalized = (v - mean) / std
                            normalized = float(np.clip(normalized, -5, 5))
                        except:
                            normalized = 0.0

                    numeric_row.append(normalized)

                new_numeric_features.append(numeric_row)
                pbar.update(1)

        self.numeric_features = new_numeric_features

    def _process_domain_embedding_features(self):
        """处理域名嵌入特征（概率列表）"""
        self.domain_embedding_features = []
        
        domain_columns = self.cfg.data.domain_name_embedding_features.column_list
        
        # 添加列存在性检查
        available_columns = [col for col in domain_columns if col in self.flow_df.columns]
        
        if not available_columns:
            logger.warning("配置了域名嵌入特征，但数据中未找到对应的列")
            self.domain_embedding_enabled = False  # 禁用该视图
            return
    
        # 详细的维度信息
        logger.info(f"可用的域名嵌入特征列长度: {len(available_columns)}，具体域名嵌入特征列：{available_columns}")
        label_id_map = ConfigManager.read_session_label_id_map(self.cfg.data.dataset)
        logger.info(f"嵌入的概率列表长度（类别数）: {self.prob_list_length}，具体类别映射: {label_id_map}")
        logger.info(f"期望的域名嵌入特征总维度: {len(available_columns)} × {self.prob_list_length} = {len(available_columns) * self.prob_list_length}")
        
        if not available_columns:
            logger.warning("没有找到可用的域名嵌入特征列")

        with tqdm(total=len(self.flow_df), desc="处理域名嵌入特征", position=0, leave=True) as pbar:
            for idx, row in self.flow_df.iterrows():
                embedding_data = []
                
                for col in available_columns:
                    value = row[col]
                    
                    if pd.isna(value) or value is None or value == '[]':
                        # 如果为空，创建默认的概率分布（均匀分布或零向量）
                        default_embedding = [0.0] * self.prob_list_length
                        embedding_data.extend(default_embedding)
                    else:
                        try:
                            # 解析概率列表字符串
                            if isinstance(value, str) and value.startswith('['):
                                prob_list = ast.literal_eval(value)
                            else:
                                prob_list = self._safe_parse_list(str(value))
                            
                            # 确保概率列表长度正确
                            if len(prob_list) != self.prob_list_length:
                                logger.debug(f"列 {col} 的概率列表长度不匹配: 期望{self.prob_list_length}, 实际{len(prob_list)}")
                                # 调整长度：截断或填充
                                if len(prob_list) > self.prob_list_length:
                                    prob_list = prob_list[:self.prob_list_length]
                                else:
                                    prob_list.extend([0.0] * (self.prob_list_length - len(prob_list)))
                            
                            # 添加到嵌入数据中
                            embedding_data.extend(prob_list)
                            
                        except Exception as e:
                            logger.debug(f"解析域名嵌入特征失败 {col}: {e}")                            
                            default_embedding = [0.0] * self.prob_list_length # 使用默认值
                            embedding_data.extend(default_embedding)
                
                self.domain_embedding_features.append(embedding_data)
                pbar.update(1)
    
    def _safe_parse_list(self, list_str: str) -> List:
        """安全解析列表字符串，处理各种格式"""
        if pd.isna(list_str) or list_str == '[]' or list_str == '':
            return []
        
        # 如果已经是列表类型，直接返回
        if isinstance(list_str, list):
            return list_str
        
        try:
            # 首先尝试 ast.literal_eval
            if isinstance(list_str, str) and list_str.startswith('['):
                return ast.literal_eval(list_str)
            else:
                # 对于非列表格式的字符串，直接返回原字符串（作为单元素列表）
                # 或者根据具体需求处理
                return [str(list_str)]
        except (ValueError, SyntaxError):
            # 如果解析失败，尝试其他格式
            try:
                # 尝试提取所有字母数字和常见符号
                import re
                # 匹配包含字母、数字、点、连字符、下划线的单词
                words = re.findall(r'[a-zA-Z0-9.-_]+', str(list_str))
                return words if words else []
            except:
                return []
        except Exception as e:
            if self.debug_mode:
                logger.debug(f"解析列表字符串失败: {e}, 输入: {list_str[:100]}...")
            return []
            
    def __len__(self):
        return len(self.flow_df)
    
    def __getitem__(self, row_idx):
        row = self.flow_df.iloc[row_idx]
        
        # 序列数据（可选）
        if self.sequence_features_enabled:
            # _parse_sequence_row 的时候，已经确保了序列长度一致
            sequence_data = self.sequences[row_idx]

            directions = torch.tensor(sequence_data['directions'], dtype=torch.float32)
            payload_sizes = torch.tensor(sequence_data['payload_sizes'], dtype=torch.float32)
            iat_times = torch.tensor(sequence_data['iat_times'], dtype=torch.float32)
            packet_numbers = torch.tensor(sequence_data['packet_numbers'], dtype=torch.float32)
            avg_payload_sizes = torch.tensor(sequence_data['avg_payload_sizes'], dtype=torch.float32)
            durations = torch.tensor(sequence_data['durations'], dtype=torch.float32)
            sequence_mask = torch.tensor(sequence_data['sequence_mask'], dtype=torch.bool)

        else:
            # 创建空的序列特征占位符
            directions = torch.zeros(self.max_seq_length, dtype=torch.float32)
            payload_sizes = torch.zeros(self.max_seq_length, dtype=torch.float32)
            iat_times = torch.zeros(self.max_seq_length, dtype=torch.float32)
            packet_numbers = torch.zeros(self.max_seq_length, dtype=torch.float32)
            avg_payload_sizes = torch.zeros(self.max_seq_length, dtype=torch.float32)
            durations = torch.zeros(self.max_seq_length, dtype=torch.float32)            
            sequence_mask = torch.zeros(self.max_seq_length, dtype=torch.bool)
        
        # 文本数据（可选）
        if self.text_features_enabled:
            text_data = self.text_features[row_idx]
            # 保留合并原始文本
            combined_text = " ".join([text for text in text_data.values() if text.strip()])
            ssl_server_name = text_data.get(self.cfg.data.text_features.ssl_server_name, "")
            dns_query = text_data.get(self.cfg.data.text_features.dns_query, "")
            cert0_subject = text_data.get(self.cfg.data.text_features.cert0_subject, "")
            cert0_issuer = text_data.get(self.cfg.data.text_features.cert0_issuer, "")
        else:
            text_data = {}
            combined_text = ""
            ssl_server_name = ""
            dns_query = ""
            cert0_subject = ""
            cert0_issuer = ""            
                
        # 域名嵌入特征的概率列表（可选）
        if self.domain_embedding_enabled:
            domain_embedding = self.domain_embedding_features[row_idx]
            actual_domain_dim = len(domain_embedding)
            expected_domain_dim = len(self.cfg.data.domain_name_embedding_features.column_list) * self.prob_list_length
            if len(domain_embedding) != expected_domain_dim:
                logger.warning(f"域名嵌入特征维度不匹配: 期望{expected_domain_dim}, 实际{actual_domain_dim}")
                # 自动调整维度
                if len(domain_embedding) > expected_domain_dim:
                    domain_embedding = domain_embedding[:expected_domain_dim]
                else:
                    domain_embedding.extend([0.0] * (expected_domain_dim - len(domain_embedding)))
            
            domain_embedding_features = torch.tensor(domain_embedding, dtype=torch.float32)
        else:
            # 创建空的域名嵌入特征
            domain_embedding_features = torch.zeros(0, dtype=torch.float32)

        # 类别型特征（必选）
        if self.categorical_features_enabled:
            categorical_features = self.categorical_features[row_idx]
        else:
            categorical_features = torch.zeros(0, dtype=torch.long)

        # 数值型特征（必选）
        numeric_features = torch.tensor(self.numeric_features[row_idx], dtype=torch.float32)
        all_numeric_columns = (
            self.cfg.data.tabular_features.numeric_features.flow_features +
            self.cfg.data.tabular_features.numeric_features.x509_features +
            self.cfg.data.tabular_features.numeric_features.dns_features
        )
        available_numeric_columns = [col for col in all_numeric_columns if col in self.flow_df.columns]
        expected_numeric_dim = len(available_numeric_columns)

        if len(numeric_features) != expected_numeric_dim:
            logger.warning(f"数值特征维度不匹配: 期望{expected_numeric_dim}, 实际{len(numeric_features)}")
            # 自动调整
            if len(numeric_features) > expected_numeric_dim:
                numeric_features = numeric_features[:expected_numeric_dim]
            else:
                padding = torch.zeros(expected_numeric_dim - len(numeric_features))
                numeric_features = torch.cat([numeric_features, padding])

        # 善意/恶意的二分类标签列
        is_malicious_value = row[self.is_malicious_column].astype(int)
        is_malicious_label = torch.tensor(is_malicious_value, dtype=torch.float32)
        
        # 多分类标签列
        multiclass_label_string = row[self.multiclass_label_column]
        is_malicious_value_alt = self._compute_is_malicious_label(multiclass_label_string)
        if is_malicious_value != is_malicious_value_alt:
            raise ValueError(
                f"is_malicious 不一致: csv={is_malicious_value}, "
                f"derived={is_malicious_value_alt}, "
                f"label={multiclass_label_string}"
            )
        attack_family_vec, attack_type_vec = self._compute_attack_family_and_type_labels(multiclass_label_string)

        data = {
            # 序列特征（根据启用状态）
            'directions': directions,
            'payload_sizes': payload_sizes,
            'iat_times': iat_times,
            'packet_numbers': packet_numbers,
            'avg_payload_sizes': avg_payload_sizes,
            'durations': durations,
            'sequence_mask': sequence_mask, # 有效token掩码

            # 文本特征（根据启用状态）
            'ssl_server_name': ssl_server_name,
            'dns_query': dns_query,
            'cert0_subject': cert0_subject,
            'cert0_issuer': cert0_issuer,
            'combined_text': combined_text, # 合并文本（给BERT使用）
                                                
            # 域名嵌入特征（根据启用状态）
            'domain_embedding_features': domain_embedding_features,

            # 类别型特征（必选）
            'categorical_features': categorical_features,

            # 数值型特征（必选）
            'numeric_features': numeric_features,

            # 标签
            'is_malicious_label': is_malicious_label,
            'multiclass_label_string': multiclass_label_string,
            'attack_family_label': attack_family_vec,
            'attack_type_label': attack_type_vec,
            
            # 元数据
            'uid': row.get('uid', ''),
            'idx': row_idx,

            # 添加视图启用标志
            'sequence_features_enabled': self.sequence_features_enabled,
            'text_features_enabled': self.text_features_enabled,
            'domain_embedding_enabled': self.domain_embedding_enabled,
        }

        # 最终验证：确保所有tensor都没有NaN
        for key, value in data.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    logger.error(f"严重错误: idx={row_idx}, key={key} 仍然包含NaN值")
                    # 强制修复
                    data[key] = torch.nan_to_num(value, nan=0.0, posinf=1.0, neginf=-1.0)

        # print("DEBUG numeric len:", len(numeric_features[row_idx]))
        # print("DEBUG domain len:", len(self.domain_embedding_features[row_idx]) if self.domain_embedding_enabled else 0)
        # print("DEBUG categorical len:", len(self.categorical_features[row_idx]) if self.categorical_features is not None else 0)

        return data
    
    def _compute_is_malicious_label(self, multiclass_label_string):
        label_norm = str(multiclass_label_string).strip().lower()
        return 0 if label_norm == "benign" else 1

    def _compute_attack_family_and_type_labels(self, multiclass_label_string):
        """
        根据 labels 配置中定义的层级语义（attack_type → attack_family），
        将原始多分类字符串标签（如 'DoS Hulk' / 'PortScan' / 'BENIGN'）
        解析为 attack_type 与 attack_family 两个层级的监督信号。

        - attack_type_labels：
            * 标签空间：labels.attack_type.classes
            * 输出：长度为 num_types 的 0/1 向量（one-hot / multi-hot）
            * 若 label 属于某个 attack_type，则对应位置为 1；
            若为 benign 或未知类别，则为全 0（not applicable）

        - attack_family_labels：
            * 标签空间：labels.attack_family.classes
            * 输出：长度为 num_families 的 0/1 向量（one-hot / multi-hot）
            * 若 label 本身是 family，直接置位；
            若 label 是 attack_type，则通过 parent_mapping 上溯到 family；
            若为 benign 或无法映射，则为全 0（not applicable）

        说明：
        - benign 流量在 attack_family / attack_type 层级均返回全 0 向量，
        表示其不属于 malicious 子树中的任何攻击类别。
        - 该 0/1 表示方式用于支持 OVR + BCE 的多二分类建模，
        并保证 Dataset / Model / Loss 之间的标签语义一致性。
        """
        label = str(multiclass_label_string).strip()
        label_norm = label.lower()

        # 统一 benign 处理
        if label_norm == "benign":
            return (
                torch.zeros(len(self.attack_family_classes)) if self.attack_family_classes else torch.zeros(0),
                torch.zeros(len(self.attack_type_classes)) if self.attack_type_classes else torch.zeros(0)
            )

        family_vec = None
        type_vec = None

        # =========================
        # attack_type（更底层）
        # =========================
        if self.attack_type_classes is None:
            type_vec = torch.zeros(0)
        else:
            type_vec = torch.zeros(
                len(self.attack_type_classes),
                dtype=torch.float32
            )
            if label in self.attack_type_classes:
                idx = self.attack_type_classes.index(label)
                type_vec[idx] = 1.0

        # =========================
        # attack_family（通过映射）
        # =========================
        if self.attack_family_classes is None:
            family_vec = torch.zeros(0)
        else:
            family_vec = torch.zeros(
                len(self.attack_family_classes),
                dtype=torch.float32
            )

            fam_name = None

            # 情况 1：label 本身就是 family
            if label in self.attack_family_classes:
                fam_name = label

            # 情况 2：label 是 attack_type → 映射到 family
            elif (
                self.attack_type_parent_mapping is not None
                and label in self.attack_type_parent_mapping
            ):
                fam_name = self.attack_type_parent_mapping[label]

            if fam_name is not None:
                fam_idx = self.attack_family_classes.index(fam_name)
                family_vec[fam_idx] = 1.0
            
        return family_vec, type_vec


class MultiviewFlowDataModule(pl.LightningDataModule):
    """多视图流量数据模块"""
    
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        # 缓存常用配置
        self.flow_data_path = self.cfg.data.flow_data_path
        self.session_split_path = self.cfg.data.session_split.session_split_path
        self.batch_size = self.cfg.data.batch_size
        self.num_workers = self.cfg.data.num_workers

        # 缓存会话划分配置
        self.split_config = cfg.data.session_split
        self.split_column = self.split_config.split_column
        self.flow_uid_list_column = self.split_config.flow_uid_list_column
        self.train_split = self.split_config.train_split
        self.validate_split = self.split_config.validate_split
        self.test_split = self.split_config.test_split

        self.train_dataset = None
        self.validate_dataset = None
        self.test_dataset = None

        # 其他配置
        self.is_malicious_column = cfg.data.is_malicious_column
        self.multiclass_label_column = cfg.data.multiclass_label_column
        self.debug_mode = cfg.debug.debug_mode
                
        
    def prepare_data(self):
        # 下载或准备数据
        pass
    
    def setup(self, stage=None):
        # 处理不同的stage输入类型
        if stage is None:
            stage_name = "fit" # "fit" 是 Lightning 的规范名称，含义是 “训练 + 验证”
        elif hasattr(stage, 'value'):  # 如果是枚举类型
            stage_name = stage.value.lower()  # 直接使用枚举的value属性
        elif isinstance(stage, str):
            stage_name = stage.lower()
        else:
            # 其他情况，尝试转换为字符串
            stage_name = str(stage).lower()
        
        # 检查是否已经初始化
        if self._is_already_setup(stage_name):
            logger.info(f"{stage_name} 阶段数据集已初始化，跳过重复setup")
            return

        logger.info(f"数据模块 setup 阶段: {stage_name}")        

        # 一次性读取所有必要数据（支持缓存）
        self._load_data_with_cache(stage_name)

        if stage_name == "fit":
            self._create_datasets(stage_name)

        if stage in (None, "fit") and self.train_dataset is not None and self.validate_dataset is not None:
            train_is_malicious_labels = self.train_dataset.is_malicious_labels
            val_is_malicious_labels = self.validate_dataset.is_malicious_labels
            logger.info("==========================================")            
            logger.info(f"[训练集] 正样本={sum(train_is_malicious_labels)}, 负样本={len(train_is_malicious_labels)-sum(train_is_malicious_labels)}, 比例={sum(train_is_malicious_labels)/len(train_is_malicious_labels):.4f}")
            logger.info(f"[验证集] 正样本={sum(val_is_malicious_labels)}, 负样本={len(val_is_malicious_labels)-sum(val_is_malicious_labels)}, 比例={sum(val_is_malicious_labels)/len(val_is_malicious_labels):.4f}")

        if stage in (None, "test") and self.test_dataset is not None:
            test_is_malicious_labels = self.test_dataset.is_malicious_labels
            logger.info("==========================================")            
            logger.info(f"[测试集] 正样本={sum(test_is_malicious_labels)}, 比例={sum(test_is_malicious_labels)/len(test_is_malicious_labels):.4f}")


    def _is_already_setup(self, stage_name):
        """检查数据集是否已经初始化"""
        if stage_name == 'fit':
            # fit阶段需要训练集和验证集都初始化
            return self.train_dataset is not None and self.validate_dataset is not None
        elif stage_name == 'validate':
            # validate阶段只需要验证集
            return self.validate_dataset is not None
        elif stage_name == 'test':
            # test阶段只需要测试集
            return self.test_dataset is not None
        else:
            # 其他情况返回False
            return False

    def read_large_csv_with_progress(self, filepath, description="读取数据", verbose=True):
        if not verbose:
            return pd.read_csv(filepath)

        logger.info(f"{description} 从 {filepath}...")
        file_size = os.path.getsize(filepath) / (1024**3)
        logger.info(f"文件大小: {file_size:.2f} GB")

        sample_df = pd.read_csv(filepath, nrows=5)
        logger.info(f"检测到 {len(sample_df.columns)} 列，开始分块读取...")

        with open(filepath, "r") as f:
            total_rows = sum(1 for _ in f) - 1

        chunk_size = 100_000  # 每次读取10万行
        chunks = []

        # ⭐ 关键：只有 rank 0 才开 tqdm，否则每个gpu都打印进度条，会显示错乱
        if self.trainer is not None:
            is_rank_zero = self.trainer.is_global_zero
        else:
            is_rank_zero = True  # 非 Lightning 场景兜底

        pbar = tqdm(
            total=total_rows,
            desc=description,
            unit="rows",
            dynamic_ncols=True,
            leave=True,
            disable=not is_rank_zero,   # ⭐⭐ 关键
        )

        for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
            chunks.append(chunk)
            pbar.update(len(chunk))

        pbar.close()

        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"{description} 完成! 数据形状: {df.shape}")
        return df
    
    def _load_data_with_cache(self, stage_name):
        """带缓存的数据加载"""
        # 如果flow_df已经存在，直接使用缓存的数据
        if hasattr(self, 'flow_df') and self.flow_df is not None:
            logger.info("使用缓存的flow_df数据，跳过重复读取")
        else:
            # 只在第一次需要时读取数据
            flow_df = self.read_large_csv_with_progress(self.flow_data_path)
            # 增强数据质量检查
            self.flow_df = self._validate_data_quality(flow_df, stage=stage_name)  # ✅ 正确接收修改后的flow_df

        # 如果session_df已经存在，直接使用缓存的数据
        if hasattr(self, 'session_df') and self.session_df is not None:
            logger.info("使用缓存的session_df数据，跳过重复读取")
        else:
            # 只在第一次需要时读取session数据
            self.session_df = self.read_large_csv_with_progress(self.session_split_path)

            # 检查必要的列是否存在
            required_columns = [self.split_column, self.flow_uid_list_column]
            missing_columns = [col for col in required_columns if col not in self.session_df.columns]
            if missing_columns:
                raise ValueError(f"session_df缺少必要列: {missing_columns}")
            
            # 检查split值的有效性
            valid_splits = [self.train_split, self.validate_split, self.test_split]
            actual_splits = self.session_df[self.split_column].unique()
            invalid_splits = [split for split in actual_splits if split not in valid_splits]
            if invalid_splits:
                logger.warning(f"发现无效的split值: {invalid_splits}")
        
        
    def _validate_data_quality(self, df, stage="unknown"):
        """仅校验配置文件中使用到的列是否包含 NaN，不检查其它无关列。
        若 target 列存在 NaN，立即抛异常，要求用户处理。"""

        logger.info(f"检查 {stage} 阶段数据质量（仅检查 cfg 中引用的列）...")

        # ============ 1. 收集所有需要检查的列 ============
        required_columns = set()

        # is_malicious 和 multiclass_label 标签列（最关键）
        if self.is_malicious_column is not None:
            required_columns.add(self.is_malicious_column)
        if self.multiclass_label_column is not None:
            required_columns.add(self.multiclass_label_column)

        # 数值特征列
        if hasattr(self.cfg.data.tabular_features.numeric_features, "flow_features"):
            required_columns.update(self.cfg.data.tabular_features.numeric_features.flow_features)

        # 序列特征列（可选）
        if hasattr(self.cfg.data, "sequence_features") and self.cfg.data.sequence_features is not None:
            seq_cfg = self.cfg.data.sequence_features
            required_columns.update([
                seq_cfg.packet_direction,
                seq_cfg.packet_iat,
                seq_cfg.packet_payload,
            ])

        # 文本特征列（可选）
        if hasattr(self.cfg.data, "text_features") and self.cfg.data.text_features is not None:
            txt_cfg = self.cfg.data.text_features
            required_columns.update([
                txt_cfg.ssl_server_name,
                txt_cfg.dns_query,
                txt_cfg.cert0_subject,
                txt_cfg.cert0_issuer,
            ])

        # 域名嵌入特征列（可选）
        if hasattr(self.cfg.data, "domain_name_embedding_features"):
            if hasattr(self.cfg.data.domain_name_embedding_features, "column_list"):
                required_columns.update(self.cfg.data.domain_name_embedding_features.column_list)

        required_columns = list(required_columns)

        # ============ 2. 必须存在的列检查 ============
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ 数据缺少配置文件要求的列: {missing_cols}")

        # ============ 3. 只检查这些列中的 NaN ============
        df_sub = df[required_columns]
        nan_counts = df_sub.isna().sum()

        # 全部使用的列是否有 NaN
        if nan_counts.any():
            nan_cols = nan_counts[nan_counts > 0]

            # ---- 特殊处理: is_malicious 列出现 NaN → 直接报错 ----
            if nan_counts[self.is_malicious_column] > 0:
                raise ValueError(
                    f"❌ is_malicious 列 '{self.is_malicious_column}' 出现 NaN 值: {nan_counts[self.is_malicious_column]} 行。\n"
                    f"请在加载 CSV 前手动清洗数据，否则模型无法训练。"
                )

            # 其它列的 NaN → 给 warning，让用户处理
            logger.warning("⚠️ 以下使用到的特征列包含 NaN：")
            for col, count in nan_cols.items():
                logger.warning(f"  {col}: {count} 个 NaN ({count / len(df) * 100:.2f}%)")

        logger.info(f"{stage} 阶段数据质量检查完成（仅检查 cfg 使用的列）")
        return df

    def _create_datasets(self, stage_name):
        split_mode = self.cfg.data.get("split_mode", "session").lower()
        logger.info(f"数据划分模式：{split_mode}")

        if split_mode == "session":
            logger.info("使用基于 session_df 的会话级划分策略")
            assert self.split_column in self.session_df.columns
            assert "uid" in self.flow_df.columns

            # 使用session_df会话进行数据集划分
            train_session_df = self.session_df[self.session_df[self.split_column] == self.train_split]
            validate_session_df = self.session_df[self.session_df[self.split_column] == self.validate_split]
            test_session_df = self.session_df[self.session_df[self.split_column] == self.test_split]

            # 提取所有训练flow的UID
            train_flow_uids = []
            for uid_list in train_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        train_flow_uids.extend(uids)
                    except:
                        continue

            # 提取所有验证flow的UID
            validate_flow_uids = []
            for uid_list in validate_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        validate_flow_uids.extend(uids)
                    except:
                        continue

            # 提取所有测试flow的UID
            test_flow_uids = []
            for uid_list in test_session_df[self.flow_uid_list_column]:
                if pd.notna(uid_list) and uid_list != '[]':
                    try:
                        uids = ast.literal_eval(uid_list) if isinstance(uid_list, str) else uid_list
                        test_flow_uids.extend(uids)
                    except:
                        continue

            # 根据UID划分flow数据集
            train_flow_df = self.flow_df[self.flow_df['uid'].isin(train_flow_uids)]
            validate_flow_df = self.flow_df[self.flow_df['uid'].isin(validate_flow_uids)]
            test_flow_df = self.flow_df[self.flow_df['uid'].isin(test_flow_uids)]

        elif split_mode == "flow":
            
            logger.info("使用基于 flow_df 的随机逐流划分策略")

            self.flow_train_ratio = self.cfg.data.flow_split.train_ratio
            self.flow_validate_ratio = self.cfg.data.flow_split.validate_ratio
            self.flow_test_ratio = self.cfg.data.flow_split.test_ratio

            is_malicious_labels = self.flow_df[self.is_malicious_column].values

            from sklearn.model_selection import train_test_split

            train_df, temp_df = train_test_split(
                self.flow_df,
                test_size=1 - self.flow_train_ratio,
                stratify=is_malicious_labels,
                random_state=42,
                shuffle=False
            )

            temp_is_malicious_labels = temp_df[self.is_malicious_column].values
            val_ratio_in_temp = self.flow_validate_ratio / (self.flow_validate_ratio + self.flow_test_ratio)

            validate_df, test_df = train_test_split(
                temp_df,
                test_size=1 - val_ratio_in_temp,
                stratify=temp_is_malicious_labels,
                random_state=42,
                shuffle=False
            )

            train_flow_df = train_df
            validate_flow_df = validate_df
            test_flow_df = test_df
            
        else:
            raise ValueError(
                f"❌ 无效的 split_mode='{split_mode}'。"
                f"请在配置文件中使用 'session' 或 'flow'"
            )
                
        # 创建训练集 - 传入完整的 cfg 对象
        logger.info(f">>>> 创建训练集MultiviewFlowDataset ...")
        self.train_dataset = MultiviewFlowDataset(train_flow_df, self.cfg, is_training=True)
        
        # 创建验证集：在构造函数中注入训练集映射
        logger.info(f">>>> 创建验证集MultiviewFlowDataset ...")
        self.validate_dataset = MultiviewFlowDataset(
            validate_flow_df, 
            self.cfg, 
            is_training=False,
            train_categorical_mappings=self.train_dataset.categorical_val2idx_mappings,
            train_categorical_columns_effective=self.train_dataset.categorical_columns_effective
        )
        
        # 创建测试集：在构造函数中注入训练集映射
        logger.info(f">>>> 创建测试集MultiviewFlowDataset ...")
        self.test_dataset = MultiviewFlowDataset(
            test_flow_df, 
            self.cfg, 
            is_training=False,
            train_categorical_mappings=self.train_dataset.categorical_val2idx_mappings,
            train_categorical_columns_effective=self.train_dataset.categorical_columns_effective
        )

        self._validate_categorical_consistency()
        
        # 应该确保验证集使用训练集的统计信息
        if hasattr(self.train_dataset, 'numeric_stats'):
            self.validate_dataset.numeric_stats = self.train_dataset.numeric_stats  # ✅ 传递统计信息
            self.test_dataset.numeric_stats = self.train_dataset.numeric_stats  # ✅ 传递统计信息
            
            # 重新应用归一化
            logger.info(f"✅ 重新应用数值特征标准化: 验证集使用训练集的统计信息")
            logger.info(f"   覆盖特征数量: {len(self.train_dataset.numeric_stats)}")
            self.validate_dataset.apply_numeric_stats()
            
            logger.info(f"✅ 重新应用数值特征标准化: 测试集使用训练集的统计信息")
            logger.info(f"   覆盖特征数量: {len(self.train_dataset.numeric_stats)}")
            self.test_dataset.apply_numeric_stats()            
            
        else:
            logger.info("🔄 使用默认统计信息进行数值特征标准化")
            # 为每个数值列创建默认统计信息
            flow_columns = self.cfg.data.tabular_features.numeric_features.flow_features
            default_stats = {}
            for col in flow_columns:
                default_stats[col] = {'mean': 0, 'std': 1}
            
            self.validate_dataset.numeric_stats = default_stats
            self.test_dataset.numeric_stats = default_stats
            logger.info(f"   默认统计信息覆盖特征数量: {len(default_stats)}")
        
        logger.info(f"数据集划分: 训练集 {len(train_flow_df)}, 验证集 {len(validate_flow_df)}, 测试集 {len(test_flow_df)}")

    # 验证映射一致性
    def _validate_categorical_consistency(self):
        """验证所有数据集的类别映射一致性"""
        train_mappings = self.train_dataset.categorical_val2idx_mappings
        val_mappings = self.validate_dataset.categorical_val2idx_mappings
        test_mappings = self.test_dataset.categorical_val2idx_mappings
        
        assert train_mappings == val_mappings == test_mappings, "类别映射不一致！"
        logger.info("✅ 所有数据集的类别映射验证一致")
            
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.validate_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True
        )