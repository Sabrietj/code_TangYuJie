import tqdm
import os, sys
import pandas as pd
import numpy as np
import json
import ast
from typing import Union, List
import logging
from collections import Counter
from transformers import BertTokenizer
from types import MappingProxyType
from typing import List, Tuple

# 导入配置管理模块
try:
    # 添加../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.insert(0, utils_path)
    from zeek_columns import conn_columns, http_columns
    from zeek_columns import conn_numeric_columns, conn_categorical_columns, conn_textual_columns
    from zeek_columns import flowmeter_numeric_columns, flowmeter_categorical_columns, flowmeter_textual_columns
    from zeek_columns import ssl_numeric_columns, ssl_categorical_columns, ssl_textual_columns
    from zeek_columns import dns_numeric_columns, dns_categorical_columns, dns_textual_columns
    from zeek_columns import x509_numeric_columns, x509_categorical_columns, x509_textual_columns
    from zeek_columns import max_x509_cert_chain_len, dtype_dict_in_flow_csv
    from config_manager import read_session_label_id_map, read_text_encoder_config
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.DEBUG)    
except ImportError:
    # 如果导入失败，提供一个默认实现
    def read_session_label_id_map():
        return {'benign': 0, 'background': 1, 'mixed': 2, 'malicious': 3}
    
class FlowNodeBuilder:
    """处理将网络流数据构造成图节点的类，负责加载和预处理流特征"""
    def __init__(self, flow_csv_path, session_label_id_map, max_packet_sequence_length, text_encoder_name, max_text_length, thread_count=1, enabled_views=None):
        self.mtu_normalize = 1500
        self.max_packet_sequence_length = max_packet_sequence_length
        self.text_encoder_name = text_encoder_name
        self.max_text_length = max_text_length
        self.text_tokenizer, self.max_text_length = load_text_tokenizer(
            model_name=self.text_encoder_name,
            max_text_length=self.max_text_length
        )

        self.thread_count = thread_count
        self.enabled_views = enabled_views or {
            "flow_numeric_features": True,
            "flow_categorical_features": True,
            "flow_textual_features": True,
            "packet_len_seq": True,
            "packet_iat_seq": True,
            "domain_probs": True,
            "ssl_numeric_features": True,
            "ssl_categorical_features": True,
            "ssl_textual_features": True,
            "x509_numeric_features": True,
            "x509_categorical_features": True,
            "x509_textual_features": True,
            "dns_numeric_features": True,
            "dns_categorical_features": True,
            "dns_textual_features": True,
        }

        # 读取会话标签映射配置并计算类别数量
        self.session_label_id_map = session_label_id_map
        self.num_classes = len(set(self.session_label_id_map.values()))
        logger.info(f"Loaded session label string-to-id mapping: len={self.num_classes}, mapping={self.session_label_id_map}")

        logger.info("Loading flow CSV file as a pandas dataframe...")
        flow_df = read_large_csv_with_progress(flow_csv_path)

        # 构建 flow uid -> record的字典结构，方便后续从session row基于flow_uid_list的检索归属于session的网络流。
        self.flow_dict = {}
        logger.info("Building flow dictionary using the 'uid' fields as keys...")
        for _, row in tqdm.tqdm(flow_df.iterrows(), total=len(flow_df)):
            flow_uid = row['uid']
            flow_record = row.to_dict()

            # ===== 在这里一次性规范化 flow_record =====
            flow_record['uid'] = flow_uid
            try:
                flow_record['ts'] = float(flow_record.get('conn.ts', 0.0)) # 每个record有时间戳，方便后续建图
            except Exception:
                flow_record['ts'] = 0.0

            self.flow_dict[flow_uid] = MappingProxyType(flow_record) # 把 flow_record 冻结为只读

        self.categorical_vocabulary_group = self.scan_flow_dict_for_categorical_topk_vocab_group(flow_dict = self.flow_dict)
        self.global_node_feature_dims, self.numeric_feature_stats = self.scan_flow_dict_for_node_feature_dims_and_numeric_stats(
            flow_dict = self.flow_dict,
            enabled_views = self.enabled_views,
            max_text_length = self.max_text_length,
            text_tokenizer = self.text_tokenizer,
            max_packet_sequence_length = self.max_packet_sequence_length,
            categorical_vocabulary_group = self.categorical_vocabulary_group,
            num_classes = self.num_classes,
        )
        self.categorical_vocabulary_group = MappingProxyType(self.categorical_vocabulary_group)        
        self.global_node_feature_dims = MappingProxyType(self.global_node_feature_dims)
        self.numeric_feature_stats = MappingProxyType(self.numeric_feature_stats)

        logger.info("✅ Global node feature dimension summary (enabled views):")
        for view_name, dim in self.global_node_feature_dims.items():
            enabled = self.enabled_views.get(view_name, False)
            status = "ON " if enabled else "OFF"
            logger.info(f"  - [{status}] {view_name}: {dim}")

        # 4️⃣ 显式释放 flow_df，节约内存
        del flow_df
        import gc
        gc.collect()
    
    def get_flow_record(self, flow_uid):
        """获取指定UID的流记录"""
        record = self.flow_dict.get(flow_uid)
        if record is None:
            logger.info(f"[DEBUG] Flow UID '{flow_uid}' 在flow_dict中未找到")
        return record
    
    def get_all_flow_uids(self):
        """获取所有流UID"""
        return list(self.flow_dict.keys())
    
    def get_num_classes(self):
        """获取类别数量"""
        return self.num_classes

    @staticmethod
    def scan_flow_dict_for_categorical_topk_vocab_group(flow_dict):
        """
        仅基于 flow_dict 构建 categorical 特征的 vocabulary（高效版，仅扫描一次 flow_dict）。
        flow_dict: { uid -> flow_record(dict) }

        返回:
            vocab_group = {
                col_name: { token -> index }
            }
        """

        top_k_cat = 500  # 可调
        top_k_map = {
            # ---------------- SSL ----------------
            "ssl.cipher": 50,
            "ssl.curve": 10,
            "ssl.version": 6,
            "ssl.next_protocol": 20,
            "ssl.client_signature_algorithms": 50,
            "ssl.server_signature_algorithms": 50,
            "ssl.client_key_exchange_groups": 20,
            "ssl.server_key_exchange_groups": 20,
            "ssl.client_supported_versions": 10,
            "ssl.server_supported_versions": 10,

            # ---------------- DNS ----------------
            "dns.qtype": 40,
            "dns.qclass": 10,
            "dns.rcode_name": 20,
            "dns.qtype_name": 40,
            "dns.qclass_name": 10,
            "dns.rcode": 10,

            # ---------------- conn ----------------
            "conn.proto": 10,
            "conn.service": 50,
            "conn.conn_state": 20,
            "conn.history": 30,
            "conn.local_orig": 3,
            "conn.local_resp": 3,

            # ---------------- flowmeter ----------------
            # Flowmeter categorical（只有 proto）
            "flowmeter.proto": 10,
        }
        
        # 每种类型的列名，加上前缀 → 真实 DataFrame 列名
        conn_cat_cols_prefixed      = [f"conn.{c}"      for c in conn_categorical_columns]
        flowmeter_cat_cols_prefixed = [f"flowmeter.{c}" for c in flowmeter_categorical_columns]
        flow_cat_cols_prefixed = conn_cat_cols_prefixed + flowmeter_cat_cols_prefixed
        ssl_cat_cols_prefixed       = [f"ssl.{c}"       for c in ssl_categorical_columns]
        x509_cat_cols_prefixed = []
        for n in [0, 1, 2]:
            x509_cat_cols_prefixed += [f"x509.cert{n}.{c}" for c in x509_categorical_columns]
        dns_cat_cols_prefixed       = [f"dns.{c}"       for c in dns_categorical_columns]
        categorical_columns = (
            flow_cat_cols_prefixed +
            ssl_cat_cols_prefixed +
            x509_cat_cols_prefixed +
            dns_cat_cols_prefixed
        )

        # Counter 初始化
        categorical_vocab_counter = {col: Counter() for col in categorical_columns}

        # 🔥 只扫描一次 flow_dict（高效）
        for flow_uid, flow_record in tqdm.tqdm(flow_dict.items(), 
                                               desc="[1st PASS] Scanning categorical vocab", 
                                               unit="flow"):
            for col in categorical_columns:
                raw = flow_record.get(col)

                if raw is None:
                    token = "<OOV>"
                else:
                    token = str(raw).strip() or "<OOV>"

                categorical_vocab_counter[col][token] += 1

        # 构建最终 vocab_group
        categorical_vocab_group = {}
        for col in categorical_columns:
            counter = categorical_vocab_counter[col]

            if not counter:
                categorical_vocab_group[col] = {"<OOV>": 0}
                continue

            # top-k
            this_top_k = next((v for k, v in top_k_map.items() if col.startswith(k)), top_k_cat)
            most = counter.most_common(this_top_k)

            values = [v for v, _ in most]
            mapping = {v: i+1 for i, v in enumerate(values)}
            mapping["<OOV>"] = 0

            categorical_vocab_group[col] = mapping

        return categorical_vocab_group
    
    @staticmethod
    def scan_flow_dict_for_node_feature_dims_and_numeric_stats(flow_dict, enabled_views, max_text_length, text_tokenizer, 
                                                               max_packet_sequence_length, categorical_vocabulary_group, num_classes):
        """计算全局的数值型+类别型节点特征维度"""
        global_node_feature_dims = {
            "flow_numeric_features": 0,
            "flow_categorical_features": 0,
            "packet_len_seq": 0,
            "packet_iat_seq": 0,
            "domain_probs": 0,
            "ssl_numeric_features": 0,
            "ssl_categorical_features": 0,
            "x509_numeric_features": 0,
            "x509_categorical_features": 0,
            "dns_numeric_features": 0,
            "dns_categorical_features": 0,
        }

        numeric_feature_stats = {}
        
        for view_name in [
            "flow_numeric_features",
            "ssl_numeric_features",
            "x509_numeric_features",
            "dns_numeric_features",
        ]:
            if enabled_views.get(view_name, False):
                numeric_feature_stats[view_name] = {
                    "count": 0,
                    "sum": None,
                    "sum_of_squares": None,
                }

        logger.info("Calculating global node feature dimensions and numeric features' statistics from flow_dict...")

        for flow_uid, flow_record in tqdm.tqdm(
            flow_dict.items(),
            total=len(flow_dict),
            desc="[2nd PASS] Calc global node feature dims and numeric features' statistics",
            dynamic_ncols=True,
            leave=True,
            mininterval=0.5            
        ):
            try:
                # 提取网络流的各类特征向量
                if enabled_views.get("flow_numeric_features", False) \
                    or enabled_views.get("flow_categorical_features", False) \
                    or enabled_views.get("flow_textual_features", False):
                    flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算网络流的数值型+类别型特征维度
                if enabled_views.get("flow_numeric_features", False):
                    global_node_feature_dims['flow_numeric_features'] = max(
                        global_node_feature_dims['flow_numeric_features'], len(flow_numeric_features) if len(flow_numeric_features) > 0 else 1)
                    
                    vec = np.array(flow_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["flow_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                    
                if enabled_views.get("flow_categorical_features", False):
                    global_node_feature_dims['flow_categorical_features'] = max(
                        global_node_feature_dims['flow_categorical_features'], len(flow_categorical_features) if len(flow_categorical_features) > 0 else 1)
                    
                # 计算数据包时间序列特征向量维度              
                if enabled_views.get("packet_len_seq", False):
                    global_node_feature_dims["packet_len_seq"] = max_packet_sequence_length
                else:
                    global_node_feature_dims["packet_len_seq"] = 0

                if enabled_views.get("packet_iat_seq", False):
                    global_node_feature_dims["packet_iat_seq"] = max_packet_sequence_length
                else:
                    global_node_feature_dims["packet_iat_seq"] = 0
                    
                # 提取基于domain-app共现概率的域名嵌入特征向量
                if enabled_views.get("domain_probs", False):
                    domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
                # 计算基于domain-app共现概率的域名嵌入特征向量维度
                if enabled_views.get("domain_probs", False):
                    global_node_feature_dims['domain_probs'] = max(
                        global_node_feature_dims['domain_probs'], len(domain_probs) if len(domain_probs) > 0 else 1)

                # 提取SSL的各类特征向量
                if enabled_views.get("ssl_numeric_features", False) \
                    or enabled_views.get("ssl_categorical_features", False) \
                    or enabled_views.get("ssl_textual_features", False):
                    ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算SSL的数值型+类别型特征向量的维度
                if enabled_views.get("ssl_numeric_features", False):
                    global_node_feature_dims['ssl_numeric_features'] = max(
                        global_node_feature_dims['ssl_numeric_features'], len(ssl_numeric_features) if len(ssl_numeric_features) > 0 else 1)
                    
                    vec = np.array(ssl_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值
                    
                    stats = numeric_feature_stats["ssl_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("ssl_categorical_features", False):
                    global_node_feature_dims['ssl_categorical_features'] = max(
                        global_node_feature_dims['ssl_categorical_features'], len(ssl_categorical_features) if len(ssl_categorical_features) > 0 else 1)
                    
                # 提取X509的各类特征向量
                if enabled_views.get("x509_numeric_features", False) \
                    or enabled_views.get("x509_categorical_features", False) \
                    or enabled_views.get("x509_textual_features", False):
                    x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算X509的数值型+类别型特征向量的维度
                if enabled_views.get("x509_numeric_features", False):
                    global_node_feature_dims['x509_numeric_features'] = max(
                        global_node_feature_dims['x509_numeric_features'], len(x509_numeric_features) if len(x509_numeric_features) > 0 else 1)
                    
                    vec = np.array(x509_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):                        
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["x509_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("x509_categorical_features", False):
                    global_node_feature_dims['x509_categorical_features'] = max(
                        global_node_feature_dims['x509_categorical_features'], len(x509_categorical_features) if len(x509_categorical_features) > 0 else 1)

                # 提取DNS的各类特征向量
                if enabled_views.get("dns_numeric_features", False) \
                    or enabled_views.get("dns_categorical_features", False) \
                    or enabled_views.get("dns_textual_features", False):
                    dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

                # 计算DNS的数值型+类别型特征向量的维度                
                if enabled_views.get("dns_numeric_features", False):
                    global_node_feature_dims['dns_numeric_features'] = max(
                        global_node_feature_dims['dns_numeric_features'], len(dns_numeric_features) if len(dns_numeric_features) > 0 else 1)
                    
                    vec = np.array(dns_numeric_features, dtype=np.float64)
                    if np.any(np.isnan(vec)) or np.any(np.isinf(vec)):
                        vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0) # 修复NaN和Inf值

                    stats = numeric_feature_stats["dns_numeric_features"]
                    if stats["sum"] is None:
                        stats["sum"] = np.zeros_like(vec)
                        stats["sum_of_squares"] = np.zeros_like(vec)

                    stats["sum"] += vec
                    stats["sum_of_squares"] += vec * vec
                    stats["count"] += 1
                
                if enabled_views.get("dns_categorical_features", False):
                    global_node_feature_dims['dns_categorical_features'] = max(
                        global_node_feature_dims['dns_categorical_features'], len(dns_categorical_features) if len(dns_categorical_features) > 0 else 1)
                    
            except Exception as e:
                logger.error(f"Flow {flow_uid} 特征提取错误: {e}")
                continue
        
        logger.info(f"Global feature dimensions: {global_node_feature_dims}, with max_packet_sequence_length = {max_packet_sequence_length}")
        
        # 添加调试信息：显示Flow、SSL、X509、和DNS特征的实际维度
        logger.info(f"Flow feature dimension breakdown: numeric={global_node_feature_dims['flow_numeric_features']}, categorical={global_node_feature_dims['flow_categorical_features']}")
        logger.info(f"SSL feature dimension breakdown: numeric={global_node_feature_dims['ssl_numeric_features']}, categorical={global_node_feature_dims['ssl_categorical_features']}")
        logger.info(f"X509 feature dimension breakdown: numeric={global_node_feature_dims['x509_numeric_features']}, categorical={global_node_feature_dims['x509_categorical_features']}")
        logger.info(f"DNS feature dimension breakdown: numeric={global_node_feature_dims['dns_numeric_features']}, categorical={global_node_feature_dims['dns_categorical_features']}")

        for k, stats in numeric_feature_stats.items():
            count = stats["count"]
            # ⭐ 核心防护
            if count == 0 or stats["sum"] is None:
                logger.warning(
                    f"scan_flow_dict_for_node_feature_dims_and_numeric_stats(): [NUMERIC-STATS] Skip {k}: count={count}, sum is None"
                )
                stats["mean"] = []
                stats["std"] = []

            else:
                mean = stats["sum"] / count
                var = stats["sum_of_squares"] / count - mean * mean
                std = np.sqrt(np.maximum(var, 1e-12))

                stats["mean"] = mean.tolist()
                stats["std"] = std.tolist()
            
        return global_node_feature_dims, numeric_feature_stats

    def get_global_node_feature_dims(self, key):
        assert hasattr(self, 'global_node_feature_dims'), \
            "global_node_feature_dims must be initialized in __init__"
            
        return self.global_node_feature_dims[key]
    
    def build_node_features(self, flow_uids):
        """为指定的流UID构建节点特征"""
        flow_dict = self.flow_dict
        enabled_views = self.enabled_views
        max_text_length = self.max_text_length
        categorical_vocabulary_group = self.categorical_vocabulary_group
        text_tokenizer = self.text_tokenizer
        numeric_feature_stats = self.numeric_feature_stats
        mtu_normalize = self.mtu_normalize
        max_packet_sequence_length = self.max_packet_sequence_length
        num_classes = self.num_classes

        if not flow_uids:
            return

        logger.debug("build_node_features(): begin")
        def is_nan_or_inf(x):
            """更全面的NaN/Inf检查"""
            if x is None:
                return True
            try:
                # 处理numpy类型
                if hasattr(x, 'dtype'):
                    return np.isnan(x) or np.isinf(x)
                # 处理Python数值类型
                elif isinstance(x, (int, float, np.number)):
                    return np.isnan(x) or np.isinf(x)
                return False
            except (TypeError, ValueError):
                return False
            
        nodes = []
        # 提取每个节点的特征
        for flow_uid in flow_uids:
            flow_record = flow_dict.get(flow_uid)
            if flow_record is None:
                continue

            node = {
                'uid': flow_record['uid'],
                'ts': flow_record['ts'],
            }
            nodes.append(node)
        
        if not nodes or len(nodes) == 0:
            return []
        
        logger.debug("build_node_features(): nodes list construction is ok")

        # 构建节点特征
        for n in nodes:
            flow_record = flow_dict.get(n['uid'])
            if flow_record is None:
                continue

            if (enabled_views.get("flow_numeric_features", False)
                or enabled_views.get("flow_categorical_features", False)
                or enabled_views.get("flow_textual_features", False)
            ):
                flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("flow_numeric_features", False):
                n['flow_numeric_features'] = flow_numeric_features                   
                if numeric_feature_stats["flow_numeric_features"]["count"] > 1:
                    vec = n['flow_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["flow_numeric_features"]["mean"]
                    std  = numeric_feature_stats["flow_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"Flow numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"Flow numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['flow_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_flow_numeric_len = self.get_global_node_feature_dims('flow_numeric_features')
                # 填充或裁剪 flow_numeric_features
                if len(n['flow_numeric_features']) < max_flow_numeric_len:
                    n['flow_numeric_features'] += [0.0] * (max_flow_numeric_len - len(n['flow_numeric_features']))
                else:
                    n['flow_numeric_features'] = n['flow_numeric_features'][:max_flow_numeric_len]

            if enabled_views.get("flow_categorical_features", False):
                n['flow_categorical_features'] = flow_categorical_features
                max_flow_categorical_len = self.get_global_node_feature_dims('flow_categorical_features')
                # 填充或裁剪 flow_categorical_features
                if len(n['flow_categorical_features']) < max_flow_categorical_len:
                    n['flow_categorical_features'] += [0] * (max_flow_categorical_len - len(n['flow_categorical_features']))
                else:
                    n['flow_categorical_features'] = n['flow_categorical_features'][:max_flow_categorical_len]

            if enabled_views.get("flow_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(flow_textual_features, dict)
                assert flow_textual_features["input_ids"].dim() == 2                
                n['flow_textual_features'] = flow_textual_features

            if (enabled_views.get("packet_len_seq", False)
                or enabled_views.get("packet_iat_seq", False)
            ):
                packet_len_seq, packet_iat_seq, packet_seq_mask = extract_flowmeter_packet_level_features(flow_record, mtu_normalize, max_packet_sequence_length)

            if enabled_views.get("packet_len_seq", False):
                n['packet_len_seq'] = packet_len_seq

            if enabled_views.get("packet_iat_seq", False):
                n['packet_iat_seq'] = packet_iat_seq

            n['packet_seq_mask'] = packet_seq_mask

            if enabled_views.get("domain_probs", False):
                domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
                n['domain_probs'] = domain_probs
                max_domain_prob_len = self.get_global_node_feature_dims('domain_probs')
                # 填充或裁剪 domain_probs
                if len(n['domain_probs']) < max_domain_prob_len:
                    n['domain_probs'] += [0.0] * (max_domain_prob_len - len(n['domain_probs']))
                else:
                    n['domain_probs'] = n['domain_probs'][:max_domain_prob_len]

            if (enabled_views.get("ssl_numeric_features", False)
                or enabled_views.get("ssl_categorical_features", False)
                or enabled_views.get("ssl_textual_features", False)
            ):
                ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("ssl_numeric_features", False):
                n['ssl_numeric_features'] = ssl_numeric_features
                if numeric_feature_stats["ssl_numeric_features"]["count"] > 1:
                    vec = n['ssl_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["ssl_numeric_features"]["mean"]
                    std  = numeric_feature_stats["ssl_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"SSL numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"SSL numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['ssl_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_ssl_numeric_len = self.get_global_node_feature_dims('ssl_numeric_features')
                # 填充或裁剪 ssl_numeric_features
                if len(n['ssl_numeric_features']) < max_ssl_numeric_len:
                    n['ssl_numeric_features'] += [0.0] * (max_ssl_numeric_len - len(n['ssl_numeric_features']))
                else:
                    n['ssl_numeric_features'] = n['ssl_numeric_features'][:max_ssl_numeric_len]

            if enabled_views.get("ssl_categorical_features", False):
                n['ssl_categorical_features'] = ssl_categorical_features
                max_ssl_categorical_len = self.get_global_node_feature_dims('ssl_categorical_features')
                if len(n['ssl_categorical_features']) < max_ssl_categorical_len:
                    n['ssl_categorical_features'] += [0] * (max_ssl_categorical_len - len(n['ssl_categorical_features']))
                else:
                    n['ssl_categorical_features'] = n['ssl_categorical_features'][:max_ssl_categorical_len]

            if enabled_views.get("ssl_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(ssl_textual_features, dict)
                assert ssl_textual_features["input_ids"].dim() == 2                
                n['ssl_textual_features'] = ssl_textual_features

            if (enabled_views.get("x509_numeric_features", False)
                or enabled_views.get("x509_categorical_features", False)
                or enabled_views.get("x509_textual_features", False)
            ):
                x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("x509_numeric_features", False):
                n['x509_numeric_features'] = x509_numeric_features
                if numeric_feature_stats["x509_numeric_features"]["count"] > 1:
                    vec = n['x509_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["x509_numeric_features"]["mean"]
                    std  = numeric_feature_stats["x509_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"X509 numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"X509 numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['x509_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_x509_numeric_len = self.get_global_node_feature_dims('x509_numeric_features')
                # 填充或裁剪 x509_features
                if len(n['x509_numeric_features']) < max_x509_numeric_len:
                    n['x509_numeric_features'] += [0.0] * (max_x509_numeric_len - len(n['x509_numeric_features']))
                else:
                    n['x509_numeric_features'] = n['x509_numeric_features'][:max_x509_numeric_len]
                
            if enabled_views.get("x509_categorical_features", False):
                n['x509_categorical_features'] = x509_categorical_features
                max_x509_categorical_len = self.get_global_node_feature_dims('x509_categorical_features')
                if len(n['x509_categorical_features']) < max_x509_categorical_len:
                    n['x509_categorical_features'] += [0] * (max_x509_categorical_len - len(n['x509_categorical_features']))
                else:
                    n['x509_categorical_features'] = n['x509_categorical_features'][:max_x509_categorical_len]

            if enabled_views.get("x509_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(x509_textual_features, dict)
                assert x509_textual_features["input_ids"].dim() == 2
                n['x509_textual_features'] = x509_textual_features

            if (
                enabled_views.get("dns_numeric_features", False)
                or enabled_views.get("dns_categorical_features", False)
                or enabled_views.get("dns_textual_features", False)
            ):
                dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)

            if enabled_views.get("dns_numeric_features", False):
                n['dns_numeric_features'] = dns_numeric_features
                if numeric_feature_stats["dns_numeric_features"]["count"] > 1:
                    vec = n['dns_numeric_features']
                    # 标准化前检查NaN
                    if any(is_nan_or_inf(x) for x in vec):
                        # logger.warning(f"Flow {flow_uid} 标准化前发现NaN值")
                        vec = [0.0 if is_nan_or_inf(x) else x for x in vec]                    
                    mean = numeric_feature_stats["dns_numeric_features"]["mean"]
                    std  = numeric_feature_stats["dns_numeric_features"]["std"]
                    assert len(mean) == len(vec), f"DNS numeric feature length inconsistent: mean={len(mean)}, vec={len(vec)}, uid={flow_uid}"
                    assert len(std) == len(vec), f"DNS numeric feature length inconsistent: mean={len(std)}, vec={len(vec)}, uid={flow_uid}"
                    n['dns_numeric_features'] = [(x - m) / s for x, m, s in zip(vec, mean, std)]

                max_dns_numeric_len = self.get_global_node_feature_dims('dns_numeric_features')
                # 填充或裁剪 dns_features
                if len(n['dns_numeric_features']) < max_dns_numeric_len:
                    n['dns_numeric_features'] += [0.0] * (max_dns_numeric_len - len(n['dns_numeric_features']))
                else:
                    n['dns_numeric_features'] = n['dns_numeric_features'][:max_dns_numeric_len]
                
            if enabled_views.get("dns_categorical_features", False):
                n['dns_categorical_features'] = dns_categorical_features
                max_dns_categorical_len = self.get_global_node_feature_dims('dns_categorical_features')
                if len(n['dns_categorical_features']) < max_dns_categorical_len:
                    n['dns_categorical_features'] += [0] * (max_dns_categorical_len - len(n['dns_categorical_features']))
                else:
                    n['dns_categorical_features'] = n['dns_categorical_features'][:max_dns_categorical_len]

            if enabled_views.get("dns_textual_features", False):
                # ✅ 直接保存 dict，不做任何长度处理
                assert isinstance(dns_textual_features, dict)
                assert dns_textual_features["input_ids"].dim() == 2
                n['dns_textual_features'] = dns_textual_features

        logger.debug("build_node_features() ends: node feature extraction is ok, and max feature lengths are determined")

        return nodes


def parse_list_field(field_value):
    """终极修正版列表解析函数"""
    if field_value is None or pd.isna(field_value):
        return []
    
    if isinstance(field_value, (list, np.ndarray)):
        return list(field_value)
    
    if isinstance(field_value, str):
        value = field_value.strip()
        if not value or value.lower() in ['nan', 'none', 'null', '[]', '{}']:
            return []
        
        # 尝试自动修复不完整括号
        if value.count('[') != value.count(']'):
            # 情况1：缺少闭合括号
            if value.startswith('[') and not value.endswith(']'):
                value += ']'  # 尝试自动补全
            # 情况2：多余闭合括号
            elif not value.startswith('[') and value.endswith(']'):
                value = '[' + value
            # 其他情况保持原样
        
        # 解析优先级：JSON > Python字面量 > 逗号分隔
        for parser in [json.loads, ast.literal_eval]:
            try:
                parsed = parser(value)
                if isinstance(parsed, (list, tuple)):
                    return [int(x) if isinstance(x, float) and x.is_integer() else x 
                           for x in parsed]
                return [parsed]
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
        
        # 处理纯逗号分隔字符串（无括号）
        if ',' in value:
            parts = []
            for part in value.split(','):
                part = part.strip()
                if not part:
                    continue
                try:
                    num = float(part)
                    parts.append(int(num) if num.is_integer() else num)
                except ValueError:
                    parts.append(part)
            return parts
        
        return [value]
    
    return [field_value]


def normalize_packet_direction(d):
    # 返回 1 表示 客户端->服务端，返回 -1 表示 服务器->客户端
    if isinstance(d, (int, float, np.integer, np.floating)):
        try:
            v = int(d)
            return 1 if v == 1 else -1
        except:
            return 1
    if isinstance(d, str):
        ds = d.strip().lower()
        if ds in ('1', 'true', 't', 'c2s', 'client', '->', '>'):
            return 1
        if ds in ('0','false','f','s2c','server','<-','<'):
            return -1
        # 尝试识别 -1
        if ds.startswith('-'):
            return -1
        return 1
    if isinstance(d, bool):
        return 1 if d else -1
    return 1


def extract_flowmeter_packet_level_features(
    flow_record,
    mtu_normalize=1500,
    max_packet_sequence_length: int | None = None,
    pad_value: float = 0.0,
) -> Tuple[List[float], List[float], List[int]]:
    """
    提取 packet-level 特征：
      ✔ 方向增强 + MTU归一化 payload
      ✔ 方向增强 + 分段 log 缩放 IAT
      ❗ 序列长度不一致 → 直接抛异常，阻断图构建
    """
    packet_dir_vector = parse_list_field(
        flow_record.get('flowmeter.packet_direction_vector', [])
    )
    packet_len_vector = parse_list_field(
        flow_record.get('flowmeter.packet_payload_size_vector', [])
    )
    raw_packet_iat_vector = parse_list_field(
        flow_record.get('flowmeter.packet_iat_vector', [])
    )
    packet_iat_vector = [0.0] + raw_packet_iat_vector  # ⚠ IAT 前补一个0

    # -------- 直接强校验长度 --------
    if not (
        len(packet_dir_vector) ==
        len(packet_len_vector) ==
        len(packet_iat_vector)
    ):
        raise ValueError(
            f"[SeqLenError] packet-level特征长度不一致:"
            f" packet_dir_vector={len(packet_dir_vector)},"
            f" packet_len_vector={len(packet_len_vector)},"
            f" packet_iat_vector={len(packet_iat_vector)},"
            f" uid={flow_record.get('uid')}"
        )

    # 后续逻辑都在保证长度一致前提下执行
    dir_vec_len = len(packet_dir_vector)
    if dir_vec_len == 0:
        if max_packet_sequence_length is None:
            return [], [], []
        else:
            return (
                [pad_value] * max_packet_sequence_length,
                [pad_value] * max_packet_sequence_length,
                [0] * max_packet_sequence_length,
            )

    # -------- IAT 分段时间缩放函数 --------
    def _safe_log_scale_time(time_ms):
        if time_ms == 0: return 0.0
        sign = 1 if time_ms > 0 else -1
        abs_time = abs(time_ms)

        if abs_time < 10:
            return sign * abs_time / 1000.0
        elif abs_time < 1000:
            return sign * (0.01 + np.log1p(abs_time) / 10.0)
        elif abs_time < 60000:
            return sign * (0.1 + np.log1p(abs_time / 1000.0) / 5.0)
        elif abs_time < 3600000:
            return sign * (0.5 + np.log1p(abs_time / 60000.0) / 3.0)
        else:
            return sign * (1.0 + np.log1p(abs_time / 3600000.0) / 2.0)

    packet_len_seq = []
    packet_iat_seq = []

    for dir_vec, len_vec, iat_vec in zip(packet_dir_vector, packet_len_vector, packet_iat_vector):
        sign_vec = normalize_packet_direction(dir_vec)

        # Payload 归一化
        norm_payload = float(len_vec) / float(mtu_normalize)
        norm_payload = max(-1.0, min(1.0, norm_payload))  # clip
        packet_len_seq.append(sign_vec * norm_payload)

        # IAT 缩放
        scaled_iat_seq = _safe_log_scale_time(float(iat_vec))
        packet_iat_seq.append(sign_vec * scaled_iat_seq)

    # ===== truncate + pad（如果配置了 max_packet_sequence_length） =====
    if max_packet_sequence_length is not None:
        orig_len = len(packet_len_seq)

        # truncate
        packet_len_seq = packet_len_seq[:max_packet_sequence_length]
        packet_iat_seq = packet_iat_seq[:max_packet_sequence_length]

        # pad
        if orig_len < max_packet_sequence_length:
            pad_len = max_packet_sequence_length - orig_len
            packet_len_seq.extend([pad_value] * pad_len)
            packet_iat_seq.extend([pad_value] * pad_len)

    # ✅ 构造 mask：真实位置为 1，padding 为 0
    if max_packet_sequence_length is None:
        # 不截断、不 padding
        packet_seq_mask = [1] * len(packet_len_seq)
    else:
        valid_len = min(orig_len, max_packet_sequence_length)
        packet_seq_mask = [1] * valid_len
        if orig_len < max_packet_sequence_length:
            packet_seq_mask.extend([0] * (max_packet_sequence_length - orig_len))

    return packet_len_seq, packet_iat_seq, packet_seq_mask


def extract_domain_name_probabilities(flow_record, num_classes, num_domain_name_hierarchy_levels = 5):
    """从DNS和TLS域名嵌入特征中提取多层级嵌入向量，严格校验维度"""
    domain_probs = []    
    for level in range(num_domain_name_hierarchy_levels): # 默认层级数量：0~4
        for proto in ['ssl', 'dns']:
            if proto == 'ssl':
                embed_col = f'{proto}.server_name{level}_freq'
            elif proto == 'dns':
                embed_col = f'{proto}.query{level}_freq'
            else:
                raise ValueError(f"extract_domain_name_probabilities(): unsupported protocol or domain name hierarchical level.")
            
            embed_value = flow_record.get(embed_col, None)

            # 🔹如果列不存在或值为空 → 填充全零
            if embed_value is None or pd.isna(embed_value) or embed_value == "":
                domain_probs.extend([0.0] * num_classes)
                continue

            try:
                # 🔹确保转换为 Python list 或 numpy array
                if isinstance(embed_value, str):
                    embed_vector = ast.literal_eval(embed_value)
                elif isinstance(embed_value, (list, np.ndarray)):
                    embed_vector = list(embed_value)
                else:
                    raise TypeError(f"Unsupported type for {embed_col}: {type(embed_value)}")

                # 🔹确保是可迭代的浮点向量
                embed_vector = [float(x) for x in embed_vector]

                # 严格维度校验
                if len(embed_vector) != num_classes:
                    raise ValueError(
                        f"extract_domain_name_probabilities(): [DimError] {embed_col} 维度错误: "
                        f"expected={num_classes}, got={len(embed_vector)}, value={embed_vector}"
                    )

                domain_probs.extend(embed_vector)

            except Exception as e:
                # ❌任何解析失败 → 抛出明确异常，用于定位数据问题
                raise ValueError(
                    f"[ParseError] 域名嵌入解析失败: {embed_col}={embed_value}, error={e}"
                )

    return domain_probs


def to_str_safe(val):
    """把 val 安全地变为 str 并 strip；对于 None / NaN 返回空字符串。"""
    if val is None:
        return ''
    try:
        # pandas 的 NaN / None 检测
        if pd.isna(val):
            return ''
    except Exception:
        pass
    if isinstance(val, str):
        return val.strip()
    # 其他类型（float/int/list/..）都转成字符串并 strip
    return str(val).strip()

def encode_text(text: str, text_tokenizer, max_text_length):
    """
    将任意 textual 字段编码为长度固定 max_text_length 的 token 序列（LongTensor）
    """
    if not isinstance(text, str):
        text = ""  # 非字符串统一处理为空字符串

    encoded = text_tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=max_text_length,
        return_tensors="pt",
    )

    # ✅ Sanity check（强烈推荐）
    assert encoded["input_ids"].dim() == 2, \
        f"encode_text expects input_ids to be 2D [1, L], got shape {encoded['input_ids'].shape}"
    assert encoded["attention_mask"].dim() == 2, \
        f"encode_text expects attention_mask to be 2D [1, L], got shape {encoded['attention_mask'].shape}"

    return {
        "input_ids": encoded["input_ids"],
        "attention_mask": encoded["attention_mask"],
    }

def extract_conn_and_flowmeter_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    """从flowmeter记录中提取统计特征"""
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- 数值特征 ----------
    for col in conn_numeric_columns:
        full = f"conn.{col}"
        value = flow_record.get(full, None)
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in conn_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    for col in flowmeter_numeric_columns:
        full = f"flowmeter.{col}"
        value = flow_record.get(full, None)
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in flowmeter_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- 类别特征 ----------
    for col in conn_categorical_columns:
        full = f"conn.{col}"
        value = flow_record.get(full, "")
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))        

    for col in flowmeter_categorical_columns:
        full = f"flowmeter.{col}"
        value = flow_record.get(full, "")
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- 文本特征 ----------
    for col in conn_textual_columns:
        full = f"conn.{col}"
        value = flow_record.get(full, "")
        textual_fields.append(to_str_safe(value))

    for col in flowmeter_textual_columns:
        full = f"flowmeter.{col}"
        value = flow_record.get(full, "")
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text

def extract_ssl_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    """提取 SSL 的 numeric / categorical / textual 特征（严格使用 zeek_columns 定义）"""
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- 数值特征 ----------
    for col in ssl_numeric_columns:
        full = f"ssl.{col}"
        value = flow_record.get(full, None)
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in ssl_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- 类别特征 ----------
    for col in ssl_categorical_columns:
        full = f"ssl.{col}"
        value = flow_record.get(full, "")
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- 文本特征 ----------
    for col in ssl_textual_columns:
        full = f"ssl.{col}"
        value = flow_record.get(full, "")
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text

def extract_x509_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    numeric = []
    categorical = []
    textual_fields = []

    for idx in range(max_x509_cert_chain_len):
        prefix = f"x509.cert{idx}"

        # 如果该证书不存在 → 填零占位（保持对齐）
        exists = any(k.startswith(prefix) for k in flow_record.keys())

        # ---------- numeric ----------
        for col in x509_numeric_columns:
            full = f"{prefix}.{col}"
            value = flow_record.get(full, None)
            if not exists:
                numeric.append(0.0)
                continue
            try:
                numeric.append(float(value) if value not in (None, "") else 0.0)
            except:
                numeric.append(0.0)

        # ---------- categorical ----------
        for col in x509_categorical_columns:
            full = f"{prefix}.{col}"
            value = flow_record.get(full, "")
            vocab = categorical_vocab_group.get(full)
            if vocab is None:
                categorical.append(0)
            else:
                categorical.append(vocab.get(value, 0))

        # ---------- textual ----------
        for col in x509_textual_columns:
            full = f"{prefix}.{col}"
            value = flow_record.get(full, "")
            textual_fields.append(to_str_safe(value) if exists else "")

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text


def extract_dns_features(flow_record, categorical_vocab_group, text_tokenizer, max_text_length):
    numeric = []
    categorical = []
    textual_fields = []

    # ---------- numeric ----------
    for col in dns_numeric_columns:
        full = f"dns.{col}"
        value = flow_record.get(full, None)
        try:
            numeric.append(float(value) if value not in (None, "") else 0.0)
        except:
            logger.error(
                f"[DEBUG] BAD NUMERIC in dns_numeric_columns: "
                f"col={col}, value='{value}'"
            )            
            numeric.append(0.0)

    # ---------- categorical ----------
    for col in dns_categorical_columns:
        full = f"dns.{col}"
        value = flow_record.get(full, "")
        vocab = categorical_vocab_group.get(full)
        if vocab is None:
            categorical.append(0)
        else:
            categorical.append(vocab.get(value, 0))

    # ---------- textual ----------
    for col in dns_textual_columns:
        full = f"dns.{col}"
        value = flow_record.get(full, "")
        textual_fields.append(to_str_safe(value))

    # 合并成一个字符串，可选，也可分多字段编码
    combined_text = " ".join([str(x) for x in textual_fields if isinstance(x, str)])
    encoded_text = encode_text(combined_text, text_tokenizer, max_text_length)

    return numeric, categorical, encoded_text


def get_project_root(start_path: str = None):
    import os, subprocess

    if start_path is None:
        start_path = os.path.abspath(os.path.dirname(__file__))

    # ① 尝试通过 Git
    try:
        root = subprocess.check_output(
            ["git", "rev-parse", "--show-toplevel"],
            stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
        return root
    except Exception:
        pass

    # ② 尝试查找关键文件
    markers = ("pyproject.toml", "setup.py", "requirements.txt", ".git")
    cur = start_path
    while True:
        if any(os.path.exists(os.path.join(cur, m)) for m in markers):
            return cur
        parent = os.path.abspath(os.path.join(cur, os.pardir))
        if parent == cur:
            break
        cur = parent

    # ③ fallback：使用 VSCode 工作路径
    return os.environ.get("PWD", os.getcwd())

def load_text_tokenizer(model_name="bert-base-uncased", max_text_length=64):
    """
    加载 BERT tokenizer。
    支持：
      ✔ 先尝试从本地 models_hub 求解
      ✔ 找不到则自动在线加载
    返回：
      tokenizer（BertTokenizer）
      max_text_len（int）
    """
    current_file_dir = os.path.dirname(os.path.abspath(__file__))

    # 2. 硬编码回退两层找到项目根目录
    # 结构: [ROOT]/src/build_session_graph/flow_node_builder.py
    # 回退: ../../
    project_root = os.path.abspath(os.path.join(current_file_dir, "..", ".."))

    model_path = os.path.join(project_root, 'models_hub', model_name)

    try:
        logger.info(f"尝试从本地缓存加载 BERT tokenizer: {model_path}")
        tokenizer = BertTokenizer.from_pretrained(model_path)
    except Exception as e:
        logger.warning(f"本地 tokenizer 不存在: {e}")
        logger.warning("尝试从 HuggingFace 在线下载...")
        tokenizer = BertTokenizer.from_pretrained(model_name)

    logger.info(f"BERT tokenizer 加载成功，max_text_len={max_text_length}")
    return tokenizer, max_text_length

def read_large_csv_with_progress(filepath, description="读取数据到pandas dataframe", verbose=True):
    """带进度条的大型CSV文件读取函数"""
    if verbose:
        logger.info(f"{description}，从路径 {filepath}...")
        file_size = os.path.getsize(filepath) / (1024 * 1024 * 1024)  # GB
        logger.info(f"文件大小: {file_size:.2f}GB")
    
    # 先读取前几行获取列信息
    sample_df = pd.read_csv(filepath, nrows=5)
    columns = sample_df.columns.tolist()
    
    # 分块读取
    chunks = []
    chunk_size = 100000  # 每次读取10万行

    if verbose:
        logger.info(f"检测到 {len(columns)} 列，开始每{chunk_size}行分块读取...")
    
    # 获取总行数（不读取全部内容）
    with open(filepath, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # 减去标题行
    
    if verbose:
        # 使用position=0确保进度条在同一行更新
        pbar = tqdm.tqdm(total=total_rows, desc=description, position=0, leave=True)
    
    for chunk in pd.read_csv(filepath, chunksize=chunk_size, low_memory=False):
        chunks.append(chunk)
        if verbose:
            pbar.update(len(chunk))
    
    if verbose:
        pbar.close()
    
    # 合并所有块
    df = pd.concat(chunks, ignore_index=True)
    
    if verbose:
        logger.info(f"{description}完成! 数据形状: {df.shape}")
    
    return df

def main():
    """测试函数：验证flow和session数据读取及特征提取，并计算全局特征维度"""
    # 测试数据路径
    flow_csv_path = "processed_data/CIC-AndMal2017/SMSMalware/jifake-flow.csv"
    session_csv_path = "processed_data/CIC-AndMal2017/SMSMalware/jifake-session.csv"
    
    try:
        # 1. 读取flow数据并构建flow_dict（uid到flow记录的映射）
        logger.info(f"读取flow数据: {flow_csv_path}")
        flow_df = pd.read_csv(
            flow_csv_path,
            dtype=dtype_dict_in_flow_csv,
            parse_dates=False  # 避免自动解析日期导致格式问题
        )
        flow_dict = {row['uid']: row.to_dict() for _, row in flow_df.iterrows()}
        logger.info(f"成功加载 {len(flow_dict)} 条flow记录")

        # 2. 读取session数据
        logger.info(f"读取session数据: {session_csv_path}")
        session_df = pd.read_csv(session_csv_path)
        logger.info(f"成功加载 {len(session_df)} 条session记录")

        # 3. 初始化全局特征维度统计
        global_node_feature_dims = {
            "flow_numeric_features": 0,
            "flow_categorical_features": 0,
            "flow_textual_features": 0,
            "packet_len_seq": 0,
            "packet_iat_seq": 0,
            "domain_probs": 0,
            "ssl_numeric_features": 0,
            "ssl_categorical_features": 0,
            "ssl_textual_features": 0,
            "x509_numeric_features": 0,
            "x509_categorical_features": 0,
            "x509_textual_features": 0,
            "dns_numeric_features": 0,
            "dns_categorical_features": 0,
            "dns_textual_features": 0,
        }

        # 获取默认的类别数量（用于测试）
        num_classes = len(set(read_session_label_id_map().values()))
        logger.info(f"类别数量: {num_classes}")

        categorical_vocabulary_group = FlowNodeBuilder.scan_flow_dict_for_categorical_topk_vocab_group(flow_dict)

        text_encoder_name, max_text_length = read_text_encoder_config()
        text_tokenizer, max_text_length = load_text_tokenizer(
            model_name=text_encoder_name,
            max_text_length=max_text_length
        )
            
        # 4. 遍历所有会话和流，计算全局特征维度
        logger.info("计算全局特征维度...")
        for _, session_row in tqdm.tqdm(session_df.iterrows(), total=len(session_df), desc="处理会话"):
            # 解析session中的flow列表
            if 'flow_uid_list' not in session_row:
                continue

            flow_uid_list = ast.literal_eval(session_row['flow_uid_list'])

            # 遍历每个flow
            for flow_uid in flow_uid_list:
                if flow_uid not in flow_dict:
                    continue

                flow_record = flow_dict[flow_uid]

                try:
                    # 提取加密流量基本特征
                    flow_numeric_features, flow_categorical_features, flow_textual_features = extract_conn_and_flowmeter_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
                    mtu_normalize = 1500
                    max_packet_sequence_length = 512
                    packet_len_seq, packet_iat_seq, packet_seq_mask = extract_flowmeter_packet_level_features(flow_record, mtu_normalize, max_packet_sequence_length)
                    logger.debug("!!! flow_uid = {flow_uid}, with flow_numeric_features = " + str(flow_numeric_features)
                                 + ", flow_categorical_features = " + str(flow_categorical_features)
                                 + ", flow_textual_features = " + str(flow_textual_features)
                                 + ", packet_len_seq = " + str(packet_len_seq)
                                 + ", packet_iat_seq = " + str(packet_iat_seq)
                                 + ", packet_seq_mask = " + str(packet_seq_mask)
                            )

                    # 提取明文部分的载荷特征
                    domain_probs = extract_domain_name_probabilities(flow_record, num_classes)
                    ssl_numeric_features, ssl_categorical_features, ssl_textual_features = extract_ssl_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
                    x509_numeric_features, x509_categorical_features, x509_textual_features = extract_x509_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
                    dns_numeric_features, dns_categorical_features, dns_textual_features = extract_dns_features(flow_record, categorical_vocabulary_group, text_tokenizer, max_text_length)
                    logger.debug("!!! flow_uid = {flow_uid}, with domain_probs = " + str(domain_probs) 
                                 + ", ssl_numeric_features = " + str(ssl_numeric_features) 
                                 + ", ssl_categorical_features = " + str(ssl_categorical_features)
                                 + ", ssl_textual_features = " + str(ssl_textual_features)
                                 + ", x509_numeric_features = " + str(x509_numeric_features) 
                                 + ", x509_categorical_features = " + str(x509_categorical_features)
                                 + ", x509_textual_features = " + str(x509_textual_features)
                                 + ", dns_numeric_features = " + str(dns_numeric_features) 
                                 + ", dns_categorical_features = " + str(dns_categorical_features)
                                 + ", dns_textual_features = " + str(dns_textual_features)
                            )

                    # 更新全局维度统计
                    global_node_feature_dims['flow_numeric_features'] = max(
                        global_node_feature_dims['flow_numeric_features'], 
                        len(flow_numeric_features) if len(flow_numeric_features) > 0 else 1
                    )
                    global_node_feature_dims['flow_categorical_features'] = max(
                        global_node_feature_dims['flow_categorical_features'], 
                        len(flow_categorical_features) if len(flow_categorical_features) > 0 else 1
                    )
                    global_node_feature_dims['flow_textual_features'] = max(
                        global_node_feature_dims['flow_textual_features'], 
                        len(flow_textual_features) if len(flow_textual_features) > 0 else 1
                    )                    
                    global_node_feature_dims['packet_len_seq'] = max_packet_sequence_length
                    global_node_feature_dims['packet_iat_seq'] = max_packet_sequence_length
                    global_node_feature_dims['domain_probs'] = max(
                        global_node_feature_dims.get('domain_probs', 0), 
                        len(domain_probs) if len(domain_probs) > 0 else 1
                    )

                    # 计算SSL的各类特征向量的维度
                    global_node_feature_dims['ssl_numeric_features'] = max(
                        global_node_feature_dims['ssl_numeric_features'], 
                        len(ssl_numeric_features) if len(ssl_numeric_features) > 0 else 1)
                    global_node_feature_dims['ssl_categorical_features'] = max(
                        global_node_feature_dims['ssl_categorical_features'], 
                        len(ssl_categorical_features) if len(ssl_categorical_features) > 0 else 1)
                    global_node_feature_dims['ssl_textual_features'] = max(
                        global_node_feature_dims['ssl_textual_features'], 
                        len(ssl_textual_features) if len(ssl_textual_features) > 0 else 1)

                    # 计算X509的各类特征向量的维度
                    global_node_feature_dims['x509_numeric_features'] = max(
                        global_node_feature_dims['x509_numeric_features'], 
                        len(x509_numeric_features) if len(x509_numeric_features) > 0 else 1)
                    global_node_feature_dims['x509_categorical_features'] = max(
                        global_node_feature_dims['x509_categorical_features'], 
                        len(x509_categorical_features) if len(x509_categorical_features) > 0 else 1)
                    global_node_feature_dims['x509_textual_features'] = max(
                        global_node_feature_dims['x509_textual_features'], 
                        len(x509_textual_features) if len(x509_textual_features) > 0 else 1)

                    # 计算DNS的各类特征向量的维度                
                    global_node_feature_dims['dns_numeric_features'] = max(
                        global_node_feature_dims['dns_numeric_features'], 
                        len(dns_numeric_features) if len(dns_numeric_features) > 0 else 1)
                    global_node_feature_dims['dns_categorical_features'] = max(
                        global_node_feature_dims['dns_categorical_features'], 
                        len(dns_categorical_features) if len(dns_categorical_features) > 0 else 1)
                    global_node_feature_dims['dns_textual_features'] = max(
                        global_node_feature_dims['dns_textual_features'], 
                        len(dns_textual_features) if len(dns_textual_features) > 0 else 1)

                except Exception as e:
                    logger.error(f"Flow {flow_uid} 特征提取错误: {str(e)}")
                    continue

        # 5. 确保最小维度为1
        for key in global_node_feature_dims:
            global_node_feature_dims[key] = max(1, global_node_feature_dims[key])

        # 6. 输出全局特征维度
        logger.info("全局特征维度统计:")
        for key, dim in global_node_feature_dims.items():
            logger.info(f"  {key}: {dim}")

        logger.info("测试完成")

    except FileNotFoundError as e:
        logger.error(f"错误: 未找到测试文件 - {str(e)}")
    except KeyError as e:
        logger.error(f"错误: 数据中缺少必要字段 - {str(e)}")
    except Exception as e:
        logger.error(f"测试过程出错: {str(e)}")


if __name__ == "__main__":
    main()
    