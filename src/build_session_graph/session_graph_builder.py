# -*- coding: utf-8 -*-
import os
import pandas as pd
import dgl
import torch
import numpy as np
from dgl.data.utils import save_graphs, save_info
import tqdm
import ast
import re
from typing import List
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from session_parser import SessionParser
import ipaddress
import logging
import sys
import threading
import copy
import pickle
import math
import gc

# 添加../utils目录到Python搜索路径
utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
sys.path.insert(0, utils_path)
# 导入配置管理模块
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.DEBUG)    

verbose = False
class SessionGraphBuilder():
    def __init__(self, flow_node_builder, session_csv, dumpFilename, 
                concurrent_flow_iat_threshold=1.0, sequential_flow_iat_threshold=10.0,
                session_label_id_map=None, thread_count=1):
        self.flow_node_builder = flow_node_builder
        self.session_parser = SessionParser(flow_node_builder=flow_node_builder, 
                                            session_label_id_map=session_label_id_map)        
        self.concurrent_flow_iat_threshold = concurrent_flow_iat_threshold
        self.sequential_flow_iat_threshold = sequential_flow_iat_threshold
        
        if verbose:
            logger.debug(" Flow数据中的前5个UID:")
            logger.info(str(list(flow_node_builder.flow_dict.keys())[:5]))  # 检查实际存储的UID格式

        # 配置参数
        # self.gc_interval = 1000  # 保存垃圾回收间隔参数
        self.thread_count = thread_count
        self._graph_list_lock = threading.Lock()  # 保护共享变量的线程锁
        self.dumpFilename = dumpFilename

        # 数据存储
        self.graph_list = []
        self.graph_node_uids_list = []  # 新增：存储每个图的节点uid列表（与graphs一一对应）
        self.label_name_list = []
        self.label_id_list = []
        self.is_malicious_list = []
        self.stats = {
            "total_flows": 0,
            "total_bursts": 0,
            "total_edges": 0,
            "total_graphs": 0,
            "processed_session_count": 0  # 已处理的session计数
        }

        """执行完整的图构建流程"""
        # 1. 加载会话数据
        logger.info("Loading session CSV...")
        self.session_df = pd.read_csv(session_csv, low_memory=False)

        # 添加调试
        logger.debug(f" Session DataFrame形状: {self.session_df.shape}")
        logger.debug(f" Session列名: {self.session_df.columns.tolist()}")
        
        if verbose and len(self.session_df) > 0:
            sample_session = self.session_df.iloc[0]
            logger.debug(f" 示例Session数据:")
            logger.debug(f"  session_index: {sample_session.get('session_index')}")
            logger.debug(f"  flow_uid_list: {sample_session.get('flow_uid_list')}")
            logger.debug(f"  split: {sample_session.get('split')}")

        self.stats.update({
            "mixed_session_count": 0,
        })
        self._stats_lock = threading.Lock()
        self.mixed_sessions = []   # debug / verbose 模式下才用

        # 2. 处理所有会话
        self.process_sessions_sequentially(self.session_df)   # 顺序处理
        # self.process_sessions_parallel(self.session_df)
        
        # 3. # 划分数据集（基于all_split_session.csv中的split列）
        self.split_datasets()

        # 4. 保存结果
        self.save_results()

        logger.info(
            f"[SUMMARY] Mixed sessions skipped: "
            f"{self.stats['mixed_session_count']} / {len(self.session_df)} "
            f"({self.stats['mixed_session_count'] / len(self.session_df) * 100:.2f}%)"
        )
        logger.debug(f"First 100 mixed sessions: {self.mixed_sessions[:100]}")


    def process_sessions_sequentially(self, session_df: pd.DataFrame):
        """单线程处理所有会话"""
        logger.info(f"Building graphs from sessions (single-threaded)...")
        
        if not hasattr(self, 'graph_split_list_by_session'):
            self.graph_split_list_by_session = []

        # 单线程顺序处理所有session
        for _, session_row in tqdm.tqdm(session_df.iterrows(), total=len(session_df), desc="Processing sessions"):
            self.process_single_session(session_row)
        
        # 后续的统计和数据划分逻辑保持不变
        if not self.graph_list:
            logger.info("No graphs were built!")
            return
        
        logger.info(f"[SUMMARY] Total sessions: {len(self.session_df)}, \
                total graphs: {self.stats['total_graphs']}, \
                avg flows per graph: {self.stats['total_flows']/self.stats['total_graphs']:.2f}, \
                avg bursts per graph: {self.stats['total_bursts']/self.stats['total_graphs']:.2f}, \
                avg edges per graph: {self.stats['total_edges']/len(self.graph_list):.2f}")

    def process_sessions_parallel(self, session_df: pd.DataFrame):
        """
        多线程处理所有会话（修正版：严格保证时序顺序）
        """
        logger.info(f"Building graphs from sessions (chunked parallel, thread_count={self.thread_count})...")

        if not hasattr(self, 'graph_split_list_by_session'):
            self.graph_split_list_by_session = []

        session_count = len(session_df)
        chunk_size = min(100, session_count // (self.thread_count * 4))
        chunk_size = max(1, chunk_size)  # 防止为0

        session_idx_ranges = [
            (i, min(i + chunk_size, session_count))
            for i in range(0, session_count, chunk_size)
        ]

        # 1. 提交任务，并保存 futures 列表（按提交顺序，即时间顺序）
        futures = []
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            for start_idx, end_idx in session_idx_ranges:
                # 提交纯函数任务，返回数据而非修改self
                futures.append(executor.submit(
                    self._process_session_range_pure,
                    session_df,
                    start_idx,
                    end_idx
                ))

            # 2. 按顺序收集结果 (Crucial for Temporal Order)
            # 即使后面的任务先完成，这里也会等待前面的任务result()返回，从而保证合并顺序
            for future in tqdm.tqdm(futures, desc="Processing chunks (Ordered Merge)"):
                try:
                    # 获取该 chunk 的所有结果
                    chunk_results = future.result()

                    if not chunk_results:
                        continue

                    # 3. 将结果按顺序合并到主列表
                    # chunk_results 是一个列表，包含: (graph, node_uids, label_name, label_id, is_mal, split)
                    for res in chunk_results:
                        (g, uids, lname, lid, mal, split) = res

                        self.graph_list.append(g)
                        self.graph_node_uids_list.append(uids)
                        self.label_name_list.append(lname)
                        self.label_id_list.append(lid)
                        self.is_malicious_list.append(mal)
                        self.graph_split_list_by_session.append(split)

                        # 更新统计 (使用锁或者是累加，这里单线程合并是安全的，不需要锁)
                        # 但为了保持兼容性，直接更新stats字典
                        # 注意：stats在并行中很难精确追踪 burst/edges 细节除非从 res 返回，
                        # 这里为了简化，仅在最后做简单统计，或者你可以让 pure 函数也返回 stats。
                        self.stats["total_graphs"] += 1
                        self.stats["processed_session_count"] += 1

                except Exception as e:
                    logger.exception("Future crashed or merge failed")

        # 最终统计日志...
        if not self.graph_list:
            logger.info("No graphs were built!")
            return

        logger.info(f"[SUMMARY] Ordered parallel processing finished. Total graphs: {len(self.graph_list)}")

    def _process_session_range_pure(self, session_df, start_idx, end_idx):
        """
        [新增辅助方法]
        处理一个范围的 session，返回结果列表，不产生副作用（不修改 self.graph_list）。
        """
        local_results = []  # 存储 (graph, uids, label_name, label_id, is_mal, split)

        for idx in range(start_idx, end_idx):
            session_row = session_df.iloc[idx]
            try:
                # 1. 广播地址过滤
                if self.contains_broadcast_ip_address(session_row.get('session_index')):
                    continue

                session_split = str(session_row.get('split', 'train')).strip().lower()

                # 2. 构建图 (复用现有逻辑)
                g_list, node_uids_list, stats = self.build_graphs_from_session(session_row)

                if len(g_list) != len(node_uids_list):
                    continue

                # 3. 遍历生成的图 (一个session可能生成多个图)
                for i in range(len(g_list)):
                    current_graph = g_list[i]
                    current_node_uids = node_uids_list[i]

                    # 聚合标签
                    label_name, label_id, is_malicious = self.session_parser.aggregate_session_label(current_node_uids)

                    if label_name == "mixed":
                        continue  # 跳过 mixed

                    # 4. 存入局部结果列表
                    local_results.append((
                        current_graph,
                        current_node_uids,
                        label_name,
                        label_id,
                        is_malicious,
                        session_split
                    ))

            except Exception as e:
                # 仅记录错误，不中断
                pass

        return local_results


    @staticmethod
    def _extract_ip_candidates(session_index) -> List[str]:
        """
        粗粒度提取 IP 候选：
        - 若是 tuple / list：返回其中所有字段的字符串形式
        - 若是字符串形式的 tuple / list：literal_eval 后同样处理
        - 若是单字符串：直接作为唯一候选
        后续由 _is_single_ip_broadcast 严格判定
        """
        if session_index is None:
            return []

        # 已经是 tuple / list
        if isinstance(session_index, (tuple, list)):
            return [str(x).strip() for x in session_index if x is not None]

        # 字符串
        if isinstance(session_index, str):
            s = session_index.strip()
            if not s:
                return []

            # 尝试解析字符串形式的 tuple / list
            if (s.startswith("(") and s.endswith(")")) or (s.startswith("[") and s.endswith("]")):
                try:
                    obj = ast.literal_eval(s)
                    if isinstance(obj, (tuple, list)):
                        return [str(x).strip() for x in obj if x is not None]
                except Exception:
                    pass

            # 否则当作单字段
            return [s]

        return []

    @staticmethod
    def _is_single_ip_broadcast(ip_str: str) -> bool:
        ip_str = ip_str.strip()

        # ---------- IPv4 ----------
        try:
            ipv4 = ipaddress.IPv4Address(ip_str)

            # 受限广播
            if ip_str == "255.255.255.255":
                return True

        except ipaddress.AddressValueError:
            pass

        # ---------- IPv6 ----------
        try:
            ipv6 = ipaddress.IPv6Address(ip_str)
            return ipv6.is_multicast or ipv6.is_unspecified
        except ipaddress.AddressValueError:
            pass

        return False

    @staticmethod
    def contains_broadcast_ip_address(session_index) -> bool:
        """
        判断 session_index 中是否包含广播 / 组播 / 未指定 IP。
        兼容：
        - 单 IP
        - ('ip1','ip2')
        - 字符串形式 tuple/list
        """
        ip_list = SessionGraphBuilder._extract_ip_candidates(session_index)

        for ip_str in ip_list:
            if SessionGraphBuilder._is_single_ip_broadcast(ip_str):
                return True

        return False

    def process_session_range(self, session_df: pd.DataFrame, start_idx: int, end_idx: int):
        """处理 session_df[start:end] 这一段"""
        for idx in range(start_idx, end_idx):
            session_row = session_df.iloc[idx]
            try:
                self.process_single_session(session_row)
            except Exception:
                logger.exception(f"Error processing session index={idx}")


    def process_single_session(self, session_row):
        """单线程处理单个session_row的函数"""
        try:
            session_index = session_row.get('session_index')  # 根据实际列名调整
            # 检测session_index是否为广播地址，是则直接跳过后续处理
            if self.contains_broadcast_ip_address(session_row['session_index']):
                if verbose:
                    logger.debug(f" Session {session_row['session_index']} 被广播地址过滤")
                with self._stats_lock:
                    self.stats.setdefault("broadcast_session_count", 0)
                    self.stats["broadcast_session_count"] += 1                    
                return None

            # 获取当前session的split信息
            session_split = str(session_row.get('split', 'unknown')).strip().lower()   # [修改] 不要默认 'train'，如果缺失则设为 'unknown' 以便后续过滤
            
            # 处理当前session_row
            g_list, node_uids_list, stats = self.build_graphs_from_session(session_row)
            
            # 验证g_list与node_uids_list长度必须一致，不一致则抛出明确异常
            if len(g_list) != len(node_uids_list):
                raise ValueError(
                    f"g_list与node_uids_list长度不匹配！"
                    f"g_list长度: {len(g_list)}, node_uids_list长度: {len(node_uids_list)}"
                )

            # 按索引遍历（覆盖0到len(g_list)-1的所有有效索引，用户"0或1"表述应为笔误，实际需全量遍历）
            for idx in range(len(g_list)):
                # 1 逐个提取当前索引对应的图和节点UID列表
                current_graph = g_list[idx]
                current_node_uids = node_uids_list[idx]

                # 2 为当前图匹配对应的标签和split信息（确保与当前graph/node_uids严格对齐）
                label_name, label_id, is_malicious = self.session_parser.aggregate_session_label(current_node_uids)

                if label_name == "mixed":
                    with self._stats_lock:
                        self.stats["mixed_session_count"] += 1

                        # 仅在 verbose 或前 N 个打印，避免刷屏
                        if verbose and self.stats["mixed_session_count"] <= 10:
                            logger.warning(
                                f"[MIXED_SESSION] session_index={session_row.get('session_index')}, "
                                f"node_uids={current_node_uids}"
                            )

                        if len(self.mixed_sessions) < 10: # 只保留前 10 / 100 / 1000 个mixed session。
                            self.mixed_sessions.append(session_index)

                    continue  # skip this “mixed” graph, not whole session

                # 3. append到全局列表（等效原extend，且更严谨）
                with self._graph_list_lock:
                    self.graph_list.append(current_graph)  # 替代原extend，按索引逐个添加
                    self.graph_node_uids_list.append(current_node_uids)  # 与graph_list一一对应
                    self.label_name_list.append(label_name)
                    self.label_id_list.append(label_id)
                    self.is_malicious_list.append(is_malicious)
                    self.graph_split_list_by_session.append(session_split)  # 每个图绑定对应的split

            # 更新统计信息
            with self._stats_lock:
                self.stats["total_flows"] += stats["num_flows"]
                self.stats["total_bursts"] += stats["num_bursts"]
                self.stats["total_edges"] += stats["num_edges_avg"] * len(g_list)
                self.stats["total_graphs"] += stats["num_graphs"]
                self.stats["processed_session_count"] += 1

            # 调试打印：只打印前5条（原逻辑保留，优化Graph id计算，确保唯一）
            if verbose:
                if self.stats["processed_session_count"] < 5:  # 只打印前5个会话
                    logger.debug(f" Processing session {session_index}, split: {session_split}")
                    # Graph id = 初始长度 + 当前索引（避免原len(self.graph_list)可能的计数偏差）
                    logger.info(
                        f"[DEBUG] Graph id: {len(self.graph_list) - 1}, "
                        f"label_name: {label_name}, label_id: {label_id}, "
                        f"session_index: {session_index}, split: {session_split}, "
                        f"当前session处理进度: {idx+1}/{len(g_list)}, "
                        f"processed_session_count: {self.stats['processed_session_count']}"
                    )


            # 垃圾回收（线程安全，由主线程计数触发）
            # if self.stats["processed_session_count"] % self.gc_interval == 0:
            #     gc.collect()

        except Exception as e:
            logger.exception(f"Error processing session_index={session_row.get('session_index')}")
            return None

    # ========================================================================
    # 统一三类视图：Flow / SSL / X509 / DNS
    # 所有特征从 flow_node_builder.global_dims 获取固定维度
    # ========================================================================

    def build_flow_relation_graph(self, nodes, edges):
        """
        构建 DGL Graph（最新版）
        使用 FlowNodeBuilder.global_dims 确定所有特征维度
        """

        # 数据质量验证和修复
        is_clean, fixed_count = self._validate_and_fix_node_features(nodes)
        
        if not is_clean:
            logger.warning(f"⚠️ 检测并修复了 {fixed_count} 个NaN/Inf值")

        global_node_feature_dims = self.flow_node_builder.global_node_feature_dims
        enabled_views = self.flow_node_builder.enabled_views

        # -------- 固定维度查询 --------
        def dim(name):
            if "textual" in name:
                raise RuntimeError(f"Textual feature `{name}` should not query dim()")
            return int(global_node_feature_dims.get(name, 1))

        num_nodes = len(nodes)
        graph = dgl.graph([])
        graph.add_nodes(num_nodes)

        # ===================================================================
        # 通用矩阵构建函数（根据全局维度 padding/truncate）
        # ===================================================================
        def make_matrix(key, L, dtype):
            mat = np.zeros((num_nodes, L), dtype=dtype)
            for i, node in enumerate(nodes):
                vec = node.get(key, [])
                if not vec:
                    continue
                arr = np.asarray(vec, dtype=dtype)
                L0 = min(L, len(arr))
                mat[i, :L0] = arr[:L0]
            return mat

        # ===================================================================
        # 将所有视图放入 graph.ndata
        # ===================================================================

        def build_textual_ndata(nodes, key):
            for n in nodes:
                if key not in n:
                    raise KeyError(f"Node missing textual feature `{key}`")            
            
            sample = nodes[0][key]
            assert isinstance(sample, dict), f"{key} must be dict"
            assert "input_ids" in sample and "attention_mask" in sample, \
                f"{key} must contain input_ids and attention_mask"

            return {
                "input_ids": torch.cat(
                    [n[key]["input_ids"] for n in nodes], dim=0
                ),
                "attention_mask": torch.cat(
                    [n[key]["attention_mask"] for n in nodes], dim=0
                ),
            }
        # -------- Flow --------
        if enabled_views.get("flow_numeric_features", False):
            flow_num_mat = make_matrix("flow_numeric_features", dim("flow_numeric_features"), dtype=np.float32)
            graph.ndata["flow_numeric_features"] = torch.from_numpy(flow_num_mat).float()

        if enabled_views.get("flow_categorical_features", False):
            flow_cat_mat = make_matrix("flow_categorical_features", dim("flow_categorical_features"), dtype=np.int64)
            graph.ndata["flow_categorical_features"] = torch.from_numpy(flow_cat_mat).long()

        if enabled_views.get("flow_textual_features", False):
            t = build_textual_ndata(nodes, "flow_textual_features")
            graph.ndata["flow_textual_input_ids"] = t["input_ids"]
            graph.ndata["flow_textual_attention_mask"] = t["attention_mask"]

        # -------- SSL --------
        if enabled_views.get("ssl_numeric_features", False):
            ssl_num_mat = make_matrix("ssl_numeric_features", dim("ssl_numeric_features"), dtype=np.float32)            
            graph.ndata["ssl_numeric_features"] = torch.from_numpy(ssl_num_mat).float()

        if enabled_views.get("ssl_categorical_features", False):
            ssl_cat_mat = make_matrix("ssl_categorical_features", dim("ssl_categorical_features"), dtype=np.int64)
            graph.ndata["ssl_categorical_features"] = torch.from_numpy(ssl_cat_mat).long()

        if enabled_views.get("ssl_textual_features", False):
            t = build_textual_ndata(nodes, "ssl_textual_features")
            graph.ndata["ssl_textual_input_ids"] = t["input_ids"]          # [N, L]
            graph.ndata["ssl_textual_attention_mask"] = t["attention_mask"] # [N, L]            

        # -------- X509 --------
        if enabled_views.get("x509_numeric_features", False):
            x509_num_mat = make_matrix("x509_numeric_features", dim("x509_numeric_features"), dtype=np.float32)
            graph.ndata["x509_numeric_features"] = torch.from_numpy(x509_num_mat).float()

        if enabled_views.get("x509_categorical_features", False):
            x509_cat_mat = make_matrix("x509_categorical_features", dim("x509_categorical_features"), dtype=np.int64)            
            graph.ndata["x509_categorical_features"] = torch.from_numpy(x509_cat_mat).long()

        if enabled_views.get("x509_textual_features", False):
            t = build_textual_ndata(nodes, "x509_textual_features")
            graph.ndata["x509_textual_input_ids"] = t["input_ids"]
            graph.ndata["x509_textual_attention_mask"] = t["attention_mask"]

        # -------- DNS --------
        if enabled_views.get("dns_numeric_features", False):
            dns_num_mat = make_matrix("dns_numeric_features", dim("dns_numeric_features"), dtype=np.float32)
            graph.ndata["dns_numeric_features"] = torch.from_numpy(dns_num_mat).float()

        if enabled_views.get("dns_categorical_features", False):
            dns_cat_mat = make_matrix("dns_categorical_features", dim("dns_categorical_features"), dtype=np.int64)
            graph.ndata["dns_categorical_features"] = torch.from_numpy(dns_cat_mat).long()

        if enabled_views.get("dns_textual_features", False):
            t = build_textual_ndata(nodes, "dns_textual_features")
            graph.ndata["dns_textual_input_ids"] = t["input_ids"]
            graph.ndata["dns_textual_attention_mask"] = t["attention_mask"]

        # -------- Packet Sequences --------
        if enabled_views.get("packet_len_seq", False):
            pkt_len_mat = make_matrix("packet_len_seq", dim("packet_len_seq"), dtype=np.float32)
            graph.ndata["packet_len_seq"] = torch.from_numpy(pkt_len_mat).float()

        if enabled_views.get("packet_iat_seq", False):
            pkt_iat_mat = make_matrix("packet_iat_seq", dim("packet_iat_seq"), dtype=np.float32)
            graph.ndata["packet_iat_seq"] = torch.from_numpy(pkt_iat_mat).float()

        if "packet_seq_mask" in nodes[0]:
            graph.ndata["packet_seq_mask"] = torch.tensor(
                [n["packet_seq_mask"] for n in nodes], dtype=torch.bool
            )

        # -------- 选项：域名概率向量 --------
        if enabled_views.get("domain_probs", False):
            dom_len = dim("domain_probs")
            dom_mat = make_matrix("domain_probs", dom_len, dtype=np.float32)
            graph.ndata["domain_probs"] = torch.from_numpy(dom_mat).float()

        # ===================================================================
        # 辅助字段：ts / id / burst_id
        # ===================================================================
        graph.ndata["burst_id"] = torch.tensor(
            [n.get("burst_id", 0) for n in nodes], dtype=torch.long
        )
        graph.ndata["ts"] = torch.tensor(
            [n.get("ts", 0.0) for n in nodes], dtype=torch.float32
        )
        graph.ndata["id"] = torch.tensor(
            [n.get("id", 0) for n in nodes], dtype=torch.long
        )

        # ===================================================================
        # 添加边
        # ===================================================================
        if edges:
            if len(edges[0]) == 2:
                src, dst = zip(*edges)
                graph.add_edges(list(src), list(dst))

            elif len(edges[0]) == 3:
                src, dst, etype = zip(*edges)
                graph.add_edges(list(src), list(dst))
                graph.edata["etype"] = torch.tensor(etype, dtype=torch.long)

            else:
                raise ValueError("Edge tuple must be (src, dst) or (src, dst, etype)")

        # 强一致性检查（建议保留）
        for feat_name, feat in graph.ndata.items():
            if isinstance(feat, dict):
                # textual feature：检查 input_ids / attention_mask
                for sub_key, sub_tensor in feat.items():
                    if sub_tensor.shape[0] != num_nodes:
                        raise ValueError(
                            f"[FATAL] 特征 {feat_name}.{sub_key} 的节点数 "
                            f"{sub_tensor.shape[0]} 与图节点数 {num_nodes} 不一致！"
                        )
            else:
                # 普通 tensor 特征
                if feat.shape[0] != num_nodes:
                    raise ValueError(
                        f"[FATAL] 特征 {feat_name} 的节点数 {feat.shape[0]} "
                        f"与图节点数 {num_nodes} 不一致！"
                    )        
        
        # session graph 中存在入度为 0 的节点
        # 这在你的场景里是必然的，比如：
        # session 中的第一个 flow（没有被任何 flow 指向）
        # trigger / concurrent 边是有向的：etype 0=concurrent, 1=trigger
        # 单节点 session（只有 1 个 flow）
        # 而 GAT / GATv2 的消息传递规则是：
        # 节点只能从「入边」聚合信息
        # 入度为 0 ⇒ 没有任何可聚合的信息 ⇒ 输出未定义
        # 所以建图阶段加 self-loop，这样语义清晰：节点至少能看到自己，GAT / GIN / GraphSAGE 全都安全
        # graph = dgl.add_self_loop(graph)
        # self-loop 语义：concurrent (etype=0)
        graph = self.add_self_loop_with_etype(graph, self_loop_etype=0)

        assert graph.num_edges() == graph.edata["etype"].shape[0], \
            f"Edge count mismatch: {graph.num_edges()} vs {graph.edata['etype'].shape[0]}"
        
        return graph # 返回图和对应的节点uid列表

    def _validate_and_fix_node_features(self, nodes):
        """验证并修复节点特征中的NaN值"""
        issues_found = False
        fixed_count = 0

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
        
        for i, node in enumerate(nodes):
            numeric_fields = ['flow_numeric_features', 'ssl_numeric_features', 
                            'x509_numeric_features', 'dns_numeric_features']
            
            for field in numeric_fields:
                if field in node and node[field]:
                    features = node[field]
                    new_features = []
                    nan_count = 0
                    
                    for value in features:
                        if is_nan_or_inf(value):
                            new_features.append(0.0)  # 修复为0
                            nan_count += 1
                        else:
                            new_features.append(value)
                    
                    if nan_count > 0:
                        issues_found = True
                        fixed_count += nan_count
                        node[field] = new_features
                        
                        logger.debug(f"修复节点 {i} 的 {field}: {nan_count}个NaN值")
        
        return not issues_found, fixed_count

    def add_self_loop_with_etype(self, graph, self_loop_etype=0):
        device = graph.device
        num_nodes = graph.num_nodes()

        # self-loop 边
        src = torch.arange(num_nodes, device=device)
        dst = torch.arange(num_nodes, device=device)
        self_loop_etype_tensor = torch.full(
            (num_nodes,),
            fill_value=self_loop_etype,
            dtype=torch.long,
            device=device
        )

        # 先取原 etype（如果存在）
        if "etype" in graph.edata:
            old_etype = graph.edata["etype"]
        else:
            old_etype = None

        # 添加 self-loop 边
        graph.add_edges(src, dst)

        # 重新设置 etype（一次性）
        if old_etype is not None:
            graph.edata["etype"] = torch.cat(
                [old_etype, self_loop_etype_tensor],
                dim=0
            )
        else:
            graph.edata["etype"] = self_loop_etype_tensor

        return graph


    def build_graphs_from_session(self, session_row):
        """为每个session构建流关系图（重构版）"""
        # 1. 提取流记录
        flows = self._extract_flow_records(session_row)
        if not flows:
            if verbose:
                logger.debug(f"[NO_GRAPH] session={session_row.get('session_index')}, "
                    f"split={session_row.get('split')}, "
                    f"reason=no_flows_or_invalid_uids")            
            return [], [], {"num_flows": 0, "num_bursts": 0, "num_edges_avg": 0, "num_graphs": 0}
        else:
            if verbose:
                logger.debug(f" Session {session_row['session_index']} 提取到 {len(flows)} 条Flow记录")
        
        # 2. 排序流记录
        flows.sort(key=lambda x: x['ts'])
        
        # 3. 构建图
        graph_list, node_uids_list, edges_count_list, num_bursts = self.build_graphs_from_flows(flows)
        
        # 6. 统计信息
        stats = {
            "num_flows": len(flows),
            "num_bursts": num_bursts,
            "num_edges_avg": int(np.mean(edges_count_list)) if edges_count_list else 0,
            "num_graphs": len(graph_list)
        }
        
        return graph_list, node_uids_list, stats


    def _extract_flow_records(self, session_row):
        """从session行中提取流记录（健壮解析版）"""
        flow_uid_raw = session_row.get('flow_uid_list', None)
        session_index = session_row.get('session_index', 'unknown')
        
        if verbose:
            logger.debug(f" Session {session_index}: flow_uid_raw = {flow_uid_raw}")
        
        # ---------- 1️⃣ 解析 flow_uid_list ----------
        flow_uid_list = []
        if isinstance(flow_uid_raw, str):
            try:
                parsed = ast.literal_eval(flow_uid_raw)  # 尝试将字符串安全地转为 Python 对象
                if isinstance(parsed, list):
                    flow_uid_list = [str(uid).strip() for uid in parsed if str(uid).strip()]
                else:
                    logger.warning(f" 非列表类型 flow_uid_list：{type(parsed)} session={session_row.get('session_index')}")
            except Exception as e:
                logger.error(f" 无法解析 flow_uid_list (session={session_row.get('session_index')}): {e}")
        elif isinstance(flow_uid_raw, list):
            # 已经是列表（极少数情况）
            flow_uid_list = flow_uid_raw
        else:
            logger.warning(f" flow_uid_list 类型异常: {type(flow_uid_raw)} session={session_row.get('session_index')}")
        
        if not flow_uid_list:
            if verbose:
                logger.debug(f"[EMPTY_UID] session={session_row.get('session_index')}, split={session_row.get('split')}")
            return []
        else:
            if verbose:
                logger.debug(f" Session {session_index}: 解析到 {len(flow_uid_list)} 个flow_uid")
        
        # ---------- 2️⃣ 构建 flow_record ----------
        flows = []
        missing_uids = 0
        for flow_uid in flow_uid_list:
            flow_record = self.flow_node_builder.get_flow_record(flow_uid)
            if verbose:
                logger.debug(f" Flow UID: {flow_uid}, Record: {flow_record is not None}")
            if flow_record:
                if verbose:
                    logger.debug(f" Flow TS: {flow_record.get('ts')}, Conn TS: {flow_record.get('conn.ts')}")
                flows.append(flow_record)
            else:
                missing_uids += 1

        # ---------- 3️⃣ 调试统计 ----------
        if verbose and missing_uids > 0:
            logger.debug(
                f"[LAZY_FILTER] session={session_index}: "
                f"dropped {missing_uids} / {len(flow_uid_list)} flows "
                f"due to flow-level filtering and sampling"
            )

        return flows


    def create_concurrent_edges(self, burst, edges, mode="k_nearest", k=3):
        """
        在 burst 内创建边，根据不同策略：
        - full_connect: 全连接 O(n^2)
        - chain_connect: 链式相连 O(n)
        - k_nearest: 每个节点与时间上最近的 k 个相连 O(kn)
        """
        n = len(burst)
        if n <= 1:
            return
        
        if mode == "full_connect":
            for j in range(n - 1):
                for k in range(j + 1, n):
                    edges.append((burst[j]['id'], burst[k]['id'], 0)) # etype 0=concurrent, 1=trigger

        elif mode == "chain_connect":
            # 顺序链式连接
            for j in range(n - 1):
                edges.append((burst[j]['id'], burst[j+1]['id'], 0)) # etype 0=concurrent, 1=trigger

        elif mode == "k_nearest":
            # 假设 burst 内已按时间排序
            for j in range(n):
                for d in range(1, k+1):
                    if j + d < n:
                        edges.append((burst[j]['id'], burst[j+d]['id'], 0)) # etype 0=concurrent, 1=trigger
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
    def build_graphs_from_flows(self, flows):
        if not flows or len(flows) == 0:
            return [], [], [], 0
              
        # ==================== 1. Flow Burst聚类 ====================
        flow_bursts = self._cluster_flows_into_bursts(flows)
        
        # ==================== 2. 会话窗口聚合 ====================
        def calculate_avg_interval(window_flows):
            """计算窗口中流之间的平均时间间隔"""            
            if len(window_flows) <= 1: # 如果windows的流数量少于2，那么返回缺省的流数量
                return self.concurrent_flow_iat_threshold 
            times = [f['ts'] for f in window_flows]
            return (max(times) - min(times)) / (len(window_flows) - 1)
        
        session_windows = []
        current_window = list(flow_bursts[0])
        if verbose:
            logger.debug(f" 初始窗口: {len(current_window)}个流, 平均间隔: {calculate_avg_interval(current_window):.2f}s")

        for burst_idx in range(1, len(flow_bursts)):
            flow_burst = flow_bursts[burst_idx]
            avg_before = calculate_avg_interval(current_window)
            avg_after = calculate_avg_interval(current_window + flow_burst)

            if verbose:
                logger.debug(f" Burst {burst_idx}: {len(flow_burst)}个流")
                logger.debug(f"   合并前平均间隔: {avg_before:.2f}s, 合并后平均间隔: {avg_after:.2f}s")
                logger.debug(f"   阈值检查: {avg_after:.2f}s <= {self.sequential_flow_iat_threshold * avg_before:.2f}s ?")
            
            if avg_after <= self.sequential_flow_iat_threshold * avg_before:
                current_window.extend(flow_burst)
                if verbose:
                    logger.debug(f"   ✅ 合并到当前窗口，窗口大小: {len(current_window)}")                
            else:
                session_windows.append(current_window)
                if verbose:
                    logger.debug(f"   ❌ 创建新窗口，原窗口大小: {len(current_window)}")                
                current_window = list(flow_burst)
                    
        if current_window and len(current_window) > 0:
            session_windows.append(current_window)
            if verbose:
                logger.debug(f" 最终窗口大小: {len(current_window)}")    
        
        # 最终统计
        if verbose:
            logger.debug(f" === 会话窗口聚合完成 ===")
            logger.debug(f" 总共生成 {len(session_windows)} 个会话窗口")
            for i, window in enumerate(session_windows):
                logger.debug(f" 窗口{i}: {len(window)}个流, 时间范围: {min([f['ts'] for f in window]):.1f} - {max([f['ts'] for f in window]):.1f}")

        # ==================== 3. 流关系图构建 ====================
        graph_list, node_uids_list, edges_count_list = [], [], []
        for session_win in session_windows:
            # 节点特征构建
            flow_uids = [flow['uid'] for flow in session_win]
            nodes = self.flow_node_builder.build_node_features(flow_uids)           
            if not nodes or len(nodes) == 0: 
                logger.error("流关系图构建：节点特征构建失败！检查build_node_features()")
                continue
            else:
                if verbose:
                    logger.debug(f" 流关系图构建：节点序列长度: {len(nodes)}")
                        
            # 并行边构建
            for idx, node in enumerate(nodes):
                node['id'] = idx            
            edges = []
            node_bursts = self._create_current_edges_by_node_clustering(nodes, edges) # burst聚类逻辑
            
            # 顺序边构建
            for j in range(len(node_bursts) - 1):
                current_node_burst = node_bursts[j]
                next_node_burst = node_bursts[j + 1]
                
                # 调试信息
                if len(node_bursts) > 1:
                    if verbose:
                        logger.debug(f" ✅ Burst连接 {j}→{j+1}: len(current_node_burst)={len(current_node_burst)}, len(next_node_burst)={len(next_node_burst)}, len(node_bursts)={len(node_bursts)}")
                
                if len(current_node_burst) == 1 and len(next_node_burst) == 1:
                    edges.append((current_node_burst[0]['id'], next_node_burst[0]['id'], 1))
                else:
                    edges.append((current_node_burst[-1]['id'], next_node_burst[-1]['id'], 1))
                    edges.append((current_node_burst[0]['id'], next_node_burst[0]['id'], 1))
            
            if nodes and len(nodes) > 0:
                node_uids = [node['uid'] for node in nodes]
                graph = self.build_flow_relation_graph(nodes, edges)
                graph_list.append(graph)
                node_uids_list.append(node_uids)
                edges_count_list.append(len(edges))
                if verbose:
                    logger.debug(f" 构建到 {len(edges)} 条边")

        if node_bursts and len(node_bursts) > 0:
            if verbose:
                logger.debug(f" 📊 总node_burst数: {len(node_bursts)}, 第一个node_burst大小: {len(node_bursts[0])}，最后一个node_burst大小: {len(node_bursts[-1])}")
        
        return graph_list, node_uids_list, edges_count_list, len(flow_bursts)

    def _cluster_flows_into_bursts(self, flows):
        if verbose:
            logger.debug(f" Flow时间戳序列: {[flow.get('ts', 0) for flow in flows]}")
            logger.debug(f" 时间差计算: {[abs(flows[i+1]['ts'] - flows[i]['ts']) for i in range(len(flows)-1)]}")

        flows.sort(key=lambda x: x['ts'])
        flow_bursts = []
        current_flow_burst = [flows[0]]
        
        for idx in range(1, len(flows)):
            flow = flows[idx]
            time_diff = abs(flow['ts'] - current_flow_burst[-1]['ts'])
            if time_diff <= self.concurrent_flow_iat_threshold:
                current_flow_burst.append(flow)
                if verbose:
                    logger.debug(f" 添加到当前flow burst (delta={time_diff:.2f}s)")
            elif current_flow_burst and len(current_flow_burst) > 0: # 确保当前burst不为空
                flow_bursts.append(current_flow_burst)
                if verbose:
                    logger.debug(f" 新建flow burst (delta={time_diff:.2f}s > {self.concurrent_flow_iat_threshold}s)")
                current_flow_burst = [flow]
        
        # 处理最后一个burst
        if current_flow_burst and len(current_flow_burst) > 0:
            flow_bursts.append(current_flow_burst)
        
        # 验证burst大小分布
        burst_sizes = [len(burst) for burst in flow_bursts]
        if burst_sizes and len(flow_bursts) > 0:
            if verbose:
                logger.debug(f" 📈 Flow burst大小分布: 总数={len(burst_sizes)}, 平均大小={np.mean(burst_sizes):.2f}")
                logger.debug(f" 📊 Flow burst大小详情: {burst_sizes}")
                logger.debug(f" 🔍 最后一个flow burst大小: {burst_sizes[-1]}")
            
        return flow_bursts
        
    def _create_current_edges_by_node_clustering(self, nodes, edges):        
        """窗口内burst聚类方法"""
        if not nodes or len(nodes) == 0:
            raise ValueError("节点列表为空，无法进行burst聚类")  # 正确的异常抛出
        
        node_bursts = []
        current_node_burst = [nodes[0]]
        burst_id_counter = 0  # 新增：burst_id计数器
        # 确保第一个节点有正确的burst_id
        nodes[0]['burst_id'] = burst_id_counter
        
        # 使用更精确的时间间隔计算
        for idx in range(1, len(nodes)):
            node = nodes[idx]
            # 计算与当前burst最后一个节点的时间差
            time_diff = abs(node['ts'] - current_node_burst[-1]['ts'])
            if time_diff <= self.concurrent_flow_iat_threshold:
                current_node_burst.append(nodes[idx])
                if verbose:
                    logger.debug(f" 添加到当前burst (delta={time_diff:.2f}s)")
                # 为同一burst内的节点设置相同burst_id
                nodes[idx]['burst_id'] = burst_id_counter                    
            elif current_node_burst and len(current_node_burst) > 0:  # 确保当前burst不为空
                # 完成当前burst
                self.create_concurrent_edges(current_node_burst, edges)
                node_bursts.append(current_node_burst)
                if verbose:
                    logger.debug(f" 新建burst (delta={time_diff:.2f}s > {self.concurrent_flow_iat_threshold}s)")

                # 开始新burst
                burst_id_counter += 1  # 新增burst_id
                current_node_burst = [node]
                # 使用递增后的burst_id
                nodes[idx]['burst_id'] = burst_id_counter
        
        # 处理最后一个burst
        if current_node_burst and len(current_node_burst) > 0:
            self.create_concurrent_edges(current_node_burst, edges)
            node_bursts.append(current_node_burst)
        
        # 验证burst大小分布
        burst_sizes = [len(burst) for burst in node_bursts]
        if burst_sizes and len(node_bursts) > 0:
            if verbose:
                logger.debug(f" 📈 Node Burst大小分布: 总数={len(burst_sizes)}, 平均大小={np.mean(burst_sizes):.2f}")
                logger.debug(f" 📊 Node Burst大小详情: {burst_sizes}")
                logger.debug(f" 🔍 最后一个node burst大小: {burst_sizes[-1]}")
        
        return node_bursts

    def _validate_sequential_edges(self, bursts, edges, sequential_edges_count):
        """验证sequential edges数量是否符合预期"""
        if not bursts:
            return
        
        expected_edges = 0
        validation_details = []
        
        # 计算预期的边数量
        for j in range(len(bursts) - 1):
            current_size = len(bursts[j])
            next_size = len(bursts[j + 1])
            
            if current_size == 1 and next_size == 1:
                expected_edges += 1
                validation_details.append(f"Burst {j}→{j+1}: 1条边 (1→1)")
            else:
                expected_edges += 2
                validation_details.append(f"Burst {j}→{j+1}: 2条边 ({current_size}→{next_size})")
        
        # 验证实际边数量
        actual_sequential_edges = len([e for e in edges if len(e) == 3 and e[2] == 1])
        
        # 记录验证结果
        validation_result = {
            'total_bursts': len(bursts),
            'expected_sequential_edges': expected_edges,
            'actual_sequential_edges': actual_sequential_edges,
            'sequential_edges_count': sequential_edges_count,
            'is_valid': expected_edges == actual_sequential_edges == sequential_edges_count,
            'details': validation_details
        }
        
        # 输出验证信息
        if not validation_result['is_valid']:
            logger.info(f"⚠️ SEQUENTIAL EDGES验证失败:")
            logger.info(f"   预期边数: {expected_edges}")
            logger.info(f"   实际边数: {actual_sequential_edges}")
            logger.info(f"   计数边数: {sequential_edges_count}")
            logger.info(f"   Burst数量: {len(bursts)}")
            
            for detail in validation_details:
                logger.info(f"   {detail}")
            
            # 调试信息：显示所有sequential edges
            sequential_edges = [e for e in edges if len(e) == 3 and e[2] == 1]
            logger.info(f"   Sequential edges详情:")
            for i, edge in enumerate(sequential_edges):
                logger.info(f"     边{i}: {edge[0]} → {edge[1]} (类型: {edge[2]})")
        elif verbose:
            logger.info(f"✅ Sequential edges验证通过: 存在{expected_edges}条顺序边，符合预期")
        
        # 存储验证结果（可选）
        self.last_validation_result = validation_result
        
        return validation_result

    def split_datasets(self):
        """根据session的split列划分数据集 - 修复版本"""
        if not self.graph_list:
            return

        # 确保所有列表长度一致
        min_len = min(len(self.graph_list), 
                    len(self.graph_split_list_by_session),
                    len(self.label_name_list),
                    len(self.label_id_list))
        
        if min_len != len(self.graph_list):
            logger.warning(f"列表长度不一致，将截断到最小长度 {min_len}")
            self.graph_list = self.graph_list[:min_len]
            self.graph_split_list_by_session = self.graph_split_list_by_session[:min_len]
            self.label_name_list = self.label_name_list[:min_len]
            self.label_id_list = self.label_id_list[:min_len]
        
        # 重新初始化索引列表
        self.train_index = []
        self.validate_index = []
        self.test_index = []
        
        # 统计split分布
        split_counts = {'train': 0, 'validate': 0, 'test': 0, 'other': 0}
        
        if verbose:
            logger.debug(f" 开始划分数据集，总图数: {len(self.graph_split_list_by_session)}")
        
        for idx, split in enumerate(self.graph_split_list_by_session):
            if split is None or pd.isna(split):
                split_str = 'unknown'  # 默认值
            else:
                split_str = str(split).strip().lower()
            
            # 更灵活的匹配逻辑
            if split_str == 'train':
                self.train_index.append(idx)
                split_counts['train'] += 1
            elif split_str in ['validate', 'valid', 'validation', 'val']:
                self.validate_index.append(idx)
                split_counts['validate'] += 1
            elif split_str in ['test', 'testing']:
                self.test_index.append(idx)
                split_counts['test'] += 1
            else:
                # 未知的split值，根据内容判断
                if 'test' in split_str:
                    self.test_index.append(idx)
                    split_counts['test'] += 1
                elif 'validate' in split_str or 'valid' in split_str:
                    self.validate_index.append(idx)
                    split_counts['validate'] += 1
                else:
                    if verbose and split_counts['other'] < 5:
                        logger.warning(f"Graph {idx} has unknown split '{split_str}', skipping (NOT adding to train).")
                    split_counts['other'] += 1
            
            # 调试信息
            if verbose and idx < 5:  # 只打印前5个
                logger.debug(f" 图 {idx}: split='{split}' -> 分配为 '{split_str}'")
        
        # 验证划分结果
        self._validate_split_consistency(split_counts)

    def _validate_split_consistency(self, split_counts):
        """验证划分一致性"""
        logger.info("\n=== 划分验证 ===")
        
        # 方法1：通过split列表计算
        calculated_train = []
        calculated_validate = []
        calculated_test = []
        
        for idx, split in enumerate(self.graph_split_list_by_session):
            if split is None or pd.isna(split):
                split_str = 'train'
            else:
                split_str = str(split).strip().lower()
            
            if split_str == 'train':
                calculated_train.append(idx)
            elif split_str in ['validate', 'valid', 'validation', 'val']:
                calculated_validate.append(idx)
            elif split_str == 'test':
                calculated_test.append(idx)
            else:
                calculated_train.append(idx)  # 默认分配到训练集
        
        # 方法2：直接使用存储的索引
        stored_train = self.train_index
        stored_validate = self.validate_index
        stored_test = self.test_index
        
        # 比较结果
        train_match = set(calculated_train) == set(stored_train)
        validate_match = set(calculated_validate) == set(stored_validate)
        test_match = set(calculated_test) == set(stored_test)
        
        logger.info(f"训练集匹配: {train_match} (计算: {len(calculated_train)}, 存储: {len(stored_train)})")
        logger.info(f"验证集匹配: {validate_match} (计算: {len(calculated_validate)}, 存储: {len(stored_validate)})")
        logger.info(f"测试集匹配: {test_match} (计算: {len(calculated_test)}, 存储: {len(stored_test)})")
        
        # 如果不匹配，显示差异
        if not validate_match:
            logger.info("\n验证集差异分析:")
            calc_only = set(calculated_validate) - set(stored_validate)
            stored_only = set(stored_validate) - set(calculated_validate)
            
            if calc_only:
                logger.info(f"计算有但存储无: {calc_only}")
                for idx in calc_only:
                    if idx < len(self.graph_split_list_by_session):
                        logger.info(f"  图 {idx}: split='{self.graph_split_list_by_session[idx]}'")
            
            if stored_only:
                logger.info(f"存储有但计算无: {stored_only}")
                for idx in stored_only:
                    if idx < len(self.graph_split_list_by_session):
                        logger.info(f"  图 {idx}: split='{self.graph_split_list_by_session[idx]}'")
        
        # 如果验证失败，使用计算的结果
        if not (train_match and validate_match and test_match):
            logger.warning("\n⚠️ 划分不一致，使用计算的结果")
            self.train_index = calculated_train
            self.validate_index = calculated_validate
            self.test_index = calculated_test
            
            # 更新统计信息
            split_counts = {
                'train': len(calculated_train),
                'validate': len(calculated_validate), 
                'test': len(calculated_test),
                'other': 0
            }
        
        # 最终统计
        total_graphs = len(self.graph_list)
        if total_graphs > 0:
            logger.info(f"===最终数据集划分===")
            logger.info(f"训练集: {len(self.train_index)} 图 ({len(self.train_index)/total_graphs:.1%})")
            logger.info(f"验证集: {len(self.validate_index)} 图 ({len(self.validate_index)/total_graphs:.1%})")
            logger.info(f"测试集: {len(self.test_index)} 图 ({len(self.test_index)/total_graphs:.1%})")
            

    def extract_train_flow_uids(self):
        """提取训练集中所有flow UID的集合（去重）"""
        train_flow_uids = set()
        
        for idx in self.train_index:
            node_uids = self.graph_node_uids_list[idx]  # 单个图的节点UID列表
            
            # 将当前图的节点UID添加到集合中（自动去重）
            train_flow_uids.update(node_uids)  # 正确用法：将列表元素添加到set
        
        return train_flow_uids  # 返回set，变量名改为_uids

    def save_results(self):
        """
        保存图和附加信息
        自动生成:
          xxx.bin   -> DGL 图
          xxx_info.pkl -> 标签/元信息
        """
        if not self.graph_list:
            logger.info("No graphs to save!")
            return
                
        base, _ = os.path.splitext(self.dumpFilename)
        bin_path = base + ".bin"
        info_path = base + "_info.pkl"

        logger.info(f"Saving graphs to {bin_path} ...")
        save_graphs(bin_path, self.graph_list)
        
        # 验证一致性
        assert len(self.label_name_list) == len(self.graph_split_list_by_session)
        
        # 验证索引与split列表的一致性
        calculated_train = [i for i, s in enumerate(self.graph_split_list_by_session) 
                        if s.lower() == 'train']
        calculated_validate = [i for i, s in enumerate(self.graph_split_list_by_session) 
                        if s.lower() in ['validate', 'valid', 'validation', 'val']]
        calculated_test = [i for i, s in enumerate(self.graph_split_list_by_session) 
                        if s.lower() == 'test']
        
        assert set(calculated_train) == set(self.train_index)
        assert set(calculated_validate) == set(self.validate_index) 
        assert set(calculated_test) == set(self.test_index)
         
        graph_infos = {
            # ===== 语义信息（不参与训练，仅用于分析/调试）=====
            'label_name': self.label_name_list,      # 每个图的标签名
            'label_id': self.label_id_list,          # 每个图的标签ID
            "is_malicious": self.is_malicious_list,  # 每个图的善意/恶意标签
            'split': self.graph_split_list_by_session,  # 每个图的划分信息

            # ===== 执行级索引（训练/验证/测试唯一可信来源）=====
            'train_index': self.train_index,        # 训练集索引（快速访问）
            'validate_index': self.validate_index,        # 验证集索引（快速访问）
            'test_index': self.test_index,           # 测试集索引（快速访问）

            # ===== ⭐ 各个视图的特征维度信息 =====
            "feature_dims": copy.deepcopy(dict(self.flow_node_builder.global_node_feature_dims)),
            "enabled_views": copy.deepcopy(dict(self.flow_node_builder.enabled_views)),
            # [新增] 保存每个图的流 UID 列表
            "node_uids": self.graph_node_uids_list,
        }
        
        save_info(info_path, graph_infos)
        
        logger.info("Graph construction completed successfully!")