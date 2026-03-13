import torch
from torch_geometric.data import Data
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class StreamingSessionGraphBuffer:
    def __init__(self,
                 concurrent_threshold=0.1,
                 sequential_threshold=1.0,
                 session_idle_timeout=120.0,
                 session_active_timeout=3600.0):
        """
        严格复刻离线建图逻辑的流式建图器
        """
        self.concurrent_threshold = concurrent_threshold
        self.sequential_threshold = sequential_threshold
        self.session_idle_timeout = session_idle_timeout
        self.session_active_timeout = session_active_timeout

        # 缓存池：session_id -> dict
        self.active_sessions = defaultdict(lambda: {
            'nodes_x': [],  # 存放动态更新后的流表征向量 x_i
            'timestamps': [],  # 存放流的发生时间
            'y_bin_list': [],  # 二分类标签列表
            'y_multi_list': []  # 多分类标签列表
        })

    def add_flow_and_check_completion(self, session_id, flow_timestamp, flow_x_tensor, y_bin, y_multi):
        """
        添加新流，如果导致 Session 超时，则返回打包好的 PyG Graph。
        """
        completed_graph = None
        session = self.active_sessions[session_id]

        # 检查是否超时触发建图
        if len(session['timestamps']) > 0:
            last_ts = session['timestamps'][-1]
            start_ts = session['timestamps'][0]

            is_idle = (flow_timestamp - last_ts) > self.session_idle_timeout
            is_active = (flow_timestamp - start_ts) > self.session_active_timeout

            if is_idle or is_active:
                completed_graph = self._build_topology_strictly(session)
                # 清空重新开始
                self.active_sessions[session_id] = {
                    'nodes_x': [], 'timestamps': [], 'y_bin_list': [], 'y_multi_list': []
                }
                session = self.active_sessions[session_id]

        # 存入新流
        session['nodes_x'].append(flow_x_tensor.detach().cpu())
        session['timestamps'].append(flow_timestamp)
        session['y_bin_list'].append(y_bin.detach().cpu())
        session['y_multi_list'].append(y_multi.detach().cpu())

        return completed_graph

    def force_flush_all(self):
        """在所有数据流结束时，强制打包所有还未超时的 Session"""
        graphs = []
        for sid, session in self.active_sessions.items():
            if len(session['nodes_x']) > 0:
                g = self._build_topology_strictly(session)
                if g is not None:
                    graphs.append(g)
        self.active_sessions.clear()
        return graphs

    def _build_topology_strictly(self, session):
        """【核心拓扑复刻】基于 Burst 的并发边与顺序边"""
        num_nodes = len(session['nodes_x'])
        if num_nodes == 0:
            return None

        x = torch.cat(session['nodes_x'], dim=0)  # [num_nodes, 768]
        timestamps = session['timestamps']

        edges = []
        current_burst_nodes = [0]

        for i in range(1, num_nodes):
            time_diff = timestamps[i] - timestamps[i - 1]
            if time_diff <= self.concurrent_threshold:
                current_burst_nodes.append(i)
                # Burst 内部并发边 (星型全连接)
                for prev_idx in current_burst_nodes[:-1]:
                    edges.append([prev_idx, i])
                    edges.append([i, prev_idx])
            else:
                if time_diff <= self.sequential_threshold:
                    # 顺序边：上一个 Burst 指向新 Burst
                    for prev_idx in current_burst_nodes:
                        edges.append([prev_idx, i])
                current_burst_nodes = [i]

        if len(edges) > 0:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # 图级别标签：只要有恶意流即为恶意，多分类取并集
        y_bin = torch.stack(session['y_bin_list']).max().view(1).float()
        y_multi = torch.stack(session['y_multi_list']).max(dim=0)[0].unsqueeze(0).float()

        return Data(x=x, edge_index=edge_index, y_bin=y_bin, y_multi=y_multi)