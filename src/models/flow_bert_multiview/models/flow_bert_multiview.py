import importlib
from datetime import datetime
from concept_drift_detect.concept_drift_detector import ConceptDriftManager
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from omegaconf import DictConfig
import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import os, sys
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, roc_auc_score,
    average_precision_score, confusion_matrix,
)
# 注释掉原来的SHAP导入，使用新的通用框架
# import shap    # 可视化 added  by qinyf
from pathlib import Path
import json

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

# 导入新的通用SHAP分析框架
try:
    # 添加../../utils目录到Python搜索路径
    hyper_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'hyper_optimus')
    sys.path.insert(0, hyper_path) 
    # from shap_analysis import SHAPAnalyzeMixin,ShapAnalyzer

    from shap_analysis import ShapAnalyzer
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

try:
    # 请确保 concept_drift_detector.py 文件中包含了 ConceptDriftManager 类
    # 路径根据您的实际项目结构可能需要微调
    from concept_drift_detect.concept_drift_detector import ConceptDriftManager
    DRIFT_MANAGER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"无法导入 ConceptDriftManager: {e}。概念漂移检测功能将不可用。")
    DRIFT_MANAGER_AVAILABLE = False

# =============================================
# Safe Transformers Import（仅保留 BERT + schedulers）
# =============================================
try:
    from transformers import (
        BertModel,
        BertTokenizer,
        BertConfig,
        get_linear_schedule_with_warmup,
        get_constant_schedule_with_warmup,
        get_cosine_schedule_with_warmup,
    )
    _TRANSFORMERS_IMPORT_ERROR = None
except ImportError as exc:  # pragma: no cover - optional dependency guard
    BertModel = BertTokenizer = BertConfig = None  # type: ignore[assignment]
    get_linear_schedule_with_warmup = None  # type: ignore[assignment]
    get_constant_schedule_with_warmup = None  # type: ignore[assignment]
    get_cosine_schedule_with_warmup = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc

try:
    from transformers import AdamW as _HFAdamW
except ImportError:  # pragma: no cover - optional dependency guard
    _HFAdamW = None

from torch.optim import AdamW as _TorchAdamW

AdamW = _HFAdamW or _TorchAdamW

def _require_transformers() -> None:
    """确保 transformers 已正确安装"""
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise ImportError(
            "需要安装 transformers 才能使用 `FlowBertMultiview`，请运行 `pip install transformers`。"
        ) from _TRANSFORMERS_IMPORT_ERROR

class SequenceEncoder(nn.Module):
    """序列编码器 - 处理不定长序列特征"""
    
    def __init__(
        self, 
        embedding_dim: int, 
        num_layers: int,
        num_heads: int,  # 多头注意力头数，统一使用 num_heads
        dropout: float = 0.1,
        max_seq_length: int = 1000  # 提供默认值以保持向后兼容，但推荐从外部传入
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # 序列特征嵌入层
        self.direction_projection = nn.Linear(1, embedding_dim)
        self.payload_projection = nn.Linear(1, embedding_dim)
        self.iat_projection = nn.Linear(1, embedding_dim)
        self.packet_number_projection = nn.Linear(1, embedding_dim)
        self.avg_payload_projection = nn.Linear(1, embedding_dim)
        self.duration_projection = nn.Linear(1, embedding_dim)

        # 特征融合投影层：融合方向、载荷、IAT、数据包数、平均载荷大小、持续时间 六个维度
        self.feature_fusion_projection = nn.Linear(6 * embedding_dim, embedding_dim)

        # 位置编码
        self.position_embedding = nn.Embedding(max_seq_length, embedding_dim)
        
        # 添加LayerNorm层提高稳定性
        # self.direction_norm = nn.LayerNorm(embedding_dim)
        # self.payload_norm = nn.LayerNorm(embedding_dim)
        # self.iat_norm = nn.LayerNorm(embedding_dim)
        # self.packet_number_norm = nn.LayerNorm(embedding_dim)
        # self.avg_payload_norm = nn.LayerNorm(embedding_dim)  
        # self.duration_norm = nn.LayerNorm(embedding_dim)
        # self.feature_fusion_norm = nn.LayerNorm(embedding_dim)
        # self.combined_norm = nn.LayerNorm(embedding_dim) 
        
        # Transformer编码器
        # nn.TransformerEncoderLayer 参数说明：
        #   - d_model: 输入维度
        #   - nhead: 多头注意力头数（注意：这里参数名是 nhead，不是 num_heads）
        #   - dim_feedforward: 前馈网络隐藏层大小
        #   - dropout: dropout比例
        #   - batch_first: True 表示输入为 [batch, seq_len, feature_dim]
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=4 * embedding_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 注意力池化
        # nn.MultiheadAttention 参数说明：
        #   - embed_dim: 输入特征维度
        #   - num_heads: 多头注意力头数（这里参数名是 num_heads）
        #   - dropout: dropout比例
        #   - batch_first: True 表示输入为 [batch, seq_len, feature_dim]
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=embedding_dim, 
            num_heads=num_heads, 
            dropout=dropout, 
            batch_first=True
        )
        
    def forward(self, sequence_data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            sequence_data: 包含序列特征的字典（修改后的结构）
                - directions: [batch_size, seq_len] 传输方向的序列
                - payload_sizes: [batch_size, seq_len] 载荷大小序列
                - iat_times: [batch_size, seq_len] 时间间隔序列
                - packet_numbers: [batch_size, seq_len] 数据包数量序列
                - avg_payload_sizes: [batch_size, seq_len] 平均载荷大小序列           
                - durations: [batch_size, seq_len] 持续时间序列
                - sequence_mask: [batch_size, seq_len] 序列掩码
                
        Returns:
            sequence_embeddings: [batch_size, embedding_dim] 序列嵌入表示
        """
        directions = sequence_data['directions']
        payload_sizes = sequence_data['payload_sizes']        
        iat_times = sequence_data['iat_times']
        packet_numbers = sequence_data['packet_numbers']
        avg_payload_sizes = sequence_data['avg_payload_sizes']
        durations = sequence_data['durations']
        sequence_mask = sequence_data['sequence_mask']

        batch_size, seq_len = directions.shape
               
        # 1️⃣ 内容特征 embedding
        directions_emb = self.direction_projection(directions.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # directions_emb = self.direction_norm(directions_emb)

        payload_emb = self.payload_projection(payload_sizes.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # payload_emb = self.payload_norm(payload_emb)        

        iat_emb = self.iat_projection(iat_times.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # iat_emb = self.iat_norm(iat_emb)

        packet_number_emb = self.packet_number_projection(packet_numbers.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # packet_number_emb = self.packet_number_norm(packet_number_emb)

        avg_payload_emb = self.avg_payload_projection(avg_payload_sizes.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # avg_payload_emb = self.avg_payload_norm(avg_payload_emb)

        duration_emb = self.duration_projection(durations.unsqueeze(-1))  # [batch_size, seq_len, emb_dim]
        # duration_emb = self.duration_norm(duration_emb)

        # 2️⃣ 内容融合
        combined_emb = torch.cat(
            [directions_emb, payload_emb, iat_emb, packet_number_emb, avg_payload_emb, duration_emb],
            dim=-1
        )
        combined_emb = self.feature_fusion_projection(combined_emb)        

        # 3️⃣ 加位置编码
        positions = torch.arange(seq_len, device=directions.device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.position_embedding(positions)  # [batch_size, seq_len, emb_dim]
        
        # 组合token语义编码和位置编码，通过逐元素相加（element-wise sum）
        combined_emb = combined_emb + pos_emb
        
        # -------- 把 mask 用在 Transformer 和 AttentionPooling --------        
        # combined_emb = self.combined_norm(combined_emb)
        # Transformer编码 - 使用正确的掩码格式
        # src_key_padding_mask: True表示需要被mask的位置
        sequence_output = self.transformer(
            combined_emb,
            src_key_padding_mask=~sequence_mask.bool() if sequence_mask is not None else None  # src_key_padding_mask: True表示需要被mask的位置
        )
        
        # 注意力池化得到序列表示，输出 pooled vector
        # query：序列的全局表示（均值或 learnable CLS）        
        query = torch.mean(sequence_output, dim=1, keepdim=True)  # [batch_size, 1, emb_dim]
        # 关键点：query 只有 1 个 token，key/value 有 L 个 token        
        attn_output, attn_weights = self.attention_pooling(
            query, sequence_output, sequence_output, 
            key_padding_mask=~sequence_mask.bool() if sequence_mask is not None else None   # 取反，因为Transformer需要True表示mask
        )
        pooled_sequence_embedding = attn_output.squeeze(1)  # [B,H] = [batch_size, emb_dim]

        return {
            "sequence_embedding": pooled_sequence_embedding,
        }

class MultiViewFusionFactory:
    """多视图融合策略工厂"""
    
    @staticmethod
    def create_fusion_layer(cfg: DictConfig, hidden_size: int, num_views: int):
        fusion_method = cfg.model.multiview.fusion.method
        
        if fusion_method == "cross_attention":
            return CrossAttentionFusion(
                hidden_size=hidden_size,
                num_heads=cfg.model.multiview.fusion.cross_attention_heads,
                dropout=cfg.model.multiview.fusion.cross_attention_dropout,
                num_views=num_views
            )
        elif fusion_method == "weighted_sum":
            return WeightedSumFusion(
                hidden_size=hidden_size,
                num_views=num_views,
                learnable_weights=cfg.model.multiview.fusion.weighted_sum.learnable_weights,
                initial_weights=cfg.model.multiview.fusion.weighted_sum.initial_weights
            )
        elif fusion_method == "concat":
            return ConcatFusion(
                hidden_size=hidden_size,
                num_views=num_views,
            )
        else:
            raise ValueError(f"不支持的融合方法: {fusion_method}")

class WeightedSumFusion(nn.Module):
    """加权求和多视图融合"""
    
    def __init__(
        self, 
        hidden_size: int,
        num_views: int,
        learnable_weights: bool = True,
        initial_weights: List[float] = None
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_views = num_views
        
        # 视图权重
        if learnable_weights:
            if initial_weights is not None and len(initial_weights) == num_views:
                self.view_weights = nn.Parameter(torch.tensor(initial_weights, dtype=torch.float32))
            else:
                self.view_weights = nn.Parameter(torch.ones(num_views) / num_views)
        else:
            if initial_weights is not None and len(initial_weights) == num_views:
                self.register_buffer('view_weights', torch.tensor(initial_weights, dtype=torch.float32))
            else:
                self.register_buffer('view_weights', torch.ones(num_views) / num_views)
        
        # 可选的特征变换层
        self.view_projections = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_views)
        ])
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        # 对每个视图进行投影变换
        projected_views = []
        for i, view in enumerate(view_embeddings):
            projected_view = self.view_projections[i](view)
            projected_views.append(projected_view)
        
        # 加权求和
        weights = F.softmax(self.view_weights, dim=0)
        fused = sum(weight * view for weight, view in zip(weights, projected_views))
        
        # 层归一化
        return self.layer_norm(fused)

class ConcatFusion(nn.Module):
    """简化的拼接多视图融合"""
    
    def __init__(
        self, 
        hidden_size: int,
        num_views: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_views = num_views
        
        # 拼接后的总维度 = hidden_size * num_views
        concat_dim = hidden_size * num_views
        
        self.projection = nn.Sequential(
            nn.Linear(concat_dim, hidden_size),  # 直接投影到 hidden_size
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(hidden_size)
        )

        logger.info(f"[ConcatFusion] 初始化：输入维度 = {concat_dim}, 输出维度 = {hidden_size}")
        
    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        # 拼接所有视图特征
        concatenated = torch.cat(view_embeddings, dim=1)

        # ---- 打印拼接后的维度 ----
        # logger.info(f"[ConcatFusion] 拼接后 concatenated.shape = {tuple(concatenated.shape)}")
                
        # 直接投影到目标维度
        fused = self.projection(concatenated)

        # ---- 打印投影输出维度 ----
        # logger.info(f"[ConcatFusion] 投影后 fused.shape = {tuple(fused.shape)}")
        
        return fused
    
class CrossAttentionFusion(nn.Module):
    """
    多视图交叉注意力融合层，支持获取注意力权重。
    与 MultiViewFusionFactory.create_fusion_layer 完全兼容。
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout: float = 0.1,
        num_views: int = 3,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_views = num_views
        self.dropout = dropout

        if hidden_size % num_heads != 0:
            raise ValueError(f"hidden_size={hidden_size} 必须能被 num_heads={num_heads} 整除")

        self.head_dim = hidden_size // num_heads

        # === 为每个视图构建 Q/K/V 映射 ===
        self.query_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])
        self.key_proj   = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])
        self.value_proj = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(num_views)])

        # === 为每个视图的 查询视图 构建一个交叉注意力模块 ===
        self.cross_attn = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_size,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            for _ in range(num_views)
        ])

        # === 输出融合（拼接所有 attended view） ===
        self.output_linear = nn.Linear(hidden_size * num_views, hidden_size)
        self.dropout_layer = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_size)

        self._init_weights()

    def _init_weights(self):
        for proj_list in [self.query_proj, self.key_proj, self.value_proj]:
            for proj in proj_list:
                nn.init.xavier_uniform_(proj.weight)
                nn.init.constant_(proj.bias, 0)
        nn.init.xavier_uniform_(self.output_linear.weight)
        nn.init.constant_(self.output_linear.bias, 0)

    def forward(self, view_embeddings: List[torch.Tensor]) -> torch.Tensor:
        """
        view_embeddings: List of [B, hidden_size]
        return: [B, hidden_size]
        """

        # 过滤掉 None（例如 disabled view）
        valid_views = [v for v in view_embeddings if v is not None]
        actual_num_views = len(valid_views)

        if actual_num_views == 0:
            raise ValueError("没有可用的视图输入！")

        batch_size = valid_views[0].size(0)

        attended_outputs = []

        for i in range(actual_num_views):
            # Query 为本视图
            q = self.query_proj[i](valid_views[i]).unsqueeze(1)

            # 其他视图作为 Key/Value
            other_indices = [j for j in range(actual_num_views) if j != i]

            k = torch.stack([self.key_proj[j](valid_views[j]) for j in other_indices], dim=1)
            v = torch.stack([self.value_proj[j](valid_views[j]) for j in other_indices], dim=1)

            out, _ = self.cross_attn[i](q, k, v, need_weights=False)
            attended_outputs.append(out.squeeze(1))

        # 拼接
        fused = torch.cat(attended_outputs, dim=1)

        # 投影到 hidden_size
        projected = self.output_linear(fused)

        # 残差（使用第一个视图）
        output = self.norm(valid_views[0] + self.dropout_layer(projected))

        return output

    def get_attention_weights(self, view_embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        返回每个视图作为 Query 时的 attention weight
        """
        valid_views = [v for v in view_embeddings if v is not None]
        actual_num_views = len(valid_views)

        weights = {}

        for i in range(actual_num_views):
            q = self.query_proj[i](valid_views[i]).unsqueeze(1)
            other_indices = [j for j in range(actual_num_views) if j != i]

            k = torch.stack([self.key_proj[j](valid_views[j]) for j in other_indices], dim=1)
            v = torch.stack([self.value_proj[j](valid_views[j]) for j in other_indices], dim=1)

            _, attn = self.cross_attn[i](q, k, v, need_weights=True)

            weights[f"view_{i}_attn"] = attn  # [B, 1, num_other]

        return weights
    

def get_project_root(start_path: str | None = None) -> str:
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

class FlowBertMultiview(pl.LightningModule):
    """多视图BERT模型"""
    
    def __init__(self, cfg: DictConfig, dataset):
        
        # 显式初始化两个父类，避免MRO问题 ， 2025-12-02 del by qinyf
        # pl.LightningModule.__init__(self)
        # SHAPAnalyzeMixin.__init__(self, cfg)

        super().__init__()
        
        # 保存 cfg，但忽略 dataset（不可序列化）
        self.save_hyperparameters(
            "cfg",
            logger=False,
            ignore=["dataset"]
        )
        self.cfg = cfg
        self.labels_cfg = cfg.datasets.labels

        # 1. 从 dataset 获取类别型特征的映射和有效列
        if dataset is None:
            raise ValueError("dataset must be provided when initializing FlowBertMultiview "
                            "because categorical embeddings depend on dataset statistics.")
        
        self.categorical_val2idx_mappings = dataset.categorical_val2idx_mappings
        assert self.categorical_val2idx_mappings is not None, \
                "Model loaded without dataset stats — categorical val2idx embeddings invalid!"

        self.categorical_columns_effective = dataset.categorical_columns_effective
        assert self.categorical_columns_effective is not None, \
                "Model loaded without dataset stats — categorical columns effective invalid!"

        # 2. 初始化 SHAP 组件 (放在 __init__ 最后)  2025-12-02 added by qinyf
        if self.cfg.shap.enable_shap:
            self.shap_analyzer = ShapAnalyzer(self)

        self.debug_mode = cfg.debug.debug_mode
        # 🔴 根据debug_mode 设置 nan_check_enabled 属性
        self.nan_check_enabled = getattr(cfg.debug, 'nan_check_enabled', self.debug_mode)

        # 初始化BERT模型和配置
        _require_transformers()
        self.bert, self.bert_config, self.tokenizer = self._load_bert_model(cfg)
        logger.info(f"加载的BERT模型的每个 token 的隐藏向量维度（hidden dimension）：bert_config.hidden_size = {self.bert_config.hidden_size}")

        # === 概念漂移管理器初始化 ===
        if DRIFT_MANAGER_AVAILABLE and hasattr(cfg, 'concept_drift') and cfg.concept_drift.enabled:
            logger.info(f"🎯 概念漂移检测功能已启用 (Algorithm: {cfg.concept_drift.get('algorithm', 'bndm')})")
            self.drift_enabled = True
            self.drift_manager = None  # 将在 on_train_start 中初始化
        else:
            if hasattr(cfg, 'concept_drift') and not cfg.concept_drift.enabled:
                logger.info("ℹ️ 概念漂移检测功能已在配置中禁用")
            else:
                logger.info("ℹ️ 未检测到概念漂移配置或模块不可用，功能禁用")
            self.drift_enabled = False
            self.drift_manager = None

        # 检查各视图是否启用
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

        logger.info(f"视图启用状态: 数值特征向量必选，数据包序列={self.sequence_features_enabled}, 文本={self.text_features_enabled}, 域名嵌入={self.domain_embedding_enabled}")
        # 安全检查
        if self.text_features_enabled and not hasattr(self, 'bert_config'):
            raise ValueError("文本特征已启用但BERT配置未初始化")

        # 初始化所有投影层
        self._init_projection_layers(cfg)

        # 计算实际启用的视图数量
        self.num_views = 1  # 表格数据特征：数值特征（必选） + 域名嵌入特征（可选）
        if self.text_features_enabled:
            self.num_views += 1 # 文本视图可选
        if self.sequence_features_enabled:
            self.num_views += 1 # 数据包序列视图可选
        
        logger.info(f"模型使用的视图数量: {self.num_views}")
        logger.info(f"多视图融合方法: {cfg.model.multiview.fusion.method}")
        
        # 初始化多视图融合层
        self.fusion_layer = MultiViewFusionFactory.create_fusion_layer(
            cfg=cfg,
            hidden_size=self.bert_config.hidden_size,
            num_views=self.num_views
        )

        # 初始化分类器
        self._init_classifier(cfg)

        # 初始化分类损失函数
        self._init_loss_function(cfg)

        # SHAP分析现在由SHAPAnalyzeMixin统一管理，无需重复配置

        # =========================================================================
        # 🔧 [新增] 解决概念漂移导致的 Embedding 维度不匹配问题
        # =========================================================================
        def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
            """
            在加载权重前触发。
            如果在测试/漂移检测阶段发现了新类别（导致 Embedding 层变大），
            这里会自动将 Checkpoint 中的旧权重填充到新模型中，新类别对应的权重保持随机初始化。
            """
            state_dict = checkpoint["state_dict"]
            model_state_dict = self.state_dict()

            for key in list(state_dict.keys()):
                # 仅处理 Embedding 层的维度不匹配问题
                if "categorical_embedding_layers" in key and key in model_state_dict:
                    ckpt_tensor = state_dict[key]
                    model_tensor = model_state_dict[key]

                    # 如果维度不一致
                    if ckpt_tensor.shape != model_tensor.shape:
                        logger.warning(
                            f"⚠️ 检测到 Embedding 维度漂移 {key}: 存档 {ckpt_tensor.shape} vs 当前 {model_tensor.shape}")

                        # 确保是类别数量（行数）发生了变化，且 Hidden Size（列数）一致
                        if ckpt_tensor.shape[1] == model_tensor.shape[1]:
                            old_rows = ckpt_tensor.shape[0]
                            new_rows = model_tensor.shape[0]

                            if new_rows > old_rows:
                                # 【情况1：新数据出现了新类别】
                                # 创建一个与当前模型形状一致的 Tensor (包含新类别的随机初始值)
                                # 注意：这里直接使用 model_tensor.clone() 会复制当前的随机初始化值
                                new_ckpt_tensor = model_tensor.clone()

                                # 将旧权重覆盖回去 (保留训练好的知识)
                                new_ckpt_tensor[:old_rows] = ckpt_tensor

                                # 更新 checkpoint 字典，骗过 strict 加载检查
                                state_dict[key] = new_ckpt_tensor
                                logger.info(
                                    f"✅ 自动扩展权重 {key}: 保留前 {old_rows} 行旧权重，新增 {new_rows - old_rows} 行随机权重用于增量学习。")

                            elif new_rows < old_rows:
                                # 【情况2：当前数据类别少于训练集】(较少见，但为了健壮性处理)
                                # 直接截取旧权重的前 N 行
                                state_dict[key] = ckpt_tensor[:new_rows]
                                logger.info(f"⚠️ 自动裁剪权重 {key}: {old_rows} -> {new_rows}")

    # =========================================================================
    # 🔧 [新增] 解决概念漂移导致的 Embedding 维度不匹配问题
    # =========================================================================
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """
        在加载权重前触发。
        如果在测试/漂移检测阶段发现了新类别（导致 Embedding 层变大），
        这里会自动将 Checkpoint 中的旧权重填充到新模型中，新类别对应的权重保持随机初始化。
        """
        state_dict = checkpoint["state_dict"]
        model_state_dict = self.state_dict()

        for key in list(state_dict.keys()):
            # 仅处理 Embedding 层的维度不匹配问题
            if "categorical_embedding_layers" in key and key in model_state_dict:
                ckpt_tensor = state_dict[key]
                model_tensor = model_state_dict[key]

                # 如果维度不一致
                if ckpt_tensor.shape != model_tensor.shape:
                    logger.warning(
                        f"⚠️ 检测到 Embedding 维度漂移 {key}: 存档 {ckpt_tensor.shape} vs 当前 {model_tensor.shape}")

                    # 确保是类别数量（行数）发生了变化，且 Hidden Size（列数）一致
                    if ckpt_tensor.shape[1] == model_tensor.shape[1]:
                        old_rows = ckpt_tensor.shape[0]
                        new_rows = model_tensor.shape[0]

                        if new_rows > old_rows:
                            # 【情况1：新数据出现了新类别】
                            # 创建一个与当前模型形状一致的 Tensor (包含新类别的随机初始值)
                            # 注意：这里直接使用 model_tensor.clone() 会复制当前的随机初始化值
                            new_ckpt_tensor = model_tensor.clone()

                            # 将旧权重覆盖回去 (保留训练好的知识)
                            new_ckpt_tensor[:old_rows] = ckpt_tensor

                            # 更新 checkpoint 字典，骗过 strict 加载检查
                            state_dict[key] = new_ckpt_tensor
                            logger.info(
                                f"✅ 自动扩展权重 {key}: 保留前 {old_rows} 行旧权重，新增 {new_rows - old_rows} 行随机权重用于增量学习。")

                        elif new_rows < old_rows:
                            # 【情况2：当前数据类别少于训练集】(较少见，但为了健壮性处理)
                            # 直接截取旧权重的前 N 行
                            state_dict[key] = ckpt_tensor[:new_rows]
                            logger.info(f"⚠️ 自动裁剪权重 {key}: {old_rows} -> {new_rows}")


    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, **kwargs):
        # 强制传递 dataset
        return super().load_from_checkpoint(checkpoint_path, **kwargs)

    def on_validation_epoch_end(self):
        """验证阶段结束时检查概念漂移"""
        super().on_validation_epoch_end()

        # 定期进行概念漂移检测和状态报告
        if self.drift_detector is not None:
            # 每个epoch都报告一次状态
            stats = self.drift_detector.get_statistics()

            logger.info(f"📈 概念漂移检测状态 (Epoch {self.current_epoch}):")
            logger.info(f"  总样本数: {stats['total_samples']}")
            logger.info(f"  检测到漂移次数: {stats['drift_count']}")
            logger.info(f"  当前状态: {stats['status']}")
            logger.info(f"  窗口状态: W={stats['window_W_size']}, R={stats['window_R_size']}")

            # 如果检测到漂移，记录详细信息
            if self.drift_detected:
                is_drift, B, info = self.drift_detector.detect_drift()
                if is_drift:
                    logger.warning(f"🔴 当前存在概念漂移!")
                    logger.warning(f"   贝叶斯因子B: {B:.6f}")
                    logger.warning(f"   比较: {info['comparison']}")

                self.drift_detected = False  # 重置检测标志

    def freeze_backbone(self):
        for param in self.flow_bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.flow_bert.parameters():
            param.requires_grad = True

    def _load_bert_model(self, cfg: DictConfig) -> tuple:
        """
        加载BERT模型、配置和tokenizer
        
        Args:
            cfg: 配置对象
            
        Returns:
            tuple: (bert_model, bert_config, tokenizer)
        """
        # 添加参数验证
        if not hasattr(cfg.model.bert, 'model_name') or cfg.model.bert.model_name is None:
            raise ValueError("BERT模型名称未在配置中设置。请检查配置文件中的 bert.model_name 字段。")
        
        logger.info(f"使用BERT模型: {cfg.model.bert.model_name}")
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(current_dir, "../../../../"))
        model_path = os.path.join(project_root, 'models_hub', cfg.model.bert.model_name)
        
        try:
            # 首先尝试从本地缓存加载
            logger.info("尝试从本地缓存加载BERT模型...")
            bert_config = BertConfig.from_pretrained(
                model_path,
                hidden_dropout_prob=cfg.model.bert.hidden_dropout_prob,
                attention_probs_dropout_prob=cfg.model.bert.attention_probs_dropout_prob)
            bert_model = BertModel.from_pretrained(model_path)
            tokenizer = BertTokenizer.from_pretrained(model_path)
            
            # 🔴 确保BERT模型设置为训练模式
            bert_model.train()
            logger.info(f"成功从本地缓存加载BERT模型，设置为训练模式")
            
        except (OSError, ValueError) as e:
            logger.warning(f"本地模型未找到: {e}, 尝试在线下载...")
            logger.warning(f"如果网络不可用，请手动下载模型并放置在本地路径: {model_path}")
            # 如果本地没有，下载并保存
            bert_config = BertConfig.from_pretrained(
                cfg.model.bert.model_name,
                hidden_dropout_prob=cfg.model.bert.hidden_dropout_prob,
                attention_probs_dropout_prob=cfg.model.bert.attention_probs_dropout_prob)
            bert_model = BertModel.from_pretrained(cfg.model.bert.model_name)
            tokenizer = BertTokenizer.from_pretrained(cfg.model.bert.model_name)
            
            # 🔴确保BERT模型设置为训练模式
            bert_model.train()

            # 保存到本地
            bert_config.save_pretrained(model_path)
            bert_model.save_pretrained(model_path)
            tokenizer.save_pretrained(model_path)
            logger.info("BERT模型在线下载并保存到本地完成，设置为训练模式")
        
        return bert_model, bert_config, tokenizer

    def _init_projection_layers(self, cfg: DictConfig):
        # 🔴 1. 初始化文本特征投影层（可选）
        if self.text_features_enabled:
            logger.info("初始化文本特征编码器")
        else:
            logger.info("跳过文本特征编码器初始化")

        # 🔴 2. 初始化数据包特征投影层（可选）
        if self.sequence_features_enabled:
            # 当前的两层设计（嵌入层 + 投影层）
            self.sequence_encoder = SequenceEncoder(
                embedding_dim=cfg.model.sequence.embedding_dim, # embedding_dim可以独立于 BERT 的隐藏层大小进行调优
                num_layers=cfg.model.sequence.num_layers,
                num_heads=cfg.model.sequence.num_heads,
                dropout=cfg.model.sequence.dropout, 
                max_seq_length=cfg.data.max_seq_length
            )
            # sequence_projection 是最轻量的跨模态 Adapter
            self.sequence_projection = nn.Linear(cfg.model.sequence.embedding_dim, self.bert_config.hidden_size)
            logger.info("初始化数据包序列编码器")
        else:
            self.sequence_encoder = None
            self.sequence_projection = None
            logger.info("跳过数据包序列编码器初始化")

        # 🔴 3. 初始化域名嵌入特征维度（可选），不考虑
        if self.domain_embedding_enabled:
            label_id_map = ConfigManager.read_session_label_id_map(self.cfg.data.dataset)
            self.prob_list_length = len(label_id_map)
            logger.info(f"域名嵌入特征概率列表长度: {self.prob_list_length}")
            self.domain_feature_dim = len(cfg.data.domain_name_embedding_features.column_list) * self.prob_list_length
            logger.info(f"域名嵌入特征长度: {self.domain_feature_dim}")
        else:
            self.domain_feature_dim = 0
            logger.info(f"域名嵌入特征长度: {self.domain_feature_dim}")

        # 4. 初始化类别型特征嵌入层（必选）
        # 类别型特征始终启用，因为所有流都有 conn.proto、service 等类别语义
        self.categorical_embedding_layers = nn.ModuleDict()

        # ⭐ 从 dataset 补充 category → index 映射
        for col, mapping in self.categorical_val2idx_mappings.items():
            # num_classes = (最大 index) + 1
            # 因为 index 范围为 [0 ... K]，共 K+1 个 embedding 向量
            # 其中 index=0 保留给 OOV （Out-Of-Vocabulary）
            num_classes = max(mapping.values()) + 1  
            self.categorical_embedding_layers[col] = nn.Embedding(
                num_embeddings=num_classes,
                embedding_dim=self.bert_config.hidden_size
            )
            logger.info(f"初始化 categorical embedding: {col} → {num_classes} 类别")
        
        # 🔹 初始化 categorical LayerNorm（拼接后做归一化）
        self.categorical_norm = nn.LayerNorm(
            normalized_shape=self.bert_config.hidden_size * len(self.categorical_columns_effective)
        )
        
        # 5. 初始化数值型特征（必选）+域名嵌入特征+类别型特征（必选）的表格数据投影层
        # Delay to forward 时计算 tabular_feature_dim
        self.numeric_feature_dim = (
            len(cfg.data.tabular_features.numeric_features.flow_features)
            + len(cfg.data.tabular_features.numeric_features.x509_features)
            + len(cfg.data.tabular_features.numeric_features.dns_features)
        )
        logger.info(f"数值型流特征数: {self.numeric_feature_dim}")
        self.categorical_feature_dim = len(self.categorical_columns_effective) * self.bert_config.hidden_size
        logger.info(f"总类别型特征数: {len(self.categorical_columns_effective)}")
        logger.info(f"总类别型特征维度: {self.bert_config.hidden_size} * {len(self.categorical_columns_effective)} = {self.bert_config.hidden_size * len(self.categorical_columns_effective)}")
        self.tabular_feature_dim = self.numeric_feature_dim + self.domain_feature_dim + self.categorical_feature_dim
        logger.info(f"表格数据总特征维度: 数值特征({self.numeric_feature_dim}) + 域名嵌入({self.domain_feature_dim}) + 类别型特征({self.categorical_feature_dim}) = {self.tabular_feature_dim}")

        self.tabular_projection = nn.Linear(
            self.tabular_feature_dim,
            self.bert_config.hidden_size
        )
        logger.info(f"初始化表格特征线性投影层: 输入维度={self.tabular_feature_dim}, 输出维度={self.bert_config.hidden_size}")

    def _init_classifier(self, cfg: DictConfig):
        """初始化分类器 - 根据融合方法调整输入维度"""
        self.classifier_input_dim = self._get_classifier_input_dim(cfg)
        self._init_is_malicious_classifier(cfg)
        self._init_attack_family_classifier(cfg)

    def _get_classifier_input_dim(self, cfg: DictConfig) -> int:
        fusion_method = cfg.model.multiview.fusion.method

        if fusion_method == "concat":
            # 对于concat方法，分类器输入维度是所有启用视图的维度之和
            # classifier_input_dim = self.bert_config.hidden_size  # 数值特征投影后的维度（必选）
            
            # if self.sequence_features_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size
            # if self.text_features_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size
            # if self.domain_embedding_enabled:
            #     classifier_input_dim += self.bert_config.hidden_size

            # logger.info(f"视图启用状态: 数值特征向量必选，数据包序列={self.sequence_features_enabled}, 文本={self.text_features_enabled}, 域名嵌入={self.domain_embedding_enabled}")
            # logger.info(f"拼接融合总维度: {classifier_input_dim}")

            # 🔴 四个视图的concat，改成了ConcatFusion，其内部做了拼接解释特征的线性投影到 hidden_size
            classifier_input_dim = self.bert_config.hidden_size
            logger.warning(
                f"[Fusion Warning] 'concat' 融合原始特征维度 = bert_config.hidden_size * num_views "
                f"但 ConcatFusion 会自动投影回 bert_config.hidden_size，因此分类器输入维度固定为 bert_config.hidden_size = {self.bert_config.hidden_size} "
                f"(num_views={self.num_views})"
            )
            return classifier_input_dim
        else:
            # 其他方法都输出 hidden_size 维度
            classifier_input_dim = self.bert_config.hidden_size
            logger.info(f"[Fusion Info] 使用 {fusion_method} 融合，输出维度 = bert_config.hidden_size = {self.bert_config.hidden_size}")
            return classifier_input_dim

    def _init_is_malicious_classifier(self, cfg: DictConfig):
        is_malicious_classifier_cfg = cfg.datasets.tasks.outputs.is_malicious.classifier

        logger.info(f"is_malicious善意/恶意流量分类器的输入维度: {self.classifier_input_dim} ，"
                    f"hidden_dims={is_malicious_classifier_cfg.hidden_dims}")


        # 添加 is_malicious（二分类，善意 / 恶意）的 flow-level 分类器隐藏层
        is_malicious_classifier_layers = []
        current_dim = self.classifier_input_dim
        for i, hidden_dim in enumerate(is_malicious_classifier_cfg.hidden_dims):
            logger.info(f"is_malicious善意/恶意分类器隐藏层 {i+1}: {current_dim} -> {hidden_dim}")
            
            is_malicious_classifier_layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.GELU() if is_malicious_classifier_cfg.activation == "gelu" else nn.ReLU(),
                nn.Dropout(is_malicious_classifier_cfg.dropout),
                nn.LayerNorm(hidden_dim)
            ])
            current_dim = hidden_dim  # 更新当前维度
        
        # 添加is_malicious善意/恶意流量分类器的输出层
        # 输出层：1 维，对应 BCEWithLogitsLoss
        is_malicious_classifier_layers.append(nn.Linear(current_dim, 1))

        self.is_malicious_classifier = nn.Sequential(*is_malicious_classifier_layers)

        logger.info(
            f"is_malicious 分类器结构: "
            f"input_dim={self.classifier_input_dim} -> "
            f"hidden_dims={is_malicious_classifier_cfg.hidden_dims} -> "
            f"output_dim=1 (binary classification)"
        )


    def _init_attack_family_classifier(self, cfg: DictConfig):
        """
        初始化 attack_family 分类器（OVR 多分类）

        attack_family 采用 One-vs-Rest (OVR) 形式：
        - 每个攻击家族对应一个二分类器
        - 输出维度 = 攻击家族数量
        - 配合 BCEWithLogitsLoss + multi-hot 标签使用
        """

        # ---------- 1. 配置检查 ----------
        if "attack_family" not in cfg.datasets.tasks.outputs:
            logger.info("未配置 attack_family 任务，跳过 attack_family 分类器初始化")
            self.attack_family_classifier = None
            return

        attack_family_cfg = cfg.datasets.tasks.outputs.attack_family
        assert attack_family_cfg.strategy == "ovr", \
            f"attack_family 目前仅支持 OVR 策略，当前配置为 {attack_family_cfg.strategy}"

        # ---------- 2. 解析攻击家族信息 ----------
        assert "attack_family" in self.labels_cfg, \
            "attack_family 已启用，但 labels_cfg.attack_family 未定义"

        # 1️⃣ 类别定义：唯一来源
        self.attack_family_classes = [
            c.strip() for c in self.labels_cfg.attack_family.classes
        ]

        # 2️⃣ 模型 / loss / logits 统一使用同一顺序
        self.attack_family_names = self.attack_family_classes
        num_fam = len(self.attack_family_names)

        logger.info(
            f"[attack_family] 启用 OVR 攻击家族分类任务，共 {num_fam} 个攻击家族: "
            f"{self.attack_family_names}"
        )

        # 3️⃣ 校验 class_weights 是否与 labels_cfg 一致（fail-fast）
        weight_keys = list(attack_family_cfg.class_weights.keys())
        if set(weight_keys) != set(self.attack_family_names):
            raise ValueError(
                "[attack_family] labels_cfg.attack_family.classes 与 "
                "task.outputs.attack_family.class_weights 键不一致!\n"
                f"labels_cfg: {self.attack_family_names}\n"
                f"class_weights: {weight_keys}"
            )

        # ---------- 3. 分类器输入维度 ----------
        input_dim = self.classifier_input_dim
        logger.info(
            f"[attack_family] 分类器输入维度 = {input_dim} "
            f"(来自多视图融合输出)"
        )

        # ---------- 4. 构建 OVR 分类器网络 ----------
        layers = []
        cur = input_dim

        for i, h in enumerate(attack_family_cfg.classifier.hidden_dims):
            logger.info(
                f"[attack_family] 分类器隐藏层 {i+1}: {cur} -> {h}"
            )
            layers.extend([
                nn.Linear(cur, h),
                nn.GELU(),
                nn.Dropout(attack_family_cfg.classifier.dropout),
                nn.LayerNorm(h),
            ])
            cur = h

        # ---------- 5. 输出层 ----------
        # 输出维度 = 攻击家族数量（OVR，每一维对应一个家族）
        layers.append(nn.Linear(cur, num_fam))

        self.attack_family_classifier = nn.Sequential(*layers)

        logger.info(
            f"[attack_family] OVR 分类器结构构建完成: "
            f"input_dim={input_dim} -> "
            f"hidden_dims={attack_family_cfg.classifier.hidden_dims} -> "
            f"output_dim={num_fam} (OVR, multi-label)"
        )


    def _init_loss_function(self, cfg: DictConfig):
        self._init_is_malicious_loss_function(cfg)
        self._init_attack_family_loss_function(cfg)

    def _init_is_malicious_loss_function(self, cfg: DictConfig):
        """
        初始化 is_malicious（二分类，善意/恶意）的损失函数
        使用 BCEWithLogitsLoss + 类别权重
        """
        # 初始化分类损失函数
        self.is_malicious_class_loss = nn.BCEWithLogitsLoss(reduction='none')

        # 🔹 默认权重（不加权）
        class_weights = [1.0, 1.0]

        # 🔹 从 task.outputs 中读取（如果配置了）
        try:
            task_cfg = cfg.datasets.tasks.outputs.get("is_malicious", None)
            if task_cfg is not None and "class_weights" in task_cfg:
                cw = task_cfg.class_weights
                if isinstance(cw, (list, tuple)) and len(cw) == 2:
                    class_weights = list(map(float, cw))
        except Exception as e:
            logger.warning(f"读取 task.outputs.is_malicious.class_weights 失败，使用默认权重: {e}")

        if class_weights is None or len(class_weights) != 2:
            logger.warning("[is_malicious] class_weights 非法，重置为 [1.0, 1.0]")
            # 使用默认权重 [1.0, 1.0]
            class_weights = [1.0, 1.0]

        self.is_malicious_class_weights = torch.tensor(class_weights, dtype=torch.float32)
        logger.info(
            f"[is_malicious] 损失函数初始化完成: "
            f"BCEWithLogitsLoss, class_weights={class_weights}"
        )

    def _init_attack_family_loss_function(self, cfg: DictConfig):
        """
        初始化 attack_family 的 OVR 多分类损失函数
        - 每个攻击家族一个 BCE loss
        - 使用 per-family [neg, pos] 权重
        """

        if not hasattr(self, "attack_family_classifier") or \
        self.attack_family_classifier is None:
            self.attack_family_loss_fn = None
            self.attack_family_class_weights = None
            logger.info("[attack_family] 未启用任务，跳过损失函数初始化")
            return

        attack_family_cfg = cfg.datasets.tasks.outputs.attack_family

        self.attack_family_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        # 构建权重矩阵 [num_families, 2]
        weights = []
        for fam in self.attack_family_names:
            w = attack_family_cfg.class_weights.get(fam, [1.0, 1.0])
            if len(w) != 2:
                logger.warning(
                    f"[attack_family] 家族 {fam} 的 class_weights 非法，使用默认 [1.0, 1.0]"
                )
                w = [1.0, 1.0]
            weights.append(w)

        self.attack_family_class_weights = torch.tensor(
            weights, dtype=torch.float32
        )

        logger.info(
            f"[attack_family] OVR 损失函数初始化完成: "
            f"BCEWithLogitsLoss, "
            f"class_weights shape={self.attack_family_class_weights.shape}"
        )

    def _compute_losses(self, outputs, batch, stage: str):
        """计算分类损失 + tabular特征重建损失 + 总损失"""

        # 初始化所有损失变量
        total_loss = torch.tensor(0.0, device=self.device)

        # ===== is_malicious 分类损失 =====
        is_malicious_cls_logits = outputs['is_malicious_cls_logits']
        # is_malicious_prob = torch.sigmoid(is_malicious_cls_logits)
        # is_malicious_pred = (is_malicious_prob > 0.5).float()
        is_malicious_label = batch['is_malicious_label']
        is_malicious_class_loss = self._compute_is_malicious_class_loss(is_malicious_cls_logits, is_malicious_label)

        # 检查is_malicious分类损失值
        if torch.isnan(is_malicious_class_loss) or torch.isinf(is_malicious_class_loss):
            logger.error(f"🚨 {stage}损失值为NaN或Inf: {is_malicious_class_loss}")
            # 尝试使用小损失值继续训练
            is_malicious_class_loss = torch.tensor(1.0, device=self.device, requires_grad=True)

        is_malicious_weight = getattr(self.cfg.datasets.tasks.outputs.is_malicious, "weight", 1.0)
        total_loss = total_loss + is_malicious_weight * is_malicious_class_loss

        # ===== attack_family 分类损失 =====
        attack_family_class_loss = None
        if self.attack_family_classifier is not None:
            attack_family_cls_logits = outputs["attack_family_cls_logits"]
            assert "attack_family_label" in batch, \
                f"attack_family_label not found in batch keys: {batch.keys()}"
            attack_family_label = batch["attack_family_label"]
            attack_family_class_loss = self._compute_attack_family_class_loss(attack_family_cls_logits, attack_family_label, is_malicious_label)
            # 检查attack_family分类损失值
            if torch.isnan(attack_family_class_loss) or torch.isinf(attack_family_class_loss):
                logger.error(f"🚨 {stage} attack_family_loss 为 NaN/Inf: {attack_family_class_loss}")
                attack_family_class_loss = torch.tensor(
                    1.0, device=self.device, requires_grad=True
                )
            attack_family_weight = getattr(self.cfg.datasets.tasks.outputs.attack_family, "weight", 1.0)
            total_loss = total_loss + attack_family_weight * attack_family_class_loss

        return {
            "total_loss": total_loss,
            "is_malicious_class_loss": is_malicious_class_loss,
            "attack_family_class_loss": attack_family_class_loss,
        }

    def _compute_is_malicious_class_loss(self, logits, labels):
        """计算分类损失"""
        # 确保标签形状正确 [batch_size, 1]
        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        # 基本损失计算
        base_loss = self.is_malicious_class_loss(logits, labels)

        # 确保 is_malicious_class_weights 已初始化
        if self.is_malicious_class_weights is not None:
            # 应用类别权重
            binary_weights = self.is_malicious_class_weights.to(logits.device) # ✅ 确保权重在正确的设备上
            binary_weights = torch.where(
                labels == 1,
                binary_weights[1],  # is_malicious=1 的权重
                binary_weights[0],  # is_malicious=0 的权重
            )
            weighted_loss = base_loss * binary_weights

            return weighted_loss.mean()
        else:
            return base_loss.mean()

    def _compute_attack_family_class_loss(self, logits, label, is_malicious_label):
        """
        计算 attack_family 的 OVR 多分类损失（条件损失）

        说明：
        - attack_family 仅在真实恶意样本（is_malicious_label == 1）上具有语义定义；
        - benign 样本在 attack_family 维度上属于 not applicable，
        不参与 attack_family 的损失计算与梯度反传；
        - 因此，该损失函数刻画的是条件分布下的分类性能：
            P(attack_family | is_malicious = 1)。

        参数说明：
        - logits: [B, K]
            attack_family 分类器的原始输出（每个攻击家族一个 OVR logit）
        - label: [B, K]（multi-hot）
            attack_family 的真实标签（OVR / multi-label 表示）
        - is_malicious_label: [B]（0/1）
            is_malicious 的真实标签，用于筛选具有 attack_family 语义的样本
        """
        assert logits.shape == label.shape, \
            f"attack_family logits/labels shape mismatch: {logits.shape} vs {label.shape}"

        # 仅保留真实恶意样本（ground truth），在该子集上计算 attack_family 损失
        # 注意：这里使用的是 is_malicious 的真实标签，而不是模型预测结果，
        # 以避免级联误差（cascaded error）对 family 学习的影响。
        mask = is_malicious_label.view(-1) == 1
        if mask.sum() == 0:
            # 当前 batch 中没有真实恶意样本：
            # attack_family 在该 batch 上不定义，返回 0 loss（不产生梯度）
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        logits = logits[mask]
        label = label[mask].float()  # BCE loss 要求 label 为 float

        # 基础 BCE 损失（OVR）：[B_malicious, K]
        base_loss = self.attack_family_loss_fn(logits, label)

        if self.attack_family_class_weights is not None:
            # class_weights 形状为 [K, 2]，表示每个攻击家族的 [neg, pos] 权重
            # 用于缓解不同 attack_family 之间的类别不平衡问题
            weights = self.attack_family_class_weights.to(logits.device)

            # 对每个 OVR 维度：
            # - label == 1 时使用正类权重（pos）
            # - label == 0 时使用负类权重（neg）
            # 通过 broadcasting 扩展到 [B_malicious, K]
            weighted = torch.where(
                label == 1,
                weights[:, 1],
                weights[:, 0]
            )

            loss = base_loss * weighted
            return loss.mean()
        else:
            return base_loss.mean()

    def _process_text_features(self, batch: Dict[str, Any]) -> torch.Tensor:
        """处理文本特征，返回[CLS] token嵌入"""
        if not self.text_features_enabled:
            # 返回零向量占位符
            batch_size = batch['numeric_features'].shape[0] # 利用必备的numeric_features维度
            return torch.zeros(batch_size, self.bert_config.hidden_size if self.bert_config else 512, 
                            device=self.device)

        combined_texts = batch['combined_text']
        
        encoding = self.tokenizer(
            combined_texts,
            padding=True,
            truncation=True,
            max_length=self.cfg.data.max_seq_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        text_outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        return text_outputs.last_hidden_state[:, 0]  # 返回[CLS] token嵌入

    # ---------------- 安全训练模式 ----------------
    def on_train_start(self) -> None:
        """确保所有子模块都在 train 模式"""
        super().on_train_start()

        # 设置所有模块为训练模式
        self.train()

        # 设置序列编码器训练模式（如果存在）
        if hasattr(self, 'sequence_encoder') and self.sequence_encoder is not None:
            self.sequence_encoder.train()
            if self.debug_mode:
                logger.info("序列编码器设置为训练模式")

        # 设置用于文本特征表征的BERT（如果存在），进入训练模式
        if self.bert is not None:
            self.bert.train()

        # 设置数值投影层训练模式（必选）
        if hasattr(self, 'tabular_projection') and self.tabular_projection is not None:
            self.tabular_projection.train()
            if self.debug_mode:
                logger.info("表格特征的投影层设置为训练模式")
        
        # 设置分类器训练模式
        if hasattr(self, 'is_malicious_classifier') and self.is_malicious_classifier is not None:
            self.is_malicious_classifier.train()

        if hasattr(self, 'attack_family_classifier') and self.attack_family_classifier is not None:
            self.attack_family_classifier.train()

        # 设置所有投影层为训练模式
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.LayerNorm, nn.Dropout)):
                module.train()

        # 额外检查：列出所有模块的训练状态
        if self.debug_mode:
            eval_modules = []
            for name, module in self.named_modules():
                if not module.training:
                    eval_modules.append(name)

            if eval_modules:
                logger.warning(f"以下模块仍在评估模式: {eval_modules}")
            else:
                logger.info("✅ 所有模块都已正确设置为训练模式")

        if self.drift_enabled and self.drift_manager is not None:
            logger.info("ℹ️ 训练阶段禁用概念漂移检测 (Manager will be ignored during training)")


    # ---------------- NaN 检查 ----------------
    def _safe_check_nan(self, x: torch.Tensor, name: str, operation: str = "") -> torch.Tensor:
        """安全的NaN检查，不会导致训练崩溃"""
        if not self.debug_mode:
            return x
            
        if torch.isnan(x).any():
            nan_count = torch.isnan(x).sum().item()
            logger.warning(f"⚠️ {name} 在 {operation} 后包含 {nan_count} 个NaN值")
            # 尝试修复：将NaN替换为0
            x = torch.where(torch.isnan(x), torch.zeros_like(x), x)
            
        if torch.isinf(x).any():
            inf_count = torch.isinf(x).sum().item()
            logger.warning(f"⚠️ {name} 在 {operation} 后包含 {inf_count} 个Inf值")
            # 尝试修复：将Inf替换为有限大值
            x = torch.where(torch.isinf(x), torch.finfo(x.dtype).max * torch.ones_like(x), x)
            
        return x
    
    def _debug_tensor(self, x: torch.Tensor, name: str):
        """调试张量信息"""
        if not self.nan_check_enabled:
            return

        stats = {
            "shape": x.shape,
            "min": x.min().item(),
            "max": x.max().item(), 
            "mean": x.mean().item() if x.dtype.is_floating_point else 0.0,
            "std": x.std(unbiased=False).item() if (x.dtype.is_floating_point and x.numel() >= 2) else 0.0,
            "nan_count": torch.isnan(x).sum().item(),
            "inf_count": torch.isinf(x).sum().item()
        }
        
        # 只有当有NaN或Inf时才输出警告
        if stats["nan_count"] > 0 or stats["inf_count"] > 0:
            msg = f"🔍🔍 {name}: " + " | ".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" 
                                            for k, v in stats.items()])
            logger.warning(msg)


    def _check_inputs_for_nan(self, batch: Dict[str, Any]):
        """更详细的输入数据检查"""
        if not self.debug_mode:
            return
            
        for key, value in batch.items():
            if torch.is_tensor(value):
                if torch.isnan(value).any():
                    nan_count = torch.isnan(value).sum().item()
                    logger.error(f"🚨🚨🚨🚨 严重错误: 输入数据 {key} 包含 {nan_count} 个NaN值")
                    raise ValueError(f"输入数据 {key} 包含NaN值，请检查数据预处理")
                    
                if torch.isinf(value).any():
                    inf_count = torch.isinf(value).sum().item()
                    logger.error(f"🚨🚨🚨🚨 严重错误: 输入数据 {key} 包含 {inf_count} 个Inf值")
                    raise ValueError(f"输入数据 {key} 包含Inf值，请检查数据预处理")
                    
                # 只对数值类型（浮点数、整数）的张量进行统计计算，跳过布尔类型
                if value.numel() > 0 and value.dtype in [torch.float16, torch.float32, torch.float64, torch.int16, torch.int32, torch.int64]:
                    try:
                        stats = {
                            'min': value.min().item(),
                            'max': value.max().item(),
                        }
                        
                        # 只有浮点类型才能计算mean和std
                        if value.dtype in [torch.float16, torch.float32, torch.float64]:
                            stats['mean'] = value.mean().item()
                            if value.numel() >= 2 and value.dtype.is_floating_point:
                                stats["std"] = value.std(unbiased=False).item()
                            else:
                                stats["std"] = 0.0
                        else:
                            # 对于整数类型，计算总和作为替代
                            stats['sum'] = value.sum().item()
                        
                        # logger.debug(f"输入数据 {key} 统计: {stats}")
                        
                    except Exception as e:
                        # 如果统计计算失败，只记录基本信息
                        logger.debug(f"无法计算 {key} 的统计信息: {e}, dtype: {value.dtype}")


    def _build_tabular_features(self, batch):
        """构建表格数据特征向量"""
        # 1. 数值型特征处理（必选）
        numeric_features = batch['numeric_features'].to(self.device)
        # print("DEBUG numeric_features shape:", numeric_features.shape)

        # 2. 类别型特征处理（必选）
        categorical_ids = batch["categorical_features"].to(self.device)   # [B, C]
        batch_size, num_cat_cols = categorical_ids.shape
        expected_num_cat_cols = len(self.categorical_columns_effective)

        # categorical_ids 形状应为 [B, num_effective_cols]
        assert num_cat_cols == expected_num_cat_cols, \
            f"⚠ categorical_features 列数不匹配：batch 中 {num_cat_cols}，dataset 中{expected_num_cat_cols}"
                
        categorical_embedded_list = []
        for i, cat_col in enumerate(self.categorical_columns_effective):
            cat_emb_layer = self.categorical_embedding_layers[cat_col]
            assert cat_emb_layer is not None, f"⚠ categorical_embedding_layer for column={cat_col} 找不到!"
            cat_col_ids = categorical_ids[:, i]       # [B]
            cat_col_emb = cat_emb_layer(cat_col_ids)  # [B, H]
            categorical_embedded_list.append(cat_col_emb)

        categorical_features = torch.cat(categorical_embedded_list, dim=1) # [B, C*H]
        self._debug_tensor(categorical_features, "categorical_features (before_norm)")
        # print("DEBUG categorical_features shape (before_norm):", categorical_features.shape)

        # ⭐ 规范化类别特征
        categorical_features = self.categorical_norm(categorical_features)
        self._debug_tensor(categorical_features, "categorical_features (after_norm)")
        # print("DEBUG categorical_features shape (after_norm):", categorical_features.shape)
        
        # 3. 域名嵌入特征处理（可选）
        if self.domain_embedding_enabled:
            domain_embeddings = batch['domain_embedding_features'].to(self.device)
            # print("DEBUG domain_embeddings shape:", domain_embeddings.shape)
        else:
            domain_embeddings = None

        # 4. 表格数据特征融合
        if self.domain_embedding_enabled:
            tabular_features = torch.cat([numeric_features, categorical_features, domain_embeddings], dim=1)
        else:
            tabular_features = torch.cat([numeric_features, categorical_features], dim=1)

        # print("DEBUG tabular_features shape:", tabular_features.shape)
        return tabular_features

    # ---------------- forward ----------------        
    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """前向传播"""
        
        # 检查输入数据是否有NaN/Inf
        self._check_inputs_for_nan(batch)        

        # 添加维度调试
        # logger.debug(f"数值特征维度: {batch['numeric_features'].shape}")
        # logger.debug(f"域名嵌入特征维度: {batch['domain_embedding_features'].shape}")
                
        # 1. 数据包序列特征处理（可选）
        if self.sequence_features_enabled and self.sequence_encoder is not None:
            sequence_data = {
                'directions': batch['directions'],
                'iat_times': batch['iat_times'],
                'payload_sizes': batch['payload_sizes'], 
                'packet_numbers': batch['packet_numbers'],
                'avg_payload_sizes': batch['avg_payload_sizes'],
                'durations': batch['durations'],
                'sequence_mask': batch['sequence_mask'],  # 有效token掩码
            }
            
            # 检查输入数据
            self._debug_tensor(sequence_data['directions'], "输入_directions")            
            self._debug_tensor(sequence_data['payload_sizes'], "输入_payload_sizes")
            self._debug_tensor(sequence_data['iat_times'], "输入_iat_times")
            self._debug_tensor(sequence_data['packet_numbers'], "输入_packet_numbers")
            self._debug_tensor(sequence_data['avg_payload_sizes'], "输入_avg_payload_sizes")
            self._debug_tensor(sequence_data['durations'], "输入_durations")
            self._debug_tensor(sequence_data['sequence_mask'], "输入_sequence_mask")

            seq_outputs = self.sequence_encoder(sequence_data)
            sequence_emb = seq_outputs["sequence_embedding"]
            self._debug_tensor(sequence_emb, "sequence_encoder输出")
            sequence_emb = self._safe_check_nan(sequence_emb, "sequence_emb", "序列编码")
            
            sequence_outputs = self.sequence_projection(sequence_emb)
            self._debug_tensor(sequence_outputs, "sequence_projection输出")
            sequence_outputs = self._safe_check_nan(sequence_outputs, "sequence_outputs", "序列投影")

        else:
            # 创建空的序列特征输出
            batch_size = batch['numeric_features'].shape[0]  # 借用必选的numeric_features，生成特征维度
            sequence_outputs = torch.zeros(batch_size, self.bert_config.hidden_size, device=self.device)

        # 2. 文本特征处理（可选）
        if self.text_features_enabled:
            text_outputs = self._process_text_features(batch)
            self._debug_tensor(text_outputs, "text_outputs")
            text_outputs = self._safe_check_nan(text_outputs, "text_outputs", "BERT处理")
        else:
            # 创建空的文本特征输出
            batch_size = batch['numeric_features'].shape[0]  # 借用必选的numeric_features，生成特征维度
            text_outputs = torch.zeros(batch_size, self.bert_config.hidden_size, device=self.device)

        # 3. 表格数据特征构建
        tabular_features = self._build_tabular_features(batch)
        self._debug_tensor(tabular_features, "输入_tabular_features")
        
        # 4. 表格数据特征投影
        tabular_outputs = self.tabular_projection(tabular_features)
        self._debug_tensor(tabular_outputs, "tabular_outputs输出")
        tabular_outputs = self._safe_check_nan(tabular_outputs, "tabular_outputs", "表格数据投影")

        # 5. 多视图特征融合：数据包序列特征+文本特征+表格数据特征
        multiview_outputs = self._fuse_multi_views(sequence_outputs, text_outputs, tabular_outputs)
        self._debug_tensor(multiview_outputs, "融合后_multiview_outputs")
        multiview_outputs = self._safe_check_nan(multiview_outputs, "multiview_outputs", "多视图融合")
        
        # 6. 分类器
        is_malicious_cls_logits = self.is_malicious_classifier(multiview_outputs)
        self._debug_tensor(is_malicious_cls_logits, "is_malicious_classifier输出")
        is_malicious_cls_logits = self._safe_check_nan(is_malicious_cls_logits, "is_malicious_cls_logits", "is_malicious_分类器")

        attack_family_cls_logits = None
        if self.attack_family_classifier is not None:
            attack_family_cls_logits = self.attack_family_classifier(multiview_outputs)
            self._debug_tensor(attack_family_cls_logits, "attack_family_cls_logits输出")
            attack_family_cls_logits = self._safe_check_nan(attack_family_cls_logits, "attack_family_cls_logits", "is_malicious_分类器")

        return {
            'sequence_embeddings': sequence_outputs,
            'text_embeddings': text_outputs if self.text_features_enabled else None,
            'tabular_embeddings': tabular_outputs,
            'multiview_embeddings': multiview_outputs,
            'is_malicious_cls_logits': is_malicious_cls_logits,
            'attack_family_cls_logits': attack_family_cls_logits
        }

    def _on_drift_detected(self, drift_info: Dict):
        """
        检测到概念漂移时的响应策略 (自定义优化版)
        """
        import math  # 确保导入math

        logger.info("\n" + "!" * 60)
        logger.info(f"🚨 概念漂移检测触发 (第 {self.drift_count} 次)")
        logger.info("!" * 60)

        # 1. 基础信息 (移除 Log B，仅保留 B 值)
        bayes_factor = drift_info.get('bayes_factor', 0.0)
        threshold = self.drift_detector.detection_threshold if self.drift_detector else 0

        # 获取配置的漂移类型
        drift_type_config = getattr(self.drift_detector, 'drift_type', 'sudden')

        logger.info(f"📊 [指标分析]")
        # 将科学计数法改为小数形式
        logger.info(f"   当前 B 值 : {bayes_factor:.10f} (阈值 τ = {threshold})")

        # 2. 历史趋势
        if 'history' in drift_info and drift_info['history']:
            history = drift_info['history']
            # 将趋势中的值也改为小数形式
            history_str = " -> ".join([f"{b:.8f}" for b in history[-5:]])
            logger.info(f"   B值变化趋势 (近5次): {history_str}")

            # 3. 漂移判定 (替代突变比率)
            # 根据 B 值大小判断漂移证据的强弱
            evidence_str = ""
            if bayes_factor < 1e-10:
                evidence_str = "极强 (Extreme)"
            elif bayes_factor < 1e-4:
                evidence_str = "强 (Strong)"
            else:
                evidence_str = "中等 (Moderate)"

            logger.info(f"   漂移类型判定: {drift_type_config} drift (证据强度: {evidence_str})")

        # 4. 漂移诊断 (优化层级显示)
        if 'diagnosis' in drift_info and drift_info['diagnosis']:
            logger.info(f"🔍 [窗口分布差异诊断 (Top-3 差异节点)]")
            logger.info(f"   说明: 'Ref'为参考窗口分布，'Cur'为当前窗口分布")

            for i, node_info in enumerate(drift_info['diagnosis']):
                level = node_info['level']
                code = node_info['code']
                log_b = node_info['log_B_s']
                # 将节点的 log contribution 转为 B 值 contribution
                node_b_val = math.exp(log_b) if log_b > -700 else 0.0

                ref_cnt = node_info['ref_counts']  # (L, R)
                cur_cnt = node_info['cur_counts']  # (L, R)

                # 构建层级路径字符串：Root -> 0 -> 01
                path_visual = "Root"
                if code and code != "ROOT":
                    current_path = ""
                    steps = []
                    for char in code:
                        current_path += char
                        steps.append(current_path)
                    # 只显示最后几层以保持简洁，或者显示全路径
                    path_visual = " -> ".join(["Root"] + steps)

                # 计算简单的比例以辅助观察
                ref_total = sum(ref_cnt) + 1e-9
                cur_total = sum(cur_cnt) + 1e-9
                ref_ratio = f"{ref_cnt[0] / ref_total:.1%} vs {ref_cnt[1] / ref_total:.1%}"
                cur_ratio = f"{cur_cnt[0] / cur_total:.1%} vs {cur_cnt[1] / cur_total:.1%}"

                # 将节点贡献的 B 值也改为小数形式
                logger.info(f"   📍 节点层级 {level} | 路径: [{path_visual}] | 贡献 B值: {node_b_val:.8f}")
                logger.info(
                    f"      参考 R (N={int(ref_total)}): L {ref_cnt[0]} - R {ref_cnt[1]} ({ref_ratio})")
                logger.info(
                    f"      当前 W (N={int(cur_total)}): L {cur_cnt[0]} - R {cur_cnt[1]} ({cur_ratio})")

        # 5. 响应策略
        logger.info(f"🛡️ [响应策略 - 观察模式]")
        # if hasattr(self, 'trainer') and self.trainer is not None:
        #     opt = self.trainer.optimizers[0]
        #     current_lr = opt.param_groups[0]['lr']
        #
        #     # 🟢 修改点：设置最小学习率 (例如 1e-6)
        #     min_lr = 1e-6
        #     # 🟢 修改点：衰减系数改为 0.8 (比 0.5 温和)
        #     decay_factor = 0.8
        #
        #     new_lr = max(current_lr * decay_factor, min_lr)
        #
        #     if new_lr < current_lr:
        #         for param_group in opt.param_groups:
        #             param_group['lr'] = new_lr
        #         # 学习率使用小数形式显示
        #         logger.info(f"   📉 学习率下调: {current_lr:.8f} -> {new_lr:.8f} (下限: {min_lr})")
        #     else:
        #         logger.info(f"   🛑 学习率已达下限 ({min_lr})，不再下调")
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        return self._shared_step(batch, batch_idx, stage = "train")

    def _shared_step(self, batch, batch_idx, stage: str, return_outputs: bool = False):
        """共享的训练/验证/测试步骤"""
        # ======== Forward ========
        outputs = self(batch)  # 前向传播

        # 检查梯度相关的NaN
        if stage == "train":
            self._check_gradients("前向传播后")

        # 计算损失函数
        losses = self._compute_losses(outputs, batch, stage)

        # 获取batch_size
        batch_size = batch['numeric_features'].shape[0] if 'numeric_features' in batch else 1

        is_malicious_class_loss = losses.get("is_malicious_class_loss")
        # 为了在 tensorboard/logging 中记录指标
        if stage == "train":
            self.log(f"{stage}_is_malicious_class_loss", is_malicious_class_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        attack_family_class_loss = losses.get("attack_family_class_loss")
        if stage == "train" and attack_family_class_loss is not None:
            self.log(f"{stage}_attack_family_class_loss", attack_family_class_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        total_loss = losses["total_loss"]
        if stage == "train":
            self.log(f"{stage}_total_loss", total_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=batch_size)

        # 为is_malicious任务计算和记录性能Metrics
        self._compute_and_log_is_malicious_batch_metrics(stage, outputs, batch, batch_size)

        # 为attack_family多分类任务计算和记录性能Metrics
        self._compute_and_log_attack_family_batch_metrics(stage, outputs, batch, batch_size)

        # 反向传播前检查
        if stage == "train":
            self._check_gradients("反向传播前")

        if return_outputs:
            return outputs

        return total_loss
    
    def _compute_and_log_is_malicious_batch_metrics(self, stage, outputs, batch, batch_size):
        is_malicious_cls_logits = outputs['is_malicious_cls_logits']
        is_malicious_probs = torch.sigmoid(is_malicious_cls_logits)
        is_malicious_preds = (is_malicious_probs > 0.5).float()

        is_malicious_labels = batch['is_malicious_label'].to(is_malicious_cls_logits.device).float()
        if is_malicious_labels.dim() == 1:
            is_malicious_labels = is_malicious_labels.unsqueeze(1)

        if stage == "train":
            # 计算 accuracy / precision / recall / f1
            try:
                is_malicious_trues_np = is_malicious_labels.squeeze(1).cpu().numpy()
                is_malicious_preds_np = is_malicious_preds.squeeze(1).cpu().numpy()

                accuracy = accuracy_score(is_malicious_trues_np, is_malicious_preds_np)
                precision = precision_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
                recall = recall_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
                f1 = f1_score(is_malicious_trues_np, is_malicious_preds_np, zero_division=0)
            except Exception as e:
                logger.warning(f"计算指标时出错: {e}")
                accuracy = precision = recall = f1 = 0.0

            self.log(f"{stage}_is_malicious_accuracy", accuracy, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_is_malicious_precision", precision, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_is_malicious_recall", recall, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_is_malicious_f1", f1, prog_bar=True, sync_dist=True, batch_size=batch_size)
        elif stage == "val":
            # 保存每个 batch 的结果（必须 detach + cpu）
            self.val_is_malicious_labels.append(is_malicious_labels.detach().cpu())
            self.val_is_malicious_probs.append(is_malicious_probs.detach().cpu())
            self.val_is_malicious_preds.append(is_malicious_preds.detach().cpu())
        elif stage == "test":
            # 保存每个 batch 的结果（必须 detach + cpu）
            self.test_is_malicious_labels.append(is_malicious_labels.detach().cpu())
            self.test_is_malicious_probs.append(is_malicious_probs.detach().cpu())
            self.test_is_malicious_preds.append(is_malicious_preds.detach().cpu())
        else:
            raise ValueError(f"不支持的stage字符串: {stage}")

        return

    def _compute_and_log_attack_family_batch_metrics(self, stage, outputs, batch, batch_size):
        """
        计算并记录 attack_family 的 batch 级指标
        - 仅在 malicious 样本子集上评估
        - 使用 OVR + macro-F1
        """
        if "attack_family_cls_logits" not in outputs:
            return

        # logits / labels
        attack_family_logits = outputs["attack_family_cls_logits"]          # [B, K]
        attack_family_labels = batch["attack_family_label"].to(attack_family_logits.device)  # [B, K]

        # 只在 malicious 子集上评估
        is_malicious_label = batch["is_malicious_label"].to(attack_family_logits.device).view(-1) == 1
        if is_malicious_label.sum() == 0:
            return

        # 仅在「真实恶意流量」样本上评估 attack_family 分类结果。
        # 这里使用的是 is_malicious 的真实标签（ground truth），而不是模型预测结果。
        # 因此评估的是条件分类性能：
        #   P(attack_family | is_malicious = 1)，
        # 而不是“先预测是否恶意，再预测攻击家族”的级联预测流程。
        attack_family_logits = attack_family_logits[is_malicious_label]
        attack_family_labels = attack_family_labels[is_malicious_label]

        # OVR threshold
        attack_family_probs = torch.sigmoid(attack_family_logits)
        attack_family_preds = (attack_family_probs > 0.5).int()

        if stage == "train":
            try:
                # 转 numpy
                labels_np = attack_family_labels.cpu().numpy()
                preds_np = attack_family_preds.cpu().numpy()

                macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
                micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)
            except Exception as e:
                logger.warning(f"计算 attack_family 指标时出错: {e}")
                macro_f1 = micro_f1 = 0.0

            self.log(f"{stage}_attack_family_macro_f1", macro_f1, prog_bar=True, sync_dist=True, batch_size=batch_size)
            self.log(f"{stage}_attack_family_micro_f1", micro_f1, prog_bar=False, sync_dist=True, batch_size=batch_size)
        elif stage == "val":
            self.val_attack_family_labels.append(attack_family_labels.detach().cpu())
            self.val_attack_family_probs.append(attack_family_probs.detach().cpu())
            self.val_attack_family_preds.append(attack_family_preds.detach().cpu())
        elif stage == "test":
            self.test_attack_family_labels.append(attack_family_labels.detach().cpu())
            self.test_attack_family_probs.append(attack_family_probs.detach().cpu())
            self.test_attack_family_preds.append(attack_family_preds.detach().cpu())
        else:
            raise ValueError(f"不支持的stage字符串: {stage}")


    def _check_gradients(self, stage: str):
        """检查梯度状态"""
        if not self.debug_mode:
            return
            
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_count = torch.isnan(param.grad).sum().item()
                    logger.warning(f"⚠️ 梯度NaN: {name} 在 {stage} 有 {nan_count} 个NaN梯度")
                if torch.isinf(param.grad).any():
                    inf_count = torch.isinf(param.grad).sum().item()
                    logger.warning(f"⚠️ 梯度Inf: {name} 在 {stage} 有 {inf_count} 个Inf梯度")

    def on_after_backward(self):
        """反向传播后的钩子"""
        if self.debug_mode:
            self._check_gradients("反向传播后")
            
            # 梯度裁剪（预防梯度爆炸）
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

    def on_validation_epoch_start(self):
        """验证阶段开始：清空缓存"""
        self.val_is_malicious_labels = []
        self.val_is_malicious_preds = []
        self.val_is_malicious_probs = []

        self.val_attack_family_labels = []
        self.val_attack_family_preds = []
        self.val_attack_family_probs = []

    def validation_step(self, batch, batch_idx):
        # 调用 _shared_step 返回 outputs，不重复计算 loss
        self._shared_step(batch, batch_idx, stage="val", return_outputs=False)

        # 使用新的通用SHAP分析框架 del by qinyf 2012-12-02
        # if self.cfg.shap.enable_shap:
        #     if self.should_run_shap_analysis(self.current_epoch, batch_idx):
        #         logger.info(f"开始通用SHAP分析，epoch: {self.current_epoch}")
                
        #         try:
        #             # 执行SHAP分析
        #             shap_results = self.perform_shap_analysis(batch)
                    
        #             if shap_results and not shap_results.get('error'):
        #                 # 调用钩子方法，可以在子类中重写
        #                 self.on_shap_analysis_completed(shap_results)
                        
        #         except Exception as e:
        #             logger.error(f"通用SHAP分析失败: {e}")

        return None
        
    def on_validation_epoch_end(self):
        """验证集 epoch 结束时统一计算 F1 / precision / recall"""
        self._compute_and_log_is_malicious_epoch_metrics(
            stage="val",
            labels_list=self.val_is_malicious_labels,
            preds_list=self.val_is_malicious_preds,
            probs_list=self.val_is_malicious_probs
        )

        # 清空缓存
        self.val_is_malicious_labels.clear()
        self.val_is_malicious_preds.clear()
        self.val_is_malicious_probs.clear()

        # attack_family 的 epoch-level 逻辑（公共函数）
        self._compute_and_log_attack_family_epoch_metrics(
            stage="val",
            labels_list=self.val_attack_family_labels,
            preds_list=self.val_attack_family_preds,
        )

        # 清空缓存
        self.val_attack_family_labels.clear()
        self.val_attack_family_preds.clear()
        self.val_attack_family_probs.clear()
                
    def on_test_epoch_start(self):
        """测试阶段开始时初始化全局存储"""
        self.test_is_malicious_labels = []
        self.test_is_malicious_preds = []
        self.test_is_malicious_probs = []

        self.test_attack_family_labels = []
        self.test_attack_family_preds = []
        self.test_attack_family_probs = []

        # [新增] 初始化用于可视化的记录列表
        self.test_step_accuracies = []  # 存储每一步的准确率
        self.test_drift_steps = []  # 存储发生漂移的 step 索引

        # [新增] 在测试开始时初始化概念漂移管理器
        if self.drift_enabled and DRIFT_MANAGER_AVAILABLE and self.drift_manager is None:
            logger.info("🔧 [Test Phase] 初始化概念漂移管理器...")
            try:
                self.drift_manager = ConceptDriftManager(self.cfg, self)

                # [关键修改] 强制清空适应策略配置，实现“适应先不做”
                # 这会防止 adapter.adapt() 执行任何 LR 衰减或优化器重置操作
                if hasattr(self.drift_manager, 'adapter'):
                    logger.info("🛑 [Test Phase] 强制禁用漂移适应策略 (Adaptation Disabled)")
                    self.drift_manager.adapter.config = {}

                logger.info(f"✅ 主检测器: {self.drift_manager.main_name.upper()}")
            except Exception as e:
                logger.error(f"❌ 概念漂移管理器初始化失败: {e}")
                self.drift_enabled = False

        # 3. 组件重置 added by qinyf 2025-12-02
        if self.cfg.shap.enable_shap:
            self.shap_analyzer.reset()

    def test_step(self, batch, batch_idx):
        """测试步骤 - 完全复用 _shared_step"""
        # [修改] 调用 _shared_step 并获取 outputs (return_outputs=True)
        # 这样我们才能拿到 multiview_embeddings 用于检测
        outputs = self._shared_step(batch, batch_idx, stage="test", return_outputs=True)

        if batch_idx % 10 == 0:  # 防止刷屏，每10个batch打印一次，或者您可以去掉这个if
            try:
                # === A. 解析 Is Malicious (二分类) ===
                mal_labels = batch['is_malicious_label'].cpu().numpy()
                mal_names = ["Benign(0)", "Malicious(1)"]
                # 统计数量
                mal_counts = {name: np.sum(mal_labels == i) for i, name in enumerate(mal_names)}

                # === B. 解析 Attack Family (多分类) ===
                if 'attack_family_label' in batch:
                    family_labels = batch['attack_family_label']  # [B, Num_Classes] (Multi-hot or One-hot)

                    # 如果是 One-Hot/Multi-Hot，转为索引
                    if family_labels.dim() > 1:
                        family_indices = torch.argmax(family_labels, dim=1).cpu().numpy()
                    else:
                        family_indices = family_labels.cpu().numpy()

                    # 获取类别名称映射
                    # self.attack_family_names 来自初始化时的 labels_cfg
                    family_map = {i: name for i, name in enumerate(self.attack_family_names)}

                    # 统计当前 Batch 里有哪些攻击
                    unique_idxs, counts = np.unique(family_indices, return_counts=True)
                    family_stats = []
                    for idx, count in zip(unique_idxs, counts):
                        name = family_map.get(idx, f"Unknown({idx})")
                        family_stats.append(f"{name}: {count}")

                    # === 打印日志 ===
                    logger.info(f"🔍 [Step {batch_idx}] 数据分布:")
                    logger.info(f"   ► 二分类 (Is Malicious): {mal_counts}")
                    logger.info(f"   ► 攻击家族 (Ground Truth): {', '.join(family_stats)}")

                    # === C. (可选) 查看模型预测了什么 ===
                    # 获取预测 logits -> probabilities -> predictions
                    pred_logits = outputs['attack_family_cls_logits']
                    pred_indices = torch.argmax(pred_logits, dim=1).cpu().numpy()

                    unique_pred_idxs, pred_counts = np.unique(pred_indices, return_counts=True)
                    pred_stats = []
                    for idx, count in zip(unique_pred_idxs, pred_counts):
                        name = family_map.get(idx, f"Unknown({idx})")
                        pred_stats.append(f"{name}: {count}")

                    logger.info(f"   ► 模型预测 (Prediction)  : {', '.join(pred_stats)}")
                    logger.info("-" * 40)

            except Exception as e:
                logger.warning(f"打印标签分布出错: {e}")

            # 3. 漂移检测 (保持原有逻辑)
        if self.drift_enabled and self.drift_manager:
            try:
                features = outputs['multiview_embeddings'].detach()
                result = self.drift_manager.process_batch(
                    features=features,
                    global_step=batch_idx,
                    current_epoch=self.current_epoch
                )
            except Exception as e:
                if self.debug_mode:
                    logger.warning(f"Drift detection error in test_step: {e}")

        # 4. 组件数据收集 (SHAP)
        if self.cfg.shap.enable_shap:
            self.shap_analyzer.collect_batch(batch)

        return None

    def on_test_epoch_end(self):
        """测试阶段结束，汇总全局指标，多 GPU 下支持同步"""
        self._compute_and_log_is_malicious_epoch_metrics(
            stage="test",
            labels_list=self.test_is_malicious_labels,
            preds_list=self.test_is_malicious_preds,
            probs_list=self.test_is_malicious_probs,
        )
        # 清空存储
        self.test_is_malicious_labels.clear()
        self.test_is_malicious_preds.clear()
        self.test_is_malicious_probs.clear()

        # attack_family 的 epoch-level 逻辑（公共函数）
        self._compute_and_log_attack_family_epoch_metrics(
            stage="test",
            labels_list=self.test_attack_family_labels,
            preds_list=self.test_attack_family_preds,
        )
        # [新增] 绘制 精度趋势与概念漂移点 图
        if len(self.test_step_accuracies) > 0 and self.trainer.is_global_zero:
            self._plot_drift_accuracy_curve()

        # 清空存储
        self.test_attack_family_labels.clear()
        self.test_attack_family_preds.clear()
        self.test_attack_family_probs.clear()

        # 5. 组件执行分析  added by qinyf 2025-12-02
        if self.cfg.shap.enable_shap:
            self.shap_analyzer.finalize()

    def _plot_drift_accuracy_curve(self):
        """绘制测试阶段的精度变化与概念漂移点"""
        try:
            steps = range(len(self.test_step_accuracies))
            accuracies = self.test_step_accuracies

            plt.figure(figsize=(12, 6))

            # 1. 绘制精度曲线
            plt.plot(steps, accuracies, label='Batch Accuracy', color='blue', alpha=0.6, linewidth=1)

            # (可选) 绘制移动平均线使其更平滑
            if len(accuracies) > 10:
                window_size = 10
                moving_avg = np.convolve(accuracies, np.ones(window_size) / window_size, mode='valid')
                # 对齐 x 轴
                plt.plot(range(window_size - 1, len(accuracies)), moving_avg,
                         label=f'Moving Avg (w={window_size})', color='darkblue', linewidth=2)

            # 2. 标记漂移点
            if self.test_drift_steps:
                for drift_step in self.test_drift_steps:
                    plt.axvline(x=drift_step, color='red', linestyle='--', alpha=0.8)

                # 仅在图例中添加一次 Drift 标签
                plt.axvline(x=self.test_drift_steps[0], color='red', linestyle='--', alpha=0.8, label='Drift Detected')

            plt.title('Test Phase: Accuracy Trend & Concept Drift Detection')
            plt.xlabel('Test Step (Batch Index)')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1.05)
            plt.legend(loc='lower right')
            plt.grid(True, alpha=0.3)

            # 3. 保存图片
            save_dir = self.cfg.logging.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"test_drift_accuracy_{timestamp}.png")

            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()

            logger.info(f"📊 概念漂移与精度分析图已保存至: {save_path}")

        except Exception as e:
            logger.warning(f"绘制漂移分析图失败: {e}")

    def _compute_and_log_is_malicious_epoch_metrics(
        self,
        stage: str,
        labels_list: List[torch.Tensor],
        preds_list:  List[torch.Tensor],
        probs_list:  List[torch.Tensor],
    ):
        if len(labels_list) == 0:
            logger.warning(f"[{stage}] is_malicious labels empty")
            return

        labels = torch.cat(labels_list, dim=0)
        preds  = torch.cat(preds_list, dim=0)
        probs  = torch.cat(probs_list, dim=0)

        # ---- DDP gather ----
        if self.trainer.world_size > 1:
            labels = self.all_gather(labels).view(-1)
            preds  = self.all_gather(preds).view(-1)
            probs  = self.all_gather(probs).view(-1)

        labels_np = labels.cpu().numpy()
        preds_np  = preds.cpu().numpy()
        probs_np  = probs.cpu().numpy()

        accuracy  = accuracy_score(labels_np, preds_np)
        precision = precision_score(labels_np, preds_np, zero_division=0)
        recall  = recall_score(labels_np, preds_np, zero_division=0)
        f1   = f1_score(labels_np, preds_np, zero_division=0)

        self.log(f"{stage}_is_malicious_accuracy", accuracy, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_malicious_precision", precision, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_malicious_recall", recall, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log(f"{stage}_is_malicious_f1", f1, on_epoch=True, prog_bar=False, sync_dist=True)

        # 🔴🔴 只在主进程上输出is_malicious任务的测试阶段的完整报告
        if stage == "test" and self.trainer.is_global_zero:
            logger.info("=" * 60)
            logger.info("🤖 最佳模型的is_malicious任务的测试报告")
            logger.info("=" * 60)

            # ---- 整理详细的模型性能报告 ----
            logger.info(f"📊 is_malicious任务的基础指标:")
            logger.info(f"   准确率: {accuracy:.4f}")
            logger.info(f"   精确率: {precision:.4f}")
            logger.info(f"   召回率: {recall:.4f}")
            logger.info(f"   F1分数: {f1:.4f}")
            
            # 分类报告
            try:
                report = classification_report(labels_np, preds_np, digits=4, target_names=['正常', '恶意'])
                logger.info("📋 is_malicious任务的详细分类报告:")
                logger.info("\n" + report)
            except Exception as e:
                logger.warning(f"is_malicious任务的分类报告生成失败: {e}")
            
            # 混淆矩阵
            try:
                cm = confusion_matrix(labels_np, preds_np)
                logger.info("🎯 is_malicious任务的混淆矩阵:")
                logger.info(f"\n{cm}")
            except Exception as e:
                logger.warning(f"is_malicious任务的混淆矩阵生成失败: {e}")
            
            # ROC-AUC 和 PR曲线
            try:
                auc = roc_auc_score(labels_np, probs_np)
                avg_precision = average_precision_score(labels_np, probs_np)
                logger.info(f"📈 is_malicious任务的高级指标:")
                logger.info(f"   ROC-AUC: {auc:.4f}")
                logger.info(f"   Average Precision: {avg_precision:.4f}")
            except Exception as e:
                logger.warning(f"高级指标计算失败: {e}")
            
            # 样本统计
            logger.info(f"📊 样本的is_malicious标签数量统计:")
            logger.info(f"   总样本数: {len(labels)}")
            logger.info(f"   正样本数: {labels_np.sum()}")
            logger.info(f"   负样本数: {len(labels) - labels_np.sum()}")
            logger.info(f"   正样本比例: {labels_np.mean():.2%}")
            
            logger.info("=" * 60)


    def _compute_and_log_attack_family_epoch_metrics(self, stage: str, labels_list, preds_list):
        """
        在 epoch 结束时统一计算并 log attack_family 指标
        用于 val / test 阶段（train 不走这里）
        """
        assert stage in ("val", "test"), f"非法 stage={stage}"

        if not labels_list or len(labels_list) == 0:
            logger.warning(f"[{stage}] attack_family: 无可用样本")
            return

        # 拼接，转换为numpy数组
        try:
            labels_np = torch.cat(labels_list, dim=0).cpu().numpy()
            preds_np = torch.cat(preds_list, dim=0).cpu().numpy()
        except Exception as e:
            logger.error(f"数据拼接、转换失败: {e}")
            return

        # 计算指标
        macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        micro_f1 = f1_score(labels_np, preds_np, average="micro", zero_division=0)

        # log（epoch-level，不要 batch_size）
        self.log(f"{stage}_attack_family_macro_f1", macro_f1, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(f"{stage}_attack_family_micro_f1", micro_f1, on_epoch=True, prog_bar=False, sync_dist=True)

        # 🔴 test 阶段的简要最终报告
        if stage == "test" and self.trainer.is_global_zero:
            logger.info("=" * 60)
            logger.info("🤖 attack_family 任务测试报告（简要）")
            logger.info(f"macro_f1={macro_f1:.4f}, micro_f1={micro_f1:.4f}")

            try:
                report = classification_report(labels_np, preds_np, digits=4, zero_division=0,
                    target_names=self.attack_family_names)
                logger.info("📋 attack_family 任务的分类报告:")
                logger.info("\n" + report)
            except Exception as e:
                logger.warning(f"attack_family 分类报告生成失败: {e}")

            per_attack_family_f1 = f1_score(labels_np, preds_np, average=None, zero_division=0)
            logger.info("📊 attack_family per-class F1:")

            for name, f1v in zip(self.attack_family_names, per_attack_family_f1):
                logger.info(f"  {name:20s}: F1={f1v:.4f}")

            logger.info("=" * 60)


    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 使用 datamodule 获取训练集长度
        if self.trainer and hasattr(self.trainer, "datamodule") and self.trainer.datamodule is not None:
            train_loader = self.trainer.datamodule.train_dataloader()
            total_steps = len(train_loader) * self.trainer.max_epochs
        else:
            # 如果 trainer 还未初始化，使用默认步数（可根据 cfg 设置）
            total_steps = self.cfg.optimizer.default_total_steps

        warmup_steps = int(total_steps * self.cfg.optimizer.warmup_ratio)

        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.optimizer.lr,
            weight_decay=self.cfg.optimizer.weight_decay,
            eps=getattr(self.cfg.optimizer, 'eps', 1e-8)
        )

        scheduler_type = self.cfg.scheduler.type
        if scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        elif scheduler_type == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )
        else:  # linear
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            }
        }
    

    def _fuse_multi_views(self, sequence_outputs, text_outputs, tabular_outputs):
        """使用配置的融合方法融合多视图特征"""
        view_embeddings = []
        
        # 收集所有启用的视图
        view_embeddings.append(tabular_outputs)  # 数值特征必选 + 域名特征可选
        
        if self.text_features_enabled:
            view_embeddings.append(text_outputs)
        
        if self.sequence_features_enabled:
            view_embeddings.append(sequence_outputs)
        
        # 使用配置的融合方法
        if len(view_embeddings) > 1:
            fused_embedding = self.fusion_layer(view_embeddings)
        else:
            # 单视图情况，直接使用
            fused_embedding = view_embeddings[0]
        
        return fused_embedding

    # ================== 新增：训练结束回调 ==================
    def on_train_end(self):
            """训练结束时打印漂移对比报告"""
            super().on_train_end()
            if self.drift_manager:
                report = self.drift_manager.generate_report()
                logger.info("\n" + report)

    # ================== 新增：测试结束回调 ==================
    def on_test_end(self):
        super().on_test_end()
        # 如果需要在测试结束也打印，可以复用逻辑，但通常漂移主要关注训练时的适应
        pass