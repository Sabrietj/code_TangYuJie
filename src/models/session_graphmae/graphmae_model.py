# src/models/session_graphmae/graphmae_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.nn import GATConv, global_mean_pool
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import logging
import numpy as np

logger = logging.getLogger(__name__)


def sce_loss(x, y, alpha=2.0):
    """
    缩放余弦误差 (Scaled Cosine Error, SCE)
    GraphMAE 论文中证明 SCE 比 MSE 更能防止特征崩塌 (Feature Collapse)。
    alpha 越大，对困难样本的惩罚越高。
    """
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)
    # (1 - cos_sim)^alpha
    loss = (1.0 - (x * y).sum(dim=-1)).pow_(alpha)
    return loss.mean()


class SessionGraphMAE(pl.LightningModule):
    def __init__(self, cfg, node_encoder):
        """
        :param cfg: 配置文件
        :param node_encoder: 节点特征编码器 (包装了你原有的 FlowBertMultiview 提取层，
                             负责将单个流的多种字典特征变成统一的 d 维向量)
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters(ignore=['node_encoder'])

        # 1. 节点级多视图编码器 (预训练好或者联合训练)
        self.node_encoder = node_encoder
        self.hidden_dim = cfg.model.get('hidden_dim', 768)  # 默认和 BERT 维度对齐

        # 2. GraphMAE 掩码组件
        self.mask_rate = cfg.model.get('mask_rate', 0.5)  # 默认掩码 50% 的节点
        self.mask_token = nn.Parameter(torch.zeros(1, self.hidden_dim))
        nn.init.xavier_uniform_(self.mask_token)  # 初始化可学习的 MASK 向量

        # 3. 图编码器 GNN Encoder (2层 GAT，用于提取图拓扑上下文)
        # 注意：这里使用 concat=False 保持输出维度仍为 hidden_dim
        self.encoder_layer1 = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)
        self.encoder_layer2 = GATConv(self.hidden_dim, self.hidden_dim, heads=4, concat=False)

        # 4. 图解码器 GNN Decoder (单层 GAT，专门用于还原被 MASK 的节点特征)
        self.decoder = GATConv(self.hidden_dim, self.hidden_dim, heads=1, concat=False)

        # 5. 图级别分类器 (Global Graph Classifier)
        self.is_malicious_clf = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)  # 二分类：正常 vs 恶意
        )

        # 获取你的真实类别数 (正常+攻击家族)
        self.num_classes = cfg.data.get('num_classes', 6)
        self.attack_family_clf = nn.Sequential(
            nn.Linear(self.hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)  # OVR 多分类
        )

        # 损失权重控制 (用于多任务学习)
        self.lambda_recon = cfg.model.get('lambda_recon', 1.0)
        self.lambda_cls = cfg.model.get('lambda_cls', 1.0)

        # 内部缓存，用于 Epoch 结束时计算指标
        self.validation_step_outputs = []

    def mask_nodes(self, x):
        """按比例随机掩码图中的节点特征"""
        num_nodes = x.size(0)
        num_mask = int(self.mask_rate * num_nodes)

        # 随机打乱索引，划分 mask 和 keep 集合
        perm = torch.randperm(num_nodes, device=x.device)
        mask_idx = perm[:num_mask]
        keep_idx = perm[num_mask:]

        # 将选中的节点特征替换为全局统一的 mask_token
        x_masked = x.clone()
        x_masked[mask_idx] = self.mask_token

        return x_masked, mask_idx, keep_idx

    def forward(self, batch):
        """推理阶段：不做掩码，直接提取全图特征进行分类"""
        # 1. 调用多视图编码器，得到密集的节点矩阵 X [N, hidden_dim]
        x = self.node_encoder(batch)
        edge_index = batch.edge_index

        # 2. GNN Encoder 提取带拓扑信息的节点表征
        z = F.relu(self.encoder_layer1(x, edge_index))
        z = self.encoder_layer2(z, edge_index)

        # 3. 全局池化 (Readout)：平均池化得到整张图（Session）的 Embedding [Batch_Size, hidden_dim]
        graph_emb = global_mean_pool(z, batch.batch)

        # 4. 图级别分类
        logits_bin = self.is_malicious_clf(graph_emb)
        logits_multi = self.attack_family_clf(graph_emb)

        return logits_bin, logits_multi

    def training_step(self, batch, batch_idx):
        edge_index = batch.edge_index

        # 1. 提取原始多视图特征，作为重建的 Ground Truth (目标值)
        x_orig = self.node_encoder(batch)

        # 2. 对部分节点进行随机掩码
        x_masked, mask_idx, _ = self.mask_nodes(x_orig)

        # 3. GNN 编码器处理带掩码的图
        z = F.relu(self.encoder_layer1(x_masked, edge_index))
        z = self.encoder_layer2(z, edge_index)

        # 4. GNN 解码器尝试恢复节点特征
        h_recon = self.decoder(z, edge_index)

        # ==========================================
        # 损失计算 1: 仅对被掩码节点计算 SCE 重建损失
        # ==========================================
        loss_recon = sce_loss(h_recon[mask_idx], x_orig[mask_idx], alpha=2.0)

        # ==========================================
        # 损失计算 2: 下游分类任务损失 (多任务)
        # ==========================================
        graph_emb = global_mean_pool(z, batch.batch)
        logits_bin = self.is_malicious_clf(graph_emb)
        logits_multi = self.attack_family_clf(graph_emb)

        # 假设图对象自带图级别的标签 (y_bin, y_multi)
        loss_bin = F.binary_cross_entropy_with_logits(logits_bin.squeeze(), batch.y_bin.float())
        # OVR 多分类，所以依然用 BCE
        loss_multi = F.binary_cross_entropy_with_logits(logits_multi, batch.y_multi.float())

        # 总损失 = 重建损失 + 分类损失
        total_loss = self.lambda_recon * loss_recon + self.lambda_cls * (loss_bin + loss_multi)

        # 记录日志
        self.log('train_loss', total_loss, batch_size=batch.num_graphs)
        self.log('loss_recon', loss_recon, batch_size=batch.num_graphs)
        self.log('loss_cls', loss_bin + loss_multi, batch_size=batch.num_graphs)

        return total_loss

    def validation_step(self, batch, batch_idx):
        """验证阶段不使用掩码，主要评估分类性能"""
        logits_bin, logits_multi = self(batch)  # 调用 forward

        # 收集结果用于 epoch 结束打印报告
        self.validation_step_outputs.append({
            "preds_bin": torch.sigmoid(logits_bin).squeeze() > 0.5,
            "targets_bin": batch.y_bin,
            "preds_multi": torch.sigmoid(logits_multi) > 0.5,  # 多标签/OVR 阈值0.5
            "targets_multi": batch.y_multi
        })

    def on_validation_epoch_end(self):
        """验证集统一输出成绩单 (Baseline 专属打印)"""
        # 检查是否为正式评估阶段（防止训练中途狂刷屏）
        is_final_baseline = getattr(self, "print_baseline_report", False)
        if not is_final_baseline or self.trainer.sanity_checking:
            self.validation_step_outputs.clear()
            return

        preds_bin = torch.cat([x["preds_bin"] for x in self.validation_step_outputs]).cpu().numpy()
        targets_bin = torch.cat([x["targets_bin"] for x in self.validation_step_outputs]).cpu().numpy()

        logger.info("=" * 60)
        logger.info("🏆 GraphMAE (Session-level) 验证集 Baseline 成绩单")
        logger.info("=" * 60)
        logger.info("【任务 1】二分类 (Is Malicious) 报告:")

        # 增加 labels=[0,1] 避免时序切分导致的数据缺失报错
        report_bin = classification_report(
            targets_bin, preds_bin,
            labels=[0, 1],
            target_names=["Benign", "Malicious"],
            digits=4,
            zero_division=0
        )
        logger.info(f"\n{report_bin}")

        # --- 多分类评估 (省略具体拼接，逻辑类似) ---
        # preds_multi = torch.cat(...); targets_multi = torch.cat(...)
        # report_multi = classification_report(...)

        logger.info("=" * 60)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        # 联合优化 GNN、分类器 以及底层的 Node Encoder（如果未冻结）
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr, weight_decay=1e-4)
        return optimizer