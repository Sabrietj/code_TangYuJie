import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class IncrementalAdapter:
    def __init__(self, model, config: dict):
        self.model = model
        self.config = config
        self.buffer_size = config.get('buffer_size', 2000)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('lr', 1e-5)
        self.epochs = config.get('epochs', 3)

        self.replay_buffer = []

        # 自动寻找分类头层
        self.classifier_layer_name = self._find_classifier_layer()
        logger.info(f"🔍 Found classifier layer: {self.classifier_layer_name}")

    def _find_classifier_layer(self):
        """尝试寻找模型中的分类全连接层"""
        candidates = ['classifier_head', 'classifier', 'fc', 'output_layer']
        for name in candidates:
            if hasattr(self.model, name):
                return name
        # 如果找不到，打印所有模块供调试
        logger.warning("Could not find standard classifier layer. Using default 'classifier_head'.")
        return 'classifier_head'

    def update_buffer(self, features, labels):
        """蓄水池采样更新 Replay Buffer"""
        for i in range(len(features)):
            feat = features[i].detach().cpu()
            lbl = labels[i].detach().cpu()
            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append((feat, lbl))
            else:
                idx = torch.randint(0, len(self.replay_buffer), (1,)).item()
                self.replay_buffer[idx] = (feat, lbl)

    def adapt(self, drift_data_features, drift_data_labels, new_class_detected=False):
        device = self.model.device
        logger.info(f"🔄 Adaptation Triggered | New Samples: {len(drift_data_features)}")

        # 1. 扩展分类头
        if new_class_detected:
            self._expand_classification_head(drift_data_labels)

        # 2. 更新缓冲区
        self.update_buffer(drift_data_features, drift_data_labels)

        # 3. 准备数据 (Mix)
        if not self.replay_buffer:
            return

        buf_feats = torch.stack([x[0] for x in self.replay_buffer])
        buf_lbls = torch.stack([x[1] for x in self.replay_buffer])

        train_feats = torch.cat([drift_data_features.cpu(), buf_feats], dim=0)
        train_lbls = torch.cat([drift_data_labels.cpu(), buf_lbls], dim=0)

        dataset = TensorDataset(train_feats, train_lbls)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # 4. 微调
        self._freeze_backbone()
        classifier = getattr(self.model, self.classifier_layer_name)
        optimizer = optim.Adam(classifier.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()
                outputs = classifier(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

        self._unfreeze_backbone()
        self.model.eval()
        logger.info("✅ Model Adapted Successfully")

    def _expand_classification_head(self, new_labels):
        classifier = getattr(self.model, self.classifier_layer_name)
        max_label = new_labels.max().item()
        current_dims = classifier.out_features

        if max_label >= current_dims:
            new_dims = max_label + 1
            logger.info(f"📈 Expanding Classifier: {current_dims} -> {new_dims}")

            old_weights = classifier.weight.data
            old_bias = classifier.bias.data

            new_layer = nn.Linear(classifier.in_features, new_dims).to(self.model.device)
            new_layer.weight.data[:current_dims] = old_weights
            new_layer.bias.data[:current_dims] = old_bias

            setattr(self.model, self.classifier_layer_name, new_layer)

    def _freeze_backbone(self):
        # 简单策略：冻结除分类头以外的所有参数
        for name, param in self.model.named_parameters():
            if self.classifier_layer_name not in name:
                param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True