import torch
import torch.nn as nn
import torch.optim as optim
import logging
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class IncrementalAdapter:
    def __init__(self, model, config: dict, task_mode="binary"):
        self.model = model
        self.config = config
        self.task_mode = task_mode

        self.buffer_size = config.get('buffer_size', 2000)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('lr', 1e-4)
        self.epochs = config.get('epochs', 3)

        self.replay_buffer = []

        # 根据任务模式选择分类头
        self.classifier_layer_name = self._get_target_classifier()
        logger.info(f"🔍 Adapter targeted classifier: {self.classifier_layer_name} (Mode: {task_mode})")

    def _get_target_classifier(self):
        """根据任务模式返回分类器名称"""
        if self.task_mode == "multiclass":
            # 优先找 attack_family_classifier
            if hasattr(self.model, 'attack_family_classifier') and self.model.attack_family_classifier is not None:
                return 'attack_family_classifier'
            else:
                logger.warning(
                    "Task is multiclass but 'attack_family_classifier' not found. Falling back to is_malicious.")
                return 'is_malicious_classifier'
        else:
            # 默认二分类
            return 'is_malicious_classifier'

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
        # logger.info(f"🔄 Adaptation Triggered | New Samples: {len(drift_data_features)}")

        # 1. 扩展分类头 (仅限 multiclass 且发现新类)
        if new_class_detected and self.task_mode == "multiclass":
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

        # 选择 Loss
        # 如果是二分类且输出维度为1 -> BCE
        # 如果是多分类 -> CrossEntropy

        # 判断最后一层的输出维度
        if isinstance(classifier, nn.Sequential):
            out_dim = classifier[-1].out_features
        else:
            out_dim = classifier.out_features

        is_binary_loss = (out_dim == 1)

        if is_binary_loss:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.CrossEntropyLoss()

        self.model.train()
        total_loss = 0
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad()

                outputs = classifier(batch_x)

                if is_binary_loss:
                    if batch_y.dim() == 1:
                        batch_y = batch_y.unsqueeze(1)
                    loss = criterion(outputs, batch_y.float())
                else:
                    loss = criterion(outputs, batch_y.long())  # CE需要long

                loss.backward()
                optimizer.step()
                total_loss += loss.item()

        self._unfreeze_backbone()
        self.model.eval()
        # logger.info(f"✅ Model Adapted (Loss: {total_loss/len(loader):.4f})")

    def _expand_classification_head(self, new_labels):
        classifier = getattr(self.model, self.classifier_layer_name)

        if isinstance(classifier, nn.Sequential):
            last_layer_idx = len(classifier) - 1
            last_layer = classifier[last_layer_idx]
        else:
            last_layer = classifier

        if not isinstance(last_layer, nn.Linear):
            return

        max_label = new_labels.max().item()
        current_dims = last_layer.out_features

        if max_label >= current_dims:
            new_dims = max_label + 1
            logger.info(f"📈 Expanding Classifier Last Layer: {current_dims} -> {new_dims}")

            old_weights = last_layer.weight.data
            old_bias = last_layer.bias.data

            new_layer = nn.Linear(last_layer.in_features, new_dims).to(self.model.device)

            with torch.no_grad():
                # 复制旧权重
                new_layer.weight.data[:current_dims] = old_weights
                new_layer.bias.data[:current_dims] = old_bias
                # 新增部分保持随机初始化

            if isinstance(classifier, nn.Sequential):
                classifier[last_layer_idx] = new_layer
            else:
                setattr(self.model, self.classifier_layer_name, new_layer)

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            if self.classifier_layer_name not in name:
                param.requires_grad = False

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True