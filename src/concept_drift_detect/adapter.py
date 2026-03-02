import torch
import torch.nn as nn
import torch.optim as optim
import logging
import copy
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)


class IncrementalAdapter:
    def __init__(self, model, config: dict, **kwargs):
        self.model = model
        self.config = config

        self.buffer_size = config.get('buffer_size', 2000)
        self.batch_size = config.get('batch_size', 32)
        self.learning_rate = config.get('lr', 1e-4)
        self.epochs = config.get('epochs', 3)

        self.lambda_topo = config.get('lambda_topo', 0.1)
        self.lambda_cons = config.get('lambda_cons', 0.1)
        self.lambda_proto = config.get('lambda_proto', 0.1)

        self.replay_buffer = []

        self.classifiers = self._get_target_classifiers()
        logger.info(f"🔍 Joint Adapter targeted classifiers: {self.classifiers}")

    def _get_target_classifiers(self):
        clss = []
        if hasattr(self.model, 'is_malicious_classifier') and self.model.is_malicious_classifier is not None:
            clss.append('is_malicious_classifier')
        if hasattr(self.model, 'attack_family_classifier') and self.model.attack_family_classifier is not None:
            clss.append('attack_family_classifier')
        return clss

    def update_buffer(self, features, bin_labels, mul_labels):
        for i in range(len(features)):
            feat = features[i].detach().cpu()
            b_lbl = bin_labels[i].detach().cpu()
            m_lbl = mul_labels[i].detach().cpu()

            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append((feat, b_lbl, m_lbl))
            else:
                idx = torch.randint(0, len(self.replay_buffer), (1,)).item()
                self.replay_buffer[idx] = (feat, b_lbl, m_lbl)

    def _get_features_and_logits(self, classifier_or_name, x):
        if isinstance(classifier_or_name, str):
            classifier = getattr(self.model, classifier_or_name)
        else:
            classifier = classifier_or_name

        if isinstance(classifier, nn.Sequential):
            h = x
            for layer in classifier[:-1]:
                h = layer(h)
            logits = classifier[-1](h)
            return h, logits
        else:
            return x, classifier(x)

    def adapt(self, drift_data_features, label_arg1, label_arg2=None, **kwargs):
        device = self.model.device

        # 解析双标签输入
        if label_arg2 is not None and isinstance(label_arg2, torch.Tensor):
            drift_bin_labels = label_arg1
            drift_mul_labels = label_arg2
        else:
            labels = label_arg1
            if labels.dim() > 1 and labels.shape[1] > 1:
                drift_mul_labels = torch.argmax(labels, dim=1)
                drift_bin_labels = (labels.sum(dim=-1) > 0).float()
            else:
                drift_bin_labels = labels.float()
                drift_mul_labels = torch.zeros_like(labels)

        self.update_buffer(drift_data_features, drift_bin_labels, drift_mul_labels)

        if not self.replay_buffer:
            return

        buf_feats = torch.stack([x[0] for x in self.replay_buffer])
        buf_bins = torch.stack([x[1] for x in self.replay_buffer])
        buf_muls = torch.stack([x[2] for x in self.replay_buffer])

        train_feats = torch.cat([drift_data_features.cpu(), buf_feats], dim=0)
        train_bins = torch.cat([drift_bin_labels.cpu(), buf_bins], dim=0)
        train_muls = torch.cat([drift_mul_labels.cpu(), buf_muls], dim=0)

        dataset = TensorDataset(train_feats, train_bins, train_muls)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._freeze_backbone()

        opt_params = []
        old_classifiers = {}
        for cls_name in self.classifiers:
            classifier = getattr(self.model, cls_name)
            opt_params.extend(classifier.parameters())

            old_c = copy.deepcopy(classifier)
            old_c.eval()
            for p in old_c.parameters(): p.requires_grad = False
            old_classifiers[cls_name] = old_c

        optimizer = optim.Adam(opt_params, lr=self.learning_rate)
        criterion_bin = nn.BCEWithLogitsLoss()
        criterion_mul = nn.CrossEntropyLoss()

        self.model.train()

        # 🚨 [致命修复]: 强行开启梯度，打破 Lightning Test阶段的 no_grad() 枷锁！
        with torch.enable_grad():
            for epoch in range(self.epochs):
                for batch_x, batch_bin, batch_mul in loader:
                    batch_x, batch_bin, batch_mul = batch_x.to(device), batch_bin.to(device), batch_mul.to(device)
                    optimizer.zero_grad()

                    loss_cls = torch.tensor(0.0, device=device)
                    total_loss_topo = torch.tensor(0.0, device=device)
                    total_loss_cons = torch.tensor(0.0, device=device)
                    total_loss_proto = torch.tensor(0.0, device=device)

                    for cls_name in self.classifiers:
                        h_new, logits_new = self._get_features_and_logits(cls_name, batch_x)

                        with torch.no_grad():
                            old_c = old_classifiers[cls_name]
                            h_old, _ = self._get_features_and_logits(old_c, batch_x)

                        if cls_name == 'is_malicious_classifier':
                            target_y = batch_bin.unsqueeze(1).float() if batch_bin.dim() == 1 else batch_bin.float()
                            loss_cls += criterion_bin(logits_new, target_y)
                        elif cls_name == 'attack_family_classifier':
                            mask = (batch_bin.view(-1) == 1)
                            if mask.sum() > 0:
                                loss_cls += criterion_mul(logits_new[mask], batch_mul[mask].long())

                        dist_new = torch.cdist(h_new, h_new)
                        dist_old = torch.cdist(h_old, h_old)
                        total_loss_topo += nn.functional.mse_loss(dist_new, dist_old)

                        noise = torch.randn_like(batch_x) * 0.05
                        _, logits_noisy = self._get_features_and_logits(cls_name, batch_x + noise)
                        total_loss_cons += nn.functional.mse_loss(logits_noisy, logits_new.detach())

                        y_idx = batch_bin if cls_name == 'is_malicious_classifier' else batch_mul
                        h_proto = h_new

                        if cls_name == 'attack_family_classifier':
                            mask = (batch_bin.view(-1) == 1)
                            if mask.sum() > 1:
                                h_proto = h_new[mask]
                                y_idx = y_idx[mask]
                            else:
                                continue

                        unique_c = torch.unique(y_idx.long())
                        valid_p = 0
                        lp = torch.tensor(0.0, device=device)

                        for c in unique_c:
                            c_mask = (y_idx.long() == c)
                            if c_mask.sum() > 1:
                                feats_c = h_proto[c_mask]
                                proto = feats_c.mean(dim=0, keepdim=True)
                                lp += nn.functional.mse_loss(feats_c, proto.expand_as(feats_c))
                                valid_p += 1

                        if valid_p > 0:
                            total_loss_proto += lp / valid_p

                    loss = loss_cls + (self.lambda_topo * total_loss_topo) + (self.lambda_cons * total_loss_cons) + (
                                self.lambda_proto * total_loss_proto)

                    loss.backward()
                    optimizer.step()

        self._unfreeze_backbone()
        self.model.eval()

    def _freeze_backbone(self):
        for name, param in self.model.named_parameters():
            requires_grad = False
            for cls_name in self.classifiers:
                if cls_name in name:
                    requires_grad = True
            param.requires_grad = requires_grad

    def _unfreeze_backbone(self):
        for param in self.model.parameters():
            param.requires_grad = True