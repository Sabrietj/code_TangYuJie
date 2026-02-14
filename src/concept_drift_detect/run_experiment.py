import sys
import os
import copy
import argparse

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import hydra
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# 引入项目模块
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector
from src.concept_drift_detect.adapter import IncrementalAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftExp")


class DriftExperimentRunner:
    def __init__(self, model, cfg, detector_type="adwin", enable_adaptation=True, task_mode="binary"):
        self.model = model
        self.cfg = cfg
        self.detector_type = detector_type
        self.enable_adaptation = enable_adaptation
        self.task_mode = task_mode  # 'binary' or 'multiclass'

        # 检测器配置 (可根据需要调优)
        det_config = {
            'seed': 2026,
            'threshold': 0.01,  # BNDM 敏感度 (P-value阈值，越小越不敏感)
            'max_level': 6,  # BNDM 树深度
            'window_size': 1000,  # BNDM 窗口
            'delta': 0.002  # ADWIN 敏感度 (越小越敏感)
        }

        if detector_type == "bndm":
            self.detector = BNDMDetector(det_config)
        elif detector_type == "adwin":
            self.detector = ADWINDetector(det_config)
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

        # 适应器配置
        adapt_config = {
            'lr': 1e-4,
            'epochs': 3,  # 快速适应，避免过拟合
            'buffer_size': 2000,
            'batch_size': 32
        }
        # 传入 task_mode 让 adapter 知道优化哪个头
        self.adapter = IncrementalAdapter(model, adapt_config, task_mode=task_mode)

        self.metrics = {
            "processed": 0,
            "drifts": 0,
            "accuracy_history": [],
            "adaptation_points": []
        }

        # 缓冲区
        self.buffer_features = []
        self.buffer_labels = []

    def run_stream(self, dataloader):
        self.model.eval()
        self.model.cuda()

        status_desc = f"[{'Adapt' if self.enable_adaptation else 'Fixed'}] {self.detector_type.upper()} ({self.task_mode})"
        progress = tqdm(dataloader, desc=status_desc)

        for batch in progress:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 1. 确定标签 (Ground Truth)
            if self.task_mode == "multiclass":
                if 'attack_family_label' in batch:
                    labels = batch['attack_family_label']
                else:
                    # 如果没有多分类标签，回退到 label
                    labels = batch.get('label', batch['is_malicious_label'])
            else:  # binary
                labels = batch['is_malicious_label']

            # 2. 模型推理
            with torch.no_grad():
                outputs = self.model(batch)
                features = outputs['multiview_embeddings']

                # 确定预测 (Prediction)
                if self.task_mode == "multiclass":
                    if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                        logits = outputs['attack_family_cls_logits']
                        preds = torch.argmax(logits, dim=1)
                    else:
                        # 强行回退
                        preds = (torch.sigmoid(outputs['is_malicious_cls_logits']) > 0.5).long().squeeze()
                else:  # binary
                    logits = outputs['is_malicious_cls_logits']
                    preds = (torch.sigmoid(logits) > 0.5).long().squeeze()

            # 3. 逐样本/Batch处理
            batch_size = features.shape[0]

            # 统一转为 CPU numpy 计算精度
            if labels.dim() > 1:  # One-hot 转 index
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels

            # 确保 preds 和 labels_idx 维度一致
            if preds.shape != labels_idx.shape:
                preds = preds.view_as(labels_idx)

            acc_batch = (preds == labels_idx).float().mean().item()
            self.metrics["processed"] += batch_size
            self.metrics["accuracy_history"].extend((preds == labels_idx).tolist())  # 记录每个样本的对错

            # 更新进度条
            progress.set_postfix({"Acc": f"{np.mean(self.metrics['accuracy_history'][-1000:]):.4f}"})

            # 4. 漂移检测
            for i in range(batch_size):
                feat = features[i]
                lbl = labels_idx[i]

                # 预处理输入
                feat_input = feat.unsqueeze(0)  # [1, D]

                # 调用检测器
                drift_detected = self.detector.update(self.detector.preprocess(feat_input))

                # 存入 buffer
                self.buffer_features.append(feat)
                self.buffer_labels.append(lbl)

                if drift_detected:
                    self.metrics["drifts"] += 1

                    # 只有开启适应才执行 adaptation
                    if self.enable_adaptation:
                        self.metrics["adaptation_points"].append(self.metrics["processed"])

                        # 触发适应逻辑
                        if len(self.buffer_features) >= 32:  # 至少有一个 batch
                            # 取最近的数据进行适应
                            window = 500
                            adapt_feats = torch.stack(self.buffer_features[-window:])
                            adapt_lbls = torch.stack(self.buffer_labels[-window:])

                            # 检查是否发现新类 (仅针对 multiclass)
                            new_class_detected = False
                            if self.task_mode == "multiclass":
                                max_lbl = adapt_lbls.max().item()
                                # 需要获取当前模型输出维度
                                classifier = getattr(self.model, self.adapter.classifier_layer_name)
                                if isinstance(classifier, torch.nn.Sequential):
                                    curr_dim = classifier[-1].out_features
                                else:
                                    curr_dim = classifier.out_features

                                if max_lbl >= curr_dim:
                                    new_class_detected = True
                                    logger.info(f"🆕 New class {max_lbl} detected! (Current dim: {curr_dim})")

                            self.adapter.adapt(adapt_feats, adapt_lbls, new_class_detected=new_class_detected)

                        # 重置检测器 (避免连续报警)
                        self.detector.reset()

                # 限制 buffer 大小
                if len(self.buffer_features) > 5000:
                    self.buffer_features = self.buffer_features[-2000:]
                    self.buffer_labels = self.buffer_labels[-2000:]

    def get_results(self):
        acc = np.mean(self.metrics["accuracy_history"]) if self.metrics["accuracy_history"] else 0.0
        return {
            "Mode": "Adaptive" if self.enable_adaptation else "Baseline",
            "Detector": self.detector_type.upper(),
            "Task": self.task_mode,
            "Drifts": self.metrics["drifts"],
            "Avg Accuracy": acc,
            "Adaptations": len(self.metrics["adaptation_points"])
        }


class MappingWrapper:
    def __init__(self, mappings, effective_columns):
        self.categorical_val2idx_mappings = mappings
        self.categorical_columns_effective = effective_columns


def resolve_path(raw_path, dataset_name):
    if raw_path is None: return ""
    if "${dataset.name}" in raw_path:
        raw_path = raw_path.replace("${dataset.name}", dataset_name)
    if raw_path.startswith("."):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        raw_path = os.path.normpath(os.path.join(project_root, raw_path))
    return raw_path


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # ==========================================================
    # 0. 实验配置
    # ==========================================================
    TASK_MODE = "binary"  # 'binary' 或 'multiclass'
    DETECTORS = ["bndm", "adwin"]  # 🔥 开启两种检测器循环

    logger.info(f"🚀 Starting Drift Experiment | Task: {TASK_MODE} | Detectors: {DETECTORS}")

    # 1. 修复配置路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "cic_ids_2017"
    dataset_yaml_path = os.path.join(current_dir, f"../models/flow_bert_multiview/config/datasets/{dataset_name}.yaml")

    if os.path.exists(dataset_yaml_path):
        dataset_cfg = OmegaConf.load(dataset_yaml_path)
        cfg.data.dataset = dataset_cfg.name
        if 'flow_data_path' in dataset_cfg:
            cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
        if 'session_split_path' in dataset_cfg:
            cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)
        if 'class_weights' in dataset_cfg:
            cfg.loss.class_weights = dataset_cfg.class_weights

    cfg.data.split_mode = "flow"

    # 2. 初始化数据
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")
    test_loader = dm.test_dataloader()

    _ = dm.train_dataloader()  # 触发 lazy init
    mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
    effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)
    dataset_wrapper = MappingWrapper(mappings, effective_columns)

    # 3. 确定模型路径
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = os.path.join(project_root, "processed_data", "best_model.ckpt")

    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(project_root, "checkpoints", "best_model.ckpt")

    if not os.path.exists(ckpt_path):
        logger.warning("⚠️ No checkpoint found! Running with random weights.")
    else:
        logger.info(f"✅ Checkpoint found: {ckpt_path}")

    results = []

    # ==========================================================
    # 🔄 循环遍历所有检测器
    # ==========================================================
    for detector_name in DETECTORS:
        logger.info(f"\n{'=' * 40}\n🚀 Experiment Group: {detector_name.upper()}\n{'=' * 40}")

        # 🧪 实验 A: Baseline (No Adaptation)
        logger.info(f"🧪 [Exp {detector_name.upper()}-Baseline] Running Baseline...")

        # 每次实验重新加载模型，确保状态重置
        if os.path.exists(ckpt_path):
            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
        else:
            model = FlowBertMultiview(cfg, dataset=dataset_wrapper)

        runner_a = DriftExperimentRunner(model, cfg, detector_type=detector_name, enable_adaptation=False,
                                         task_mode=TASK_MODE)
        runner_a.run_stream(test_loader)
        results.append(runner_a.get_results())
        del model, runner_a  # 清理显存

        # 🧪 实验 B: Adaptation (With Incremental Learning)
        logger.info(f"🧪 [Exp {detector_name.upper()}-Adaptive] Running Adaptation...")

        # 重新加载模型
        if os.path.exists(ckpt_path):
            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
        else:
            model = FlowBertMultiview(cfg, dataset=dataset_wrapper)

        runner_b = DriftExperimentRunner(model, cfg, detector_type=detector_name, enable_adaptation=True,
                                         task_mode=TASK_MODE)
        runner_b.run_stream(test_loader)
        results.append(runner_b.get_results())
        del model, runner_b  # 清理显存

    # ==========================================================
    # 📊 最终报告
    # ==========================================================
    df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print(f"📊 概念漂移与适应消融报告 (Task: {TASK_MODE})")
    print("=" * 80)
    if tabulate:
        print(df.to_markdown(index=False))
    else:
        print(df.to_string(index=False))
    print("=" * 80)


if __name__ == "__main__":
    main()