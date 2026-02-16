import sys
import os
import copy
import argparse
import glob

# 移除绘图库导入
# import matplotlib.pyplot as plt
# import seaborn as sns

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
        self.task_mode = task_mode

        # 检测器配置
        self.det_config = {
            'seed': 2026,
            'threshold': 0.01,  # BNDM 敏感度
            'max_level': 6,  # BNDM 树深度
            'window_size': 1000,
            'delta': 0.002  # ADWIN 敏感度
        }

        if detector_type == "bndm":
            self.detector = BNDMDetector(self.det_config)
        elif detector_type == "adwin":
            self.detector = ADWINDetector(self.det_config)
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

        # 适应器配置
        self.adapt_config = {
            'lr': 1e-4,
            'epochs': 3,
            'buffer_size': 2000,
            'batch_size': 32
        }
        self.adapter = IncrementalAdapter(model, self.adapt_config, task_mode=task_mode)

        # 数据记录
        self.history = {
            "accuracy": []
        }

        self.metrics = {
            "processed": 0,
            "drifts": 0,
            "adaptation_points": []
        }

        # 缓冲区
        self.buffer_features = []
        self.buffer_labels = []

    def run_stream(self, dataloader):
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        status_desc = f"[{'Adapt' if self.enable_adaptation else 'Fixed'}] {self.detector_type.upper()}"
        progress = tqdm(dataloader, desc=status_desc)

        batch_idx = 0

        for batch in progress:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 1. 确定标签
            if self.task_mode == "multiclass":
                labels = batch.get('attack_family_label', batch['is_malicious_label'])
            else:
                labels = batch['is_malicious_label']

            # 2. 模型推理
            with torch.no_grad():
                outputs = self.model(batch)
                features = outputs['multiview_embeddings']

                if self.task_mode == "multiclass":
                    if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                        preds = torch.argmax(outputs['attack_family_cls_logits'], dim=1)
                    else:
                        preds = (torch.sigmoid(outputs['is_malicious_cls_logits']) > 0.5).long().squeeze()
                else:  # binary
                    preds = (torch.sigmoid(outputs['is_malicious_cls_logits']) > 0.5).long().squeeze()

            # 3. 记录 Batch 级指标
            batch_size = features.shape[0]
            if labels.dim() > 1:
                labels_idx = torch.argmax(labels, dim=1)
            else:
                labels_idx = labels

            if preds.shape != labels_idx.shape:
                preds = preds.view_as(labels_idx)

            acc = (preds == labels_idx).float().mean().item()

            # 保存精度历史
            self.history["accuracy"].append(acc)

            self.metrics["processed"] += batch_size
            progress.set_postfix({"Acc": f"{np.mean(self.history['accuracy'][-100:]):.4f}"})

            # 4. 漂移检测 (逐样本)
            for i in range(batch_size):
                feat = features[i]

                # 预处理
                feat_input = feat.unsqueeze(0)
                val = self.detector.preprocess(feat_input)

                # 更新检测器
                is_drift = self.detector.update(val)

                # 存入 buffer
                self.buffer_features.append(feat)
                self.buffer_labels.append(labels_idx[i])

                if is_drift:
                    self.metrics["drifts"] += 1

                    if self.enable_adaptation:
                        self.metrics["adaptation_points"].append(self.metrics["processed"])

                        if len(self.buffer_features) >= 32:
                            window = 500
                            adapt_feats = torch.stack(self.buffer_features[-window:])
                            adapt_lbls = torch.stack(self.buffer_labels[-window:])

                            new_class_detected = False
                            # ... (new class detection logic if needed) ...

                            self.adapter.adapt(adapt_feats, adapt_lbls, new_class_detected=new_class_detected)

                    # 🟢 无论是否 Adapt，都要重置检测器，防止 Baseline 模式下持续报警
                    self.detector.reset()

            # 限制 buffer
            if len(self.buffer_features) > 5000:
                self.buffer_features = self.buffer_features[-2000:]
                self.buffer_labels = self.buffer_labels[-2000:]

            batch_idx += 1

    def get_results(self):
        # 获取详细参数
        param_info = ""
        window_info = ""

        if self.detector_type == "bndm":
            param_info = f"Th={self.det_config['threshold']}"
            window_info = f"Win={self.det_config['window_size']}"
        elif self.detector_type == "adwin":
            param_info = f"Delta={self.det_config['delta']}"
            window_info = "Dynamic"

        return {
            "Mode": "Adaptive" if self.enable_adaptation else "Baseline",
            "Detector": self.detector_type.upper(),
            "Task": self.task_mode,
            "Drifts": self.metrics["drifts"],
            "Avg Accuracy": np.mean(self.history["accuracy"]) if self.history["accuracy"] else 0.0,
            "Adaptations": len(self.metrics["adaptation_points"]),
            "Param": param_info,
            "Window": window_info
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


def find_valid_checkpoint(project_root):
    processed_dir = os.path.join(project_root, "processed_data")
    checkpoints_dir = os.path.join(project_root, "checkpoints")

    candidates = [
        os.path.join(processed_dir, "best_model.ckpt"),
        os.path.join(processed_dir, "last.ckpt"),
        os.path.join(checkpoints_dir, "best_model.ckpt"),
        os.path.join(checkpoints_dir, "last.ckpt"),
    ]
    if os.path.exists(processed_dir):
        extra_ckpts = glob.glob(os.path.join(processed_dir, "*.ckpt"))
        extra_ckpts.sort(key=os.path.getmtime, reverse=True)
        candidates.extend(extra_ckpts)

    valid_ckpt = None
    for ckpt in candidates:
        if not os.path.exists(ckpt): continue
        try:
            logger.info(f"🔍 Checking checkpoint: {ckpt}")
            torch.load(ckpt, map_location="cpu")
            valid_ckpt = ckpt
            logger.info(f"✅ Found valid checkpoint: {valid_ckpt}")
            break
        except Exception as e:
            logger.warning(f"⚠️ Corrupted or invalid checkpoint at {ckpt}: {e}")

    return valid_ckpt


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    TASK_MODE = "binary"

    logger.info(f"🚀 Starting Drift Comparison Experiment")

    # ==========================================================
    # 🟢 1. 修复配置路径
    # ==========================================================
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "cic_ids_2017"
    dataset_yaml_path = os.path.join(current_dir, f"../models/flow_bert_multiview/config/datasets/{dataset_name}.yaml")
    dataset_yaml_path = os.path.abspath(dataset_yaml_path)

    if os.path.exists(dataset_yaml_path):
        logger.info(f"Loading dataset config from: {dataset_yaml_path}")
        dataset_cfg = OmegaConf.load(dataset_yaml_path)

        # ⚠️ 关键：设置 dataset.name，解决 InterpolationKeyError
        cfg.data.dataset = dataset_cfg.name

        # 显式解析路径，避免 Hydra 延迟解析时出错
        if 'flow_data_path' in dataset_cfg:
            cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
        if 'session_split_path' in dataset_cfg:
            cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)

        if 'class_weights' in dataset_cfg:
            if 'loss' not in cfg: cfg.loss = {}
            cfg.loss.class_weights = dataset_cfg.class_weights
    else:
        logger.warning(f"Dataset config not found at {dataset_yaml_path}. Assuming cfg is already complete.")

    # 强制流式切分
    cfg.data.split_mode = "flow"

    # ==========================================================
    # 2. 初始化数据
    # ==========================================================
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")
    test_loader = dm.test_dataloader()

    if hasattr(dm, 'train_dataset') and dm.train_dataset:
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)
    else:
        _ = dm.train_dataloader()
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)

    dataset_wrapper = MappingWrapper(mappings, effective_columns)

    # 3. 加载模型
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = find_valid_checkpoint(project_root)

    if not ckpt_path:
        logger.error("No valid checkpoint found!")
        return

    # 4. 运行实验
    all_results = []

    # 定义要运行的实验组合
    experiments = [
        ("bndm", False),  # BNDM Baseline
        ("bndm", True),  # BNDM Adaptive
        ("adwin", False),  # ADWIN Baseline
        ("adwin", True),  # ADWIN Adaptive
    ]

    for detector_name, enable_adapt in experiments:
        mode_str = "Adaptive" if enable_adapt else "Baseline"
        logger.info(f"🧪 Running {detector_name.upper()} ({mode_str})...")

        # 重新加载模型以确保起点一致
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)

        runner = DriftExperimentRunner(model, cfg, detector_type=detector_name, enable_adaptation=enable_adapt,
                                       task_mode=TASK_MODE)
        runner.run_stream(test_loader)

        res = runner.get_results()
        all_results.append(res)
        logger.info(f"Result: {res}")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # 5. 生成详细表格报告
    logger.info("📊 Generating Report...")
    if all_results:
        df = pd.DataFrame(all_results)
        print("\n" + "=" * 100)
        print(f"📊 概念漂移与适应消融详细报告 (Task: {TASK_MODE})")
        print("=" * 100)

        if tabulate:
            # 使用 tabulate 生成美观的表格
            print(tabulate(df, headers='keys', tablefmt='pipe', floatfmt=".4f", showindex=False))
        else:
            print(df.to_string(index=False))

        print("=" * 100)

        # 保存 CSV 备份
        csv_path = os.path.join(project_root, "drift_experiment_results.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"📝 结果已保存至: {csv_path}")


if __name__ == "__main__":
    main()