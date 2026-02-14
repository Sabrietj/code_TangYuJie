import sys
import os

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import hydra
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# 引入项目模块
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector
from src.concept_drift_detect.adapter import IncrementalAdapter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DriftExp")


class DriftExperimentRunner:
    def __init__(self, model, cfg, detector_type="bndm"):
        self.model = model
        self.cfg = cfg
        self.detector_type = detector_type

        # 检测器参数配置
        det_config = {
            'seed': 2026,
            'threshold': 0.05,
            'max_level': 6,
            'window_size': 2000,
            'delta': 0.002
        }

        if detector_type == "bndm":
            self.detector = BNDMDetector(det_config)
        elif detector_type == "adwin":
            self.detector = ADWINDetector(det_config)

        # 适应器参数配置
        adapt_config = {'lr': 1e-4, 'epochs': 5, 'buffer_size': 5000}
        self.adapter = IncrementalAdapter(model, adapt_config)

        self.metrics = {
            "processed": 0,
            "drifts": 0,
            "accuracy_history": [],
            "adaptation_points": []
        }

        self.buffer_features = []
        self.buffer_labels = []

    def run_stream(self, dataloader):
        self.model.eval()
        self.model.cuda()

        progress = tqdm(dataloader, desc=f"Running {self.detector_type.upper()}")

        for batch in progress:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 标签兼容性处理
            if 'attack_family_label' in batch:
                labels = batch['attack_family_label']
            elif 'label' in batch:
                labels = batch['label']
            else:
                labels = batch['is_malicious_label']

                # 特征提取
            with torch.no_grad():
                outputs = self.model(batch)
                # 使用多视图融合后的特征进行漂移检测
                features = outputs['multiview_embeddings']

                # 获取预测结果
                if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                    preds = torch.argmax(outputs['attack_family_cls_logits'], dim=1)
                else:
                    preds = (torch.sigmoid(outputs['is_malicious_cls_logits']) > 0.5).long().squeeze()

            # 逐样本处理
            batch_size = features.shape[0]
            for i in range(batch_size):
                self.metrics["processed"] += 1
                feat = features[i]

                # 处理多标签/维度不匹配情况
                if labels.dim() > 1:
                    lbl = torch.argmax(labels[i])
                else:
                    lbl = labels[i]

                if preds.dim() > 0:
                    pred = preds[i]
                else:
                    pred = preds

                is_correct = (pred == lbl).item()
                self.metrics["accuracy_history"].append(is_correct)

                # 漂移检测
                feat_input = feat.unsqueeze(0)
                drift_detected = self.detector.update(self.detector.preprocess(feat_input))

                self.buffer_features.append(feat)
                self.buffer_labels.append(lbl)

                # 触发适应
                if drift_detected:
                    self.metrics["drifts"] += 1
                    self.metrics["adaptation_points"].append(self.metrics["processed"])

                    recent_acc = 0.0
                    if len(self.metrics['accuracy_history']) > 200:
                        recent_acc = np.mean(self.metrics['accuracy_history'][-200:])

                    logger.info(f"🚨 Drift at idx {self.metrics['processed']} (Recent Acc: {recent_acc:.4f})")

                    # 只有当 buffer 足够大时才进行适应
                    if len(self.buffer_features) > 100:
                        adapt_feats = torch.stack(self.buffer_features[-1000:])
                        adapt_lbls = torch.stack(self.buffer_labels[-1000:])

                        self.adapter.adapt(adapt_feats, adapt_lbls)

                    self.detector.reset()

                    if len(self.buffer_features) > 5000:
                        self.buffer_features = self.buffer_features[-2000:]
                        self.buffer_labels = self.buffer_labels[-2000:]

    def get_results(self):
        acc = np.mean(self.metrics["accuracy_history"]) if self.metrics["accuracy_history"] else 0.0
        return {
            "Method": self.detector_type.upper(),
            "Total Samples": self.metrics["processed"],
            "Drifts Detected": self.metrics["drifts"],
            "Avg Accuracy": acc
        }


def resolve_path(raw_path, dataset_name):
    """辅助函数：处理路径插值和相对路径"""
    if raw_path is None: return ""
    if "${dataset.name}" in raw_path:
        raw_path = raw_path.replace("${dataset.name}", dataset_name)
    if raw_path.startswith("."):
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        raw_path = os.path.normpath(os.path.join(project_root, raw_path))
    return raw_path


# ==========================================================
# 🌟 [关键修复] 完整的 Dataset 包装器
# 必须包含 categorical_val2idx_mappings 和 categorical_columns_effective
# ==========================================================
class MappingWrapper:
    def __init__(self, mappings, effective_columns):
        # 1. 类别映射字典
        self.categorical_val2idx_mappings = mappings
        # 2. 有效类别列列表 (修复 AttributeError 的关键)
        self.categorical_columns_effective = effective_columns

        # 调试日志：确认属性已设置
        # print(f"DEBUG: MappingWrapper initialized with {len(effective_columns)} columns")


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    # =================================================================
    # 🔥 配置手动修补
    # =================================================================
    OmegaConf.set_struct(cfg, False)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml_path = os.path.join(
        current_dir,
        "../models/flow_bert_multiview/config/datasets/cic_ids_2017.yaml"
    )

    if os.path.exists(dataset_yaml_path):
        logger.info(f"✅ Loading dataset config: {dataset_yaml_path}")
        dataset_cfg = OmegaConf.load(dataset_yaml_path)

        cfg.data.dataset = dataset_cfg.name

        if 'flow_data_path' in dataset_cfg:
            fixed_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
            cfg.data.flow_data_path = fixed_path
            logger.info(f"🔧 Patched flow_data_path: {fixed_path}")

        if 'session_split_path' in dataset_cfg:
            fixed_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)
            cfg.data.session_split.session_split_path = fixed_path
            logger.info(f"🔧 Patched session_split_path: {fixed_path}")

        if 'class_weights' in dataset_cfg:
            cfg.loss.class_weights = dataset_cfg.class_weights

    else:
        raise FileNotFoundError(f"Critical: Dataset config not found at {dataset_yaml_path}")

    # 强制设置 Flow 模式
    cfg.data.split_mode = "flow"
    if hasattr(cfg.data, 'sampling'):
        cfg.data.sampling.random = False

    logger.info("🔧 Config patching complete. Initializing DataModule...")

    # 准备数据
    try:
        dm = MultiviewFlowDataModule(cfg)
        # 使用 "fit" 阶段初始化，确保 train_dataset 被创建
        dm.setup(stage="fit")
        test_loader = dm.test_dataloader()

        # =================================================================
        # 🌟 [关键修复] 数据元提取逻辑
        # =================================================================
        mappings = None
        effective_columns = None

        # 1. 尝试从 train_dataset 获取
        if hasattr(dm, 'train_dataset'):
            mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
            effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)

        # 2. 如果失败，尝试 lazy init
        if mappings is None or effective_columns is None:
            logger.info("Triggering lazy initialization for metadata...")
            _ = dm.train_dataloader()
            if hasattr(dm, 'train_dataset'):
                mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
                effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)

        # 3. 最终检查
        if mappings is None or effective_columns is None:
            raise AttributeError(
                "❌ FAILED to extract 'categorical_val2idx_mappings' or 'categorical_columns_effective' from DataModule/Dataset. Please check Dataset implementation.")

        # 4. 创建增强版 Wrapper
        dataset_wrapper = MappingWrapper(mappings, effective_columns)
        logger.info(f"✅ Metadata extracted. Effective Columns: {len(effective_columns)}, Mappings: {len(mappings)}")

    except Exception as e:
        logger.error(f"❌ DataModule init failed: {e}")
        raise e

    # 加载模型
    ckpt_path = "checkpoints/best_model.ckpt"
    if not os.path.exists(ckpt_path):
        logger.warning(f"⚠️ Checkpoint {ckpt_path} not found. Using random weights!")
        # ✅ 修复点: 传入增强版 wrapper
        model = FlowBertMultiview(cfg, dataset=dataset_wrapper)
    else:
        # ✅ 修复点: 传入增强版 wrapper
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)

    results = []

    # 实验 1: BNDM
    logger.info("🧪 Starting Experiment 1: BNDM Detector")
    runner_bndm = DriftExperimentRunner(model, cfg, "bndm")
    runner_bndm.run_stream(test_loader)
    results.append(runner_bndm.get_results())

    # 实验 2: ADWIN
    logger.info("🧪 Starting Experiment 2: ADWIN Detector")
    # 重置模型
    if os.path.exists(ckpt_path):
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
    else:
        model = FlowBertMultiview(cfg, dataset=dataset_wrapper)

    runner_adwin = DriftExperimentRunner(model, cfg, "adwin")
    runner_adwin.run_stream(test_loader)
    results.append(runner_adwin.get_results())

    df = pd.DataFrame(results)
    print("\n" + "=" * 60)
    print("📊 概念漂移与适应 - 消融实验报告")
    print("=" * 60)
    print(df.to_markdown(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()