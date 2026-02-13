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
            if 'attack_family' in batch:
                labels = batch['attack_family']
            elif 'label' in batch:
                labels = batch['label']
            else:
                labels = batch['is_malicious']

                # 特征提取
            with torch.no_grad():
                outputs = self.model(batch)
                features = outputs['logits']
                preds = torch.argmax(features, dim=1)

            # 逐样本处理
            batch_size = features.shape[0]
            for i in range(batch_size):
                self.metrics["processed"] += 1
                feat = features[i]
                lbl = labels[i]
                pred = preds[i]

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

                    # 仅在有足够历史数据时打印准确率
                    recent_acc = 0.0
                    if len(self.metrics['accuracy_history']) > 200:
                        recent_acc = np.mean(self.metrics['accuracy_history'][-200:])

                    logger.info(f"🚨 Drift at idx {self.metrics['processed']} (Recent Acc: {recent_acc:.4f})")

                    # 适应
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


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    # =================================================================
    # 🔥 核心修复：手动加载 Dataset 配置
    # =================================================================
    OmegaConf.set_struct(cfg, False)  # 解锁配置

    # 检查 dataset 是否缺失
    if not hasattr(cfg, 'dataset') or cfg.dataset is None or 'flow_data_path' not in cfg.dataset:
        logger.warning("⚠️ Config 'dataset' incomplete! Attempting manual load of CIC-IDS-2017.")

        # 1. 定位 yaml 文件路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 路径: src/concept_drift_detect/../models/flow_bert_multiview/config/datasets/cic_ids_2017.yaml
        dataset_yaml_path = os.path.join(
            current_dir,
            "../models/flow_bert_multiview/config/datasets/cic_ids_2017.yaml"
        )

        if os.path.exists(dataset_yaml_path):
            # 2. 加载并合并
            dataset_cfg = OmegaConf.load(dataset_yaml_path)
            cfg.dataset = dataset_cfg
            logger.info(f"✅ Manually loaded dataset config from: {dataset_yaml_path}")

            # 3. 修复插值路径 (手动覆盖 data.flow_data_path)
            # 因为原始的 ${dataset.flow_data_path} 可能因为上下文丢失而失效
            # 我们直接把 dataset 里的值赋给 data 里的值
            if 'flow_data_path' in dataset_cfg:
                # 处理相对路径问题：如果 yaml 里是 processed_data/..., 即使正确也可能因为 cwd 问题找不到
                # 这里我们假设 config 里的路径是相对于项目根目录的
                raw_path = dataset_cfg.flow_data_path
                # 如果包含插值变量 ${dataset.name}，手动替换
                if "${dataset.name}" in raw_path:
                    raw_path = raw_path.replace("${dataset.name}", dataset_cfg.name)

                cfg.data.flow_data_path = raw_path
                logger.info(f"🔧 Patched cfg.data.flow_data_path = {cfg.data.flow_data_path}")
        else:
            logger.error(f"❌ Dataset config not found at: {dataset_yaml_path}")
            raise FileNotFoundError("Critical config missing")

    # 4. 强制设置 Flow 模式 (消融实验要求)
    cfg.data.split_mode = "flow"
    if hasattr(cfg.data, 'sampling'):
        cfg.data.sampling.random = False

    logger.info(
        f"🔧 Final Config: Dataset={cfg.dataset.get('name')}, Mode={cfg.data.split_mode}, Shuffle={cfg.data.sampling.random}")
    # =================================================================

    # 准备数据
    try:
        dm = MultiviewFlowDataModule(cfg)
        dm.setup(stage="test")
        test_loader = dm.test_dataloader()
    except Exception as e:
        logger.error(f"Failed to initialize DataModule: {e}")
        # 打印部分 Config 帮助调试
        logger.error(f"cfg.data.flow_data_path: {cfg.data.get('flow_data_path', 'MISSING')}")
        raise e

    # 加载模型
    ckpt_path = "checkpoints/best_model.ckpt"
    if not os.path.exists(ckpt_path):
        logger.warning(f"⚠️ Checkpoint {ckpt_path} not found. Using random weights!")
        model = FlowBertMultiview(cfg)
    else:
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg)

    results = []

    # 实验 1: BNDM (Proposed)
    logger.info("🧪 Starting Experiment 1: BNDM Detector")
    runner_bndm = DriftExperimentRunner(model, cfg, "bndm")
    runner_bndm.run_stream(test_loader)
    results.append(runner_bndm.get_results())

    # 实验 2: ADWIN (Baseline)
    logger.info("🧪 Starting Experiment 2: ADWIN Detector")
    # 重置模型
    if os.path.exists(ckpt_path):
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg)
    else:
        model = FlowBertMultiview(cfg)

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