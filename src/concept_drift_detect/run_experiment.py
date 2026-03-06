import sys
import os
import argparse
import glob

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import hydra
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, average_precision_score
)

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# 引入项目模块
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector, KSWINDetector
from src.concept_drift_detect.adapter import IncrementalAdapter

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(filename)s:%(lineno)d] [%(levelname)s] %(message)s')
logger = logging.getLogger("DriftExp")


class DriftExperimentRunner:
    def __init__(self, model, cfg, detector_type="adwin", enable_adaptation=True):
        self.model = model
        self.cfg = cfg
        self.detector_type = detector_type
        self.enable_adaptation = enable_adaptation

        # 🟢 动态从 YAML 读取检测器配置
        cd_cfg = cfg.get("concept_drift", {})
        self.det_config = {'seed': cd_cfg.get('seed', 2026)}

        if detector_type in cd_cfg.get("detectors", {}):
            algo_cfg = OmegaConf.to_container(cd_cfg.detectors[detector_type], resolve=True)
            self.det_config.update(algo_cfg)

        self.det_config.setdefault('threshold', 0.01)
        self.det_config.setdefault('max_level', 6)
        self.det_config.setdefault('window_size', 1000)
        self.det_config.setdefault('delta', 0.002)

        # 新增 KSWIN 判别
        if detector_type == "bndm":
            self.detector = BNDMDetector(self.det_config)
        elif detector_type == "adwin":
            self.detector = ADWINDetector(self.det_config)
        elif detector_type == "kswin":
            self.detector = KSWINDetector(self.det_config)
        else:
            raise ValueError(f"Unknown detector: {detector_type}")

        # 🟢 动态从 YAML 读取 Adapter 配置
        if cd_cfg.get("adaptation"):
            self.adapt_config = OmegaConf.to_container(cd_cfg.adaptation, resolve=True)
        else:
            self.adapt_config = {
                'lr': 1e-4, 'epochs': 3, 'buffer_size': 2000, 'batch_size': 32, 'window': 500
            }

        # 传入 Joint Adapter
        self.adapter = IncrementalAdapter(model, self.adapt_config)

        self.metrics = {
            "processed": 0,
            "drifts": 0,
            "bin_preds": [],
            "bin_labels": [],
            "bin_probs": [],  # 保存概率用于算 ROC-AUC 和 AP
            "mul_preds": [],
            "mul_labels": [],
            "adaptation_points": []
        }

        self.buffer_features = []
        self.buffer_bin_labels = []
        self.buffer_mul_labels = []

    def run_stream(self, dataloader):
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        status_desc = f"[{'Adapt' if self.enable_adaptation else 'Fixed'}] {self.detector_type.upper()} (JointEval)"
        progress = tqdm(dataloader, desc=status_desc)

        for batch in progress:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            # 提取 Ground Truth
            lbl_bin = batch['is_malicious_label'].view(-1)

            if 'attack_family_label' in batch:
                lbl_mul = batch['attack_family_label']
                if lbl_mul.dim() > 1:
                    lbl_mul = torch.argmax(lbl_mul, dim=1)
            else:
                lbl_mul = torch.zeros_like(lbl_bin)

            # 模型推理
            with torch.no_grad():
                outputs = self.model(batch)
                features = outputs['multiview_embeddings']

                # 获取二分类预测
                logits_bin = outputs['is_malicious_cls_logits']
                probs_bin = torch.sigmoid(logits_bin).squeeze(-1)
                preds_bin = (probs_bin > 0.5).long()

                # 获取多分类预测
                if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                    logits_mul = outputs['attack_family_cls_logits']
                    preds_mul = torch.argmax(logits_mul, dim=1)
                else:
                    preds_mul = torch.zeros_like(preds_bin)

            batch_size = features.shape[0]

            # 收集全局评估数据
            self.metrics["processed"] += batch_size
            self.metrics["bin_labels"].extend(lbl_bin.cpu().tolist())
            self.metrics["bin_preds"].extend(preds_bin.cpu().tolist())
            self.metrics["bin_probs"].extend(probs_bin.cpu().tolist())
            self.metrics["mul_labels"].extend(lbl_mul.cpu().tolist())
            self.metrics["mul_preds"].extend(preds_mul.cpu().tolist())

            # 实时进度条显示二分类准确率
            if len(self.metrics['bin_labels']) > 0:
                current_acc = np.mean(
                    np.array(self.metrics['bin_preds'][-1000:]) == np.array(self.metrics['bin_labels'][-1000:]))
                progress.set_postfix({"Bin_Acc": f"{current_acc:.4f}"})

            # 在线漂移检测
            for i in range(batch_size):
                feat = features[i]
                feat_input = feat.unsqueeze(0)
                drift_detected = self.detector.update(self.detector.preprocess(feat_input))

                # 记录联合 Buffer
                self.buffer_features.append(feat)
                self.buffer_bin_labels.append(lbl_bin[i])
                self.buffer_mul_labels.append(lbl_mul[i])

                if drift_detected:
                    self.metrics["drifts"] += 1

                    if self.enable_adaptation:
                        self.metrics["adaptation_points"].append(self.metrics["processed"])

                        if len(self.buffer_features) >= 32:
                            window = self.adapt_config.get('window', 500)
                            adapt_feats = torch.stack(self.buffer_features[-window:])
                            adapt_bin = torch.stack(self.buffer_bin_labels[-window:])
                            adapt_mul = torch.stack(self.buffer_mul_labels[-window:])

                            # 触发联合微调
                            self.adapter.adapt(adapt_feats, adapt_bin, adapt_mul)

                        self.detector.reset()

                # 限制内存
                buffer_limit = self.adapt_config.get('buffer_size', 2000)
                if len(self.buffer_features) > buffer_limit:
                    self.buffer_features = self.buffer_features[-buffer_limit:]
                    self.buffer_bin_labels = self.buffer_bin_labels[-buffer_limit:]
                    self.buffer_mul_labels = self.buffer_mul_labels[-buffer_limit:]

    # 🔴 新增：像截图中一样的详细报告打印
    def print_detailed_report(self, target_names_mul=None):
        y_true_bin = np.array(self.metrics["bin_labels"])
        y_pred_bin = np.array(self.metrics["bin_preds"])
        y_prob_bin = np.array(self.metrics["bin_probs"])
        y_true_mul = np.array(self.metrics["mul_labels"])
        y_pred_mul = np.array(self.metrics["mul_preds"])

        logger.info("\n" + "=" * 80)
        logger.info(
            f"📊 REPORT FOR: [{'Adaptive' if self.enable_adaptation else 'Baseline'}] {self.detector_type.upper()}")
        logger.info("=" * 80)

        # 1. 二分类任务报告
        logger.info("📋 is_malicious任务的详细分类报告:")
        bin_report = classification_report(y_true_bin, y_pred_bin, target_names=["正常", "恶意"], digits=4,
                                           zero_division=0)
        print(bin_report)

        logger.info("🎯 is_malicious任务的混淆矩阵:")
        cm = confusion_matrix(y_true_bin, y_pred_bin)
        print(cm)

        logger.info("📈 is_malicious任务的高级指标:")
        try:
            roc_auc = roc_auc_score(y_true_bin, y_prob_bin)
            ap = average_precision_score(y_true_bin, y_prob_bin)
            logger.info(f"ROC-AUC: {roc_auc:.4f}")
            logger.info(f"Average Precision: {ap:.4f}")
        except Exception as e:
            logger.info(f"高级指标计算失败: {e}")

        total_num = len(y_true_bin)
        pos_num = int(sum(y_true_bin))
        neg_num = total_num - pos_num
        logger.info("📊 样本的is_malicious标签数量统计:")
        logger.info(f"总样本数: {total_num}")
        logger.info(f"正样本数: {pos_num}.0")
        logger.info(f"负样本数: {neg_num}.0")
        logger.info(f"正样本比例: {(pos_num / total_num * 100) if total_num > 0 else 0:.2f}%")
        logger.info("================================================================================")

        # 2. 多分类任务报告（只看恶意样本）
        logger.info("🤖 attack_family 任务测试报告（简要）")
        malicious_mask = (y_true_bin == 1)
        if malicious_mask.sum() > 0:
            y_true_mul_mal = y_true_mul[malicious_mask]
            y_pred_mul_mal = y_pred_mul[malicious_mask]

            mac_f1 = f1_score(y_true_mul_mal, y_pred_mul_mal, average='macro', zero_division=0)
            mic_f1 = f1_score(y_true_mul_mal, y_pred_mul_mal, average='micro', zero_division=0)
            logger.info(f"macro_f1={mac_f1:.4f}, micro_f1={mic_f1:.4f}")

            logger.info("📄 attack_family 任务的分类报告:")
            labels_present = sorted(list(set(y_true_mul_mal) | set(y_pred_mul_mal)))

            # 解析目标名字映射
            t_names = None
            if target_names_mul is not None:
                t_names = [target_names_mul[i] if i < len(target_names_mul) else f"Class_{i}" for i in labels_present]

            mul_report = classification_report(y_true_mul_mal, y_pred_mul_mal, labels=labels_present,
                                               target_names=t_names, digits=4, zero_division=0)
            print(mul_report)

            logger.info("📊 attack_family per-class F1:")
            f1s = f1_score(y_true_mul_mal, y_pred_mul_mal, labels=labels_present, average=None, zero_division=0)
            for i, f1_v in zip(labels_present, f1s):
                c_name = target_names_mul[i] if target_names_mul and i < len(target_names_mul) else f"Class_{i}"
                logger.info(f"{c_name:<15}: F1={f1_v:.4f}")
        else:
            logger.info("⚠️ 无恶意样本，跳过 attack_family 的多分类评估。")

        logger.info("================================================================================\n")

    def get_results(self):
        res = {
            "Mode": "Adaptive" if self.enable_adaptation else "Baseline",
            "Detector": self.detector_type.upper(),
            "Drifts": self.metrics["drifts"],
            "Adaptations": len(self.metrics["adaptation_points"])
        }

        # 计算二分类指标
        if self.metrics["bin_labels"]:
            y_true_bin = np.array(self.metrics["bin_labels"])
            y_pred_bin = np.array(self.metrics["bin_preds"])
            res["Bin_Acc"] = accuracy_score(y_true_bin, y_pred_bin)
            res["Bin_Pre"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            res["Bin_Rec"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            res["Bin_F1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

        # 计算多分类指标
        if self.metrics["mul_labels"]:
            y_true_mul = np.array(self.metrics["mul_labels"])
            y_pred_mul = np.array(self.metrics["mul_preds"])

            malicious_mask = (np.array(self.metrics["bin_labels"]) == 1)
            if malicious_mask.sum() > 0:
                y_true_mul_mal = y_true_mul[malicious_mask]
                y_pred_mul_mal = y_pred_mul[malicious_mask]
                res["Mul_MacF1"] = f1_score(y_true_mul_mal, y_pred_mul_mal, average="macro", zero_division=0)
                res["Mul_MicF1"] = f1_score(y_true_mul_mal, y_pred_mul_mal, average="micro", zero_division=0)
            else:
                res["Mul_MacF1"] = 0.0
                res["Mul_MicF1"] = 0.0

        return res


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
            torch.load(ckpt, map_location="cpu")
            valid_ckpt = ckpt
            break
        except Exception:
            pass
    return valid_ckpt


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # 🔴 新增 kswin 到对比实验组中
    DETECTORS = ["bndm", "adwin", "kswin"]

    logger.info(f"🚀 Starting Joint Experiment | Detectors: {DETECTORS}")

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
            if 'loss' not in cfg: cfg.loss = {}
            cfg.loss.class_weights = dataset_cfg.class_weights

    cfg.data.split_mode = "flow"
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

    # 🔴 获取多分类标签的真实字符串名，用于漂亮的打印
    target_names_mul = None
    if mappings and 'label' in mappings:
        # 根据 index 对 label 名字进行排序，确保与打印时的 index 0,1,2... 对应
        sorted_mappings = sorted(mappings['label'].items(), key=lambda x: x[1])
        target_names_mul = [k for k, v in sorted_mappings]

    dataset_wrapper = MappingWrapper(mappings, effective_columns)
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = find_valid_checkpoint(project_root)

    if not ckpt_path:
        logger.error("❌ CRITICAL: No valid checkpoint found!")
        return

    results = []

    for detector_name in DETECTORS:
        logger.info(f"\n{'=' * 40}\n🚀 Experiment Group: {detector_name.upper()}\n{'=' * 40}")

        # ======== Baseline ========
        try:
            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
            runner_a = DriftExperimentRunner(model, cfg, detector_type=detector_name, enable_adaptation=False)
            runner_a.run_stream(test_loader)

            # 打印详细报告
            runner_a.print_detailed_report(target_names_mul)

            results.append(runner_a.get_results())
            del model, runner_a
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ Failed during Baseline experiment: {e}")

        # ======== Adaptation ========
        try:
            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
            runner_b = DriftExperimentRunner(model, cfg, detector_type=detector_name, enable_adaptation=True)
            runner_b.run_stream(test_loader)

            # 打印详细报告
            runner_b.print_detailed_report(target_names_mul)

            results.append(runner_b.get_results())
            del model, runner_b
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"❌ Failed during Adaptive experiment: {e}")

    # ==================================
    # 打印全局消融报告总结
    # ==================================
    if results:
        df = pd.DataFrame(results)

        columns_order = ['Mode', 'Detector', 'Drifts', 'Bin_Acc', 'Bin_F1', 'Mul_MacF1', 'Mul_MicF1', 'Adaptations']
        df = df[[c for c in columns_order if c in df.columns]]

        print("\n" + "=" * 100)
        print(f"📊 概念漂移与适应消融报告 (Joint Evaluation)")
        print("=" * 100)
        if tabulate:
            print(df.to_markdown(index=False, floatfmt=".4f"))
        else:
            print(df.to_string(index=False, float_format="%.4f"))
        print("=" * 100)


if __name__ == "__main__":
    main()