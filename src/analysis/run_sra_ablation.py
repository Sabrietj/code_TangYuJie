import sys
import os
import time
import logging
import copy
import numpy as np
import pandas as pd
import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector
from src.concept_drift_detect.adapter import IncrementalAdapter
from src.concept_drift_detect.run_experiment import MappingWrapper, find_valid_checkpoint, resolve_path

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("SRA_Ablation")


def evaluate_baseline_on_val(model, val_loader):
    """
    专门在验证集上评估初始模型的基线性能（作为未经流式数据污染的起跑线参考）
    """
    model.eval()
    if torch.cuda.is_available():
        model.cuda()

    all_bin_preds, all_bin_labels = [], []
    all_mul_preds, all_mul_labels = [], []

    progress = tqdm(val_loader, desc="Evaluating Baseline on Validation Set")
    for batch in progress:
        batch_device = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        lbl_bin = batch_device['is_malicious_label'].view(-1)
        lbl_mul = batch_device.get('attack_family_label', torch.zeros_like(lbl_bin))
        if lbl_mul.dim() > 1: lbl_mul = torch.argmax(lbl_mul, dim=1)

        with torch.no_grad():
            outputs = model(batch_device)
            logits_bin = outputs['is_malicious_cls_logits']
            preds_bin = (torch.sigmoid(logits_bin).squeeze(-1) > 0.5).long()

            if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                preds_mul = torch.argmax(outputs['attack_family_cls_logits'], dim=1)
            else:
                preds_mul = torch.zeros_like(preds_bin)

        all_bin_preds.extend(preds_bin.cpu().tolist())
        all_bin_labels.extend(lbl_bin.cpu().tolist())
        all_mul_preds.extend(preds_mul.cpu().tolist())
        all_mul_labels.extend(lbl_mul.cpu().tolist())

    p = precision_score(all_bin_labels, all_bin_preds, zero_division=0)
    r = recall_score(all_bin_labels, all_bin_preds, zero_division=0)
    f1 = f1_score(all_bin_labels, all_bin_preds, zero_division=0)

    # 多分类评估 (仅针对恶意样本)
    y_true_bin = np.array(all_bin_labels)
    y_true_mul = np.array(all_mul_labels)
    y_pred_mul = np.array(all_mul_preds)
    malicious_mask = (y_true_bin == 1)

    if malicious_mask.sum() > 0:
        mac_f1 = f1_score(y_true_mul[malicious_mask], y_pred_mul[malicious_mask], average='macro', zero_division=0)
        mic_f1 = f1_score(y_true_mul[malicious_mask], y_pred_mul[malicious_mask], average='micro', zero_division=0)
    else:
        mac_f1, mic_f1 = 0.0, 0.0

    return {
        "Strategy": "Baseline (Val Set)",
        "Precision": p,
        "Recall": r,
        "F1_Score": f1,
        "Mul_MacF1": mac_f1,
        "Mul_MicF1": mic_f1,
        "Adaptation_Time(s)": 0.0,
        "Drift_Count": 0
    }


class SRAExperimentRunner:
    def __init__(self, model, cfg, strategy="A"):
        """
        strategy: 'S' (Static on Test), 'R' (Vanilla Retrain on Test), 'A' (Adaptation on Test)
        """
        self.model = model
        self.cfg = cfg
        self.strategy = strategy

        # 1. 初始化 BNDM 漂移检测器 (与 run_experiment.py 对齐)
        cd_cfg = cfg.get("concept_drift", {})
        self.det_config = {'seed': cd_cfg.get('seed', 2026), 'threshold': 0.01, 'max_level': 6, 'window_size': 1000}
        if "detectors" in cd_cfg and "bndm" in cd_cfg.detectors:
            algo_cfg = OmegaConf.to_container(cd_cfg.detectors["bndm"], resolve=True)
            self.det_config.update(algo_cfg)

        self.detector = BNDMDetector(self.det_config)

        # 2. 根据不同策略配置 Adapter
        adapt_config = OmegaConf.to_container(cd_cfg.get("adaptation", {}), resolve=True) if cd_cfg.get(
            "adaptation") else {}
        if self.strategy == "R":
            # 🔴 R 策略：传统 Vanilla 重训。关闭所有增量约束
            adapt_config['buffer_size'] = 20000
            adapt_config['epochs'] = 5
            adapt_config['window'] = 20000
            adapt_config['batch_size'] = 128
            adapt_config['lambda_topo'] = 0.0
            adapt_config['lambda_cons'] = 0.0
            adapt_config['lambda_proto'] = 0.0
        else:
            # 🟢 A 策略：完美继承 yaml 中的配置，仅确保基础窗口大小符合流式场景
            adapt_config['buffer_size'] = adapt_config.get('buffer_size', 2000)
            adapt_config['epochs'] = adapt_config.get('epochs', 3)
            adapt_config['window'] = adapt_config.get('window', 500)
            adapt_config['batch_size'] = adapt_config.get('batch_size', 32)
            # lambda_topo 等各种防遗忘超参数完好保留

        self.adapter = IncrementalAdapter(model, adapt_config)

        # 3. 流式评估指标记录器
        self.eval_window_size = 2000
        self.current_window_preds = []
        self.current_window_labels = []

        # 全局记录器
        self.stream_all_preds = []
        self.stream_all_labels = []
        self.stream_all_mul_preds = []
        self.stream_all_mul_labels = []

        self.f1_history = []
        self.drift_points = []
        self.time_cost = 0.0
        self.drifts = 0

        self.global_data_list = []
        self.global_bin_labels = []
        self.global_mul_labels = []

    def _collate_dicts(self, dict_list):
        collated = {}
        for k in dict_list[0].keys():
            if isinstance(dict_list[0][k], torch.Tensor):
                collated[k] = torch.stack([d[k] for d in dict_list])
            else:
                collated[k] = [d[k] for d in dict_list]
        return collated

    def run_stream(self, dataloader):
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        progress = tqdm(dataloader, desc=f"Running Strategy [{self.strategy}] on Test Stream")

        for batch in progress:
            batch_device = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            lbl_bin = batch_device['is_malicious_label'].view(-1)
            lbl_mul = batch_device.get('attack_family_label', torch.zeros_like(lbl_bin))
            if lbl_mul.dim() > 1: lbl_mul = torch.argmax(lbl_mul, dim=1)

            # ---------- A. 测试阶段 (Test-then-Train) ----------
            with torch.no_grad():
                outputs = self.model(batch_device)
                features = outputs['multiview_embeddings']
                logits_bin = outputs['is_malicious_cls_logits']
                preds_bin = (torch.sigmoid(logits_bin).squeeze(-1) > 0.5).long()

                if 'attack_family_cls_logits' in outputs and outputs['attack_family_cls_logits'] is not None:
                    preds_mul = torch.argmax(outputs['attack_family_cls_logits'], dim=1)
                else:
                    preds_mul = torch.zeros_like(preds_bin)

            batch_size = features.shape[0]

            self.stream_all_preds.extend(preds_bin.cpu().tolist())
            self.stream_all_labels.extend(lbl_bin.cpu().tolist())
            self.stream_all_mul_preds.extend(preds_mul.cpu().tolist())
            self.stream_all_mul_labels.extend(lbl_mul.cpu().tolist())

            self.current_window_preds.extend(preds_bin.cpu().tolist())
            self.current_window_labels.extend(lbl_bin.cpu().tolist())

            if len(self.current_window_preds) >= self.eval_window_size:
                f1 = f1_score(self.current_window_labels[:self.eval_window_size],
                              self.current_window_preds[:self.eval_window_size],
                              average='binary', pos_label=1, zero_division=0)
                self.f1_history.append(f1)
                self.current_window_labels = self.current_window_labels[self.eval_window_size:]
                self.current_window_preds = self.current_window_preds[self.eval_window_size:]

            # ---------- B. 漂移检测与适应阶段 ----------
            for i in range(batch_size):
                feat = features[i]
                is_drift = self.detector.update(self.detector.preprocess(feat.unsqueeze(0)))

                single_dict = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        single_dict[k] = v[i].detach().cpu()
                    else:
                        single_dict[k] = v[i] if isinstance(v, list) else v

                self.global_data_list.append(single_dict)
                self.global_bin_labels.append(lbl_bin[i].detach().cpu())
                self.global_mul_labels.append(lbl_mul[i].detach().cpu())

                if is_drift:
                    self.drifts += 1
                    current_window_idx = len(self.f1_history)
                    if current_window_idx not in self.drift_points:
                        self.drift_points.append(current_window_idx)

                    if self.strategy == 'S':
                        pass
                    elif self.strategy == 'R':
                        t0 = time.time()
                        all_bins = torch.stack(self.global_bin_labels)
                        all_muls = torch.stack(self.global_mul_labels)
                        all_dicts = self._collate_dicts(self.global_data_list)
                        self.adapter.adapt(all_dicts, all_bins, all_muls)
                        self.time_cost += (time.time() - t0)
                    elif self.strategy == 'A':
                        if len(self.global_data_list) >= 32:  # 确保最小 batch 能够前馈
                            t0 = time.time()
                            win_size = min(self.adapter.config.get('window', 500), len(self.global_data_list))
                            recent_bins = torch.stack(self.global_bin_labels[-win_size:])
                            recent_muls = torch.stack(self.global_mul_labels[-win_size:])
                            recent_dicts = self._collate_dicts(self.global_data_list[-win_size:])
                            self.adapter.adapt(recent_dicts, recent_bins, recent_muls)
                            self.time_cost += (time.time() - t0)

                    self.detector.reset()

                limit = 20000 if self.strategy == 'R' else 2000
                if len(self.global_data_list) > limit:
                    self.global_data_list.pop(0)
                    self.global_bin_labels.pop(0)
                    self.global_mul_labels.pop(0)

        if len(self.current_window_preds) > 0:
            f1 = f1_score(self.current_window_labels, self.current_window_preds, average='binary', pos_label=1,
                          zero_division=0)
            self.f1_history.append(f1)


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info("🚀 启动 S-R-A 策略对比消融实验 (Data Export Mode)")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "cic_ids_2017"
    dataset_yaml_path = os.path.join(current_dir, f"../models/flow_bert_multiview/config/datasets/{dataset_name}.yaml")

    if os.path.exists(dataset_yaml_path):
        dataset_cfg = OmegaConf.load(dataset_yaml_path)
        cfg.data.dataset = dataset_cfg.name
        if 'flow_data_path' in dataset_cfg: cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path,
                                                                                   dataset_cfg.name)
        if 'session_split_path' in dataset_cfg: cfg.data.session_split.session_split_path = resolve_path(
            dataset_cfg.session_split_path, dataset_cfg.name)

    cfg.data.split_mode = "flow"
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")
    val_loader = dm.val_dataloader()  # 用于跑 Baseline
    test_loader = dm.test_dataloader()  # 用于跑流式消融

    _ = dm.train_dataloader()
    dataset_wrapper = MappingWrapper(dm.train_dataset.categorical_val2idx_mappings,
                                     dm.train_dataset.categorical_columns_effective)

    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = find_valid_checkpoint(project_root)
    if not ckpt_path:
        logger.error("❌ 找不到有效的 Checkpoint！")
        return

    results = {}
    history_dict = {}
    drift_points_dict = {}

    # ==========================
    # 0. 运行 Validation Baseline
    # ==========================
    logger.info(f"\n{'=' * 50}\n▶️ 开始运行: Baseline (验证集初始性能)\n{'=' * 50}")
    model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
    results['Baseline'] = evaluate_baseline_on_val(model, val_loader)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ==========================
    # 1. 运行 S, R, A 策略 (在测试数据流上)
    # ==========================
    strategies = ['S', 'R', 'A']

    for strat in strategies:
        logger.info(f"\n{'=' * 50}\n▶️ 开始运行策略: {strat}\n{'=' * 50}")
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
        runner = SRAExperimentRunner(model, cfg, strategy=strat)

        runner.run_stream(test_loader)

        # 计算全局评价指标
        global_p = precision_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)
        global_r = recall_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)
        global_f1 = f1_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)

        # 多分类宏/微观 F1
        y_true_bin = np.array(runner.stream_all_labels)
        y_true_mul = np.array(runner.stream_all_mul_labels)
        y_pred_mul = np.array(runner.stream_all_mul_preds)
        mal_mask = (y_true_bin == 1)
        if mal_mask.sum() > 0:
            mac_f1 = f1_score(y_true_mul[mal_mask], y_pred_mul[mal_mask], average='macro', zero_division=0)
            mic_f1 = f1_score(y_true_mul[mal_mask], y_pred_mul[mal_mask], average='micro', zero_division=0)
        else:
            mac_f1, mic_f1 = 0.0, 0.0

        strat_name = f"multiviewflowbert-{strat}"

        results[strat] = {
            "Strategy": strat_name,
            "Precision": global_p,
            "Recall": global_r,
            "F1_Score": global_f1,
            "Mul_MacF1": mac_f1,
            "Mul_MicF1": mic_f1,
            "Adaptation_Time(s)": runner.time_cost,
            "Drift_Count": runner.drifts
        }

        history_dict[f'F1_Score_{strat}'] = runner.f1_history
        drift_points_dict[strat] = runner.drift_points

        del model, runner
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- 输出最终的论文级表格 ----
    df_summary = pd.DataFrame(list(results.values()))

    # 调整列顺序使其更像表 4.2
    cols = ['Strategy', 'Drift_Count', 'Precision', 'Recall', 'F1_Score', 'Mul_MacF1', 'Mul_MicF1',
            'Adaptation_Time(s)']
    df_summary = df_summary[cols]

    print("\n\n" + "=" * 110)
    print("📊 概念漂移适应策略最终性能对比报告 (参考论文标准)")
    print("=" * 110)
    print(df_summary.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 110 + "\n")

    # ---- 导出 MATLAB 用数据 ----
    history_df = pd.DataFrame(history_dict)
    history_df.index.name = "Time_Window_Index"
    f1_csv_path = os.path.join(project_root, "matlab_SRA_f1_trend.csv")
    history_df.to_csv(f1_csv_path)
    logger.info(f"✅ MATLAB 折线图数据 已保存至: {f1_csv_path}")

    drift_df = pd.DataFrame({"Drift_Window_Index": drift_points_dict['A']})
    drift_csv_path = os.path.join(project_root, "matlab_drift_points_A.csv")
    drift_df.to_csv(drift_csv_path, index=False)
    logger.info(f"✅ MATLAB 参考线数据 已保存至: {drift_csv_path}")


if __name__ == "__main__":
    main()