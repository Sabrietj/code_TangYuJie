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


class SRAExperimentRunner:
    def __init__(self, model, cfg, strategy="A"):
        """
        strategy: 'Baseline', 'S' (Static), 'R' (Vanilla Retrain), 'A' (Adaptation)
        """
        self.model = model
        self.cfg = cfg
        self.strategy = strategy

        # 1. 初始化 BNDM 漂移检测器
        cd_cfg = cfg.get("concept_drift", {})
        self.det_config = {'seed': cd_cfg.get('seed', 2026), 'threshold': 0.01, 'max_level': 6, 'window_size': 1000}
        self.detector = BNDMDetector(self.det_config)

        # 2. 根据不同策略配置 Adapter
        adapt_config = OmegaConf.to_container(cd_cfg.get("adaptation", {}), resolve=True) if cd_cfg.get(
            "adaptation") else {}
        if self.strategy == "R":
            # 🔴 R 策略：传统 Vanilla 重训。扩大历史 buffer，关闭所有增量约束，增大 batch_size
            adapt_config['buffer_size'] = 20000  # 保留大量历史数据模拟重训
            adapt_config['epochs'] = 5  # 增加 Epoch 保证拟合
            adapt_config['window'] = 20000
            adapt_config['batch_size'] = 128  # 加大 Batch 加速传统重训
            adapt_config['lambda_topo'] = 0.0  # 关闭拓扑结构保护
            adapt_config['lambda_cons'] = 0.0  # 关闭一致性保护
            adapt_config['lambda_proto'] = 0.0  # 关闭原型保护
        else:
            # 🟢 A 策略：基于 BNDM 的高效增量适应。保留近期数据，开启增量约束防止灾难性遗忘
            adapt_config['buffer_size'] = 800
            adapt_config['epochs'] = 3
            adapt_config['window'] = 500
            adapt_config['batch_size'] = 32
            # lambda 约束保持配置文件中的默认值 (开启状态)

        self.adapter = IncrementalAdapter(model, adapt_config)

        # 3. 流式评估指标记录器
        self.eval_window_size = 2000  # 每 2000 个样本评估一次窗口 F1 (用于画折线图)
        self.current_window_preds = []
        self.current_window_labels = []

        # 全局记录器（用于计算最终表格中的 Precision, Recall, F1）
        self.stream_all_preds = []
        self.stream_all_labels = []

        self.f1_history = []
        self.drift_points = []
        self.time_cost = 0.0
        self.drifts = 0

        # 全局数据缓冲
        self.global_data_list = []
        self.global_bin_labels = []
        self.global_mul_labels = []

    def _collate_dicts(self, dict_list):
        """将 List[Dict] 拼接组装为 Dict[Tensor] 供 Adapter 消费"""
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

        progress = tqdm(dataloader, desc=f"Running Strategy [{self.strategy}]")

        for batch in progress:
            batch_device = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            lbl_bin = batch_device['is_malicious_label'].view(-1)
            lbl_mul = batch_device.get('attack_family_label', torch.zeros_like(lbl_bin))
            if lbl_mul.dim() > 1: lbl_mul = torch.argmax(lbl_mul, dim=1)

            # ---------- A. 测试阶段 (Test-then-Train) ----------
            # 模型遇到新数据时，首先进行预测，记录真实表现（绝对防止数据泄露）
            with torch.no_grad():
                outputs = self.model(batch_device)
                features = outputs['multiview_embeddings']
                logits_bin = outputs['is_malicious_cls_logits']
                preds_bin = (torch.sigmoid(logits_bin).squeeze(-1) > 0.5).long()

            batch_size = features.shape[0]
            preds_cpu = preds_bin.cpu().tolist()
            labels_cpu = lbl_bin.cpu().tolist()

            # 记录全局表现
            self.stream_all_preds.extend(preds_cpu)
            self.stream_all_labels.extend(labels_cpu)

            # 记录窗口表现 (用于导出 MATLAB 折线图数据)
            self.current_window_preds.extend(preds_cpu)
            self.current_window_labels.extend(labels_cpu)

            if len(self.current_window_preds) >= self.eval_window_size:
                f1 = f1_score(self.current_window_labels[:self.eval_window_size],
                              self.current_window_preds[:self.eval_window_size],
                              average='binary', pos_label=1, zero_division=0)
                self.f1_history.append(f1)
                self.current_window_labels = self.current_window_labels[self.eval_window_size:]
                self.current_window_preds = self.current_window_preds[self.eval_window_size:]

            # ---------- B. 漂移检测与适应阶段 ----------
            # Baseline 策略不执行任何检测和适应，纯粹作为静态基准
            if self.strategy == 'Baseline':
                continue

            for i in range(batch_size):
                feat = features[i]
                is_drift = self.detector.update(self.detector.preprocess(feat.unsqueeze(0)))

                # 缓存数据
                single_dict = {}
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        single_dict[k] = v[i].detach().cpu()
                    else:
                        single_dict[k] = v[i] if isinstance(v, list) else v

                self.global_data_list.append(single_dict)
                self.global_bin_labels.append(lbl_bin[i].detach().cpu())
                self.global_mul_labels.append(lbl_mul[i].detach().cpu())

                # 触发漂移应对策略
                if is_drift:
                    self.drifts += 1
                    current_window_idx = len(self.f1_history)
                    if current_window_idx not in self.drift_points:
                        self.drift_points.append(current_window_idx)

                    if self.strategy == 'S':
                        # Static: 仅检测出漂移，但不做任何适应措施
                        pass
                    elif self.strategy == 'R':
                        # Vanilla Retrain: 提取缓存中大量的历史数据进行完全重训
                        t0 = time.time()
                        all_bins = torch.stack(self.global_bin_labels)
                        all_muls = torch.stack(self.global_mul_labels)
                        all_dicts = self._collate_dicts(self.global_data_list)
                        logger.info(
                            f"  [Vanilla Retrain] Triggered! Retraining on {len(self.global_data_list)} historical samples...")
                        self.adapter.adapt(all_dicts, all_bins, all_muls)
                        self.time_cost += (time.time() - t0)
                    elif self.strategy == 'A':
                        # Adaptation: 仅使用发生漂移窗口的少量数据进行增量微调
                        t0 = time.time()
                        win_size = min(500, len(self.global_data_list))
                        recent_bins = torch.stack(self.global_bin_labels[-win_size:])
                        recent_muls = torch.stack(self.global_mul_labels[-win_size:])
                        recent_dicts = self._collate_dicts(self.global_data_list[-win_size:])
                        self.adapter.adapt(recent_dicts, recent_bins, recent_muls)
                        self.time_cost += (time.time() - t0)

                    self.detector.reset()

                # 限制内存增长防止 OOM (R 策略保留 20000 模拟大体量重训，A 策略保留 2000 即可)
                limit = 20000 if self.strategy == 'R' else 2000
                if len(self.global_data_list) > limit:
                    self.global_data_list.pop(0)
                    self.global_bin_labels.pop(0)
                    self.global_mul_labels.pop(0)

        # 处理最后一个不满的窗口
        if len(self.current_window_preds) > 0:
            f1 = f1_score(self.current_window_labels, self.current_window_preds, average='binary', pos_label=1,
                          zero_division=0)
            self.f1_history.append(f1)


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    logger.info("🚀 启动 Baseline-S-R-A 策略对比消融实验 (Data Export Mode)")

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
    test_loader = dm.test_dataloader()

    _ = dm.train_dataloader()
    dataset_wrapper = MappingWrapper(dm.train_dataset.categorical_val2idx_mappings,
                                     dm.train_dataset.categorical_columns_effective)

    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = find_valid_checkpoint(project_root)
    if not ckpt_path:
        logger.error("❌ 找不到有效的 Checkpoint！")
        return

    # 依次运行四个对比策略
    strategies = ['Baseline', 'S', 'R', 'A']
    results = {}
    history_dict = {}
    drift_points_dict = {}

    for strat in strategies:
        logger.info(f"\n{'=' * 50}\n▶️ 开始运行策略: {strat}\n{'=' * 50}")
        # 从头重新加载初始模型权重，保证起跑线完全一致
        model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
        runner = SRAExperimentRunner(model, cfg, strategy=strat)

        runner.run_stream(test_loader)

        # 计算全局评价指标 (精确率、召回率、F1)
        global_p = precision_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)
        global_r = recall_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)
        global_f1 = f1_score(runner.stream_all_labels, runner.stream_all_preds, average='binary', zero_division=0)

        strat_name = "Baseline" if strat == 'Baseline' else f"multiviewflowbert-{strat}"

        results[strat] = {
            "Strategy": strat_name,
            "Precision": global_p,
            "Recall": global_r,
            "F1_Score": global_f1,
            "Adaptation_Time(s)": runner.time_cost if strat != 'Baseline' else 0.0,
            "Drift_Count": runner.drifts if strat != 'Baseline' else 0
        }

        history_dict[f'F1_Score_{strat}'] = runner.f1_history
        drift_points_dict[strat] = runner.drift_points

        del model, runner
        if torch.cuda.is_available(): torch.cuda.empty_cache()

    # ---- 输出最终的论文级表格 ----
    df_summary = pd.DataFrame(list(results.values()))
    print("\n\n" + "=" * 90)
    print("📊 概念漂移适应策略最终性能对比报告 (参考论文标准)")
    print("=" * 90)
    print(df_summary.to_markdown(index=False, floatfmt=".4f"))
    print("=" * 90 + "\n")

    # ---- 导出 MATLAB 用数据 ----
    history_df = pd.DataFrame(history_dict)
    history_df.index.name = "Time_Window_Index"
    f1_csv_path = os.path.join(project_root, "matlab_SRA_f1_trend.csv")
    history_df.to_csv(f1_csv_path)
    logger.info(f"✅ MATLAB 折线图数据 (F1分数趋势) 已保存至: {f1_csv_path}")

    drift_df = pd.DataFrame({"Drift_Window_Index": drift_points_dict['A']})
    drift_csv_path = os.path.join(project_root, "matlab_drift_points_A.csv")
    drift_df.to_csv(drift_csv_path, index=False)
    logger.info(f"✅ MATLAB 参考线数据 (漂移触发坐标点) 已保存至: {drift_csv_path}")


if __name__ == "__main__":
    main()