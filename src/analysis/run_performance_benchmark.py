import sys
import os
import time
import torch
import hydra
import logging
import pandas as pd
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.detectors import BNDMDetector, ADWINDetector, KSWINDetector
from src.concept_drift_detect.adapter import IncrementalAdapter
from src.concept_drift_detect.run_experiment import resolve_path, find_valid_checkpoint, MappingWrapper

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] %(message)s')
logger = logging.getLogger("Benchmark")


class BenchmarkRunner:
    def __init__(self, model, cfg, detector_type, window_samples):
        self.model = model
        self.detector_type = detector_type

        # 动态修改配置
        self.det_config = {'seed': 2026, 'threshold': 0.1, 'max_level': 6, 'alpha_scale': 0.05}
        self.det_config['window_size'] = window_samples  # 注入当前的滑动窗口大小

        if detector_type == "bndm":
            self.detector = BNDMDetector(self.det_config)
        elif detector_type == "adwin":
            self.detector = ADWINDetector(self.det_config)
        elif detector_type == "kswin":
            self.detector = KSWINDetector(self.det_config)

        self.adapt_config = {'lr': 1e-4, 'epochs': 5, 'buffer_size': 800, 'batch_size': 32, 'window': 500,
                             'lambda_topo': 1.0, 'lambda_proto': 1.0, 'lambda_cons': 0.5}
        self.adapter = IncrementalAdapter(model, self.adapt_config)

        self.buffer_features = []
        self.buffer_bin_labels = []
        self.buffer_mul_labels = []

    def run_stream(self, dataloader):
        self.model.eval()
        if torch.cuda.is_available():
            self.model.cuda()

        # 计时器与计数器
        times = {"inference": 0.0, "detection": 0.0, "adaptation": 0.0}
        drifts = 0
        total_start = time.time()

        # 为了加速评测，我们重置显存统计
        torch.cuda.reset_peak_memory_stats()

        progress = tqdm(dataloader, desc=f"Benchmarking {self.detector_type.upper()}")
        for batch in progress:
            batch = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            lbl_bin = batch['is_malicious_label'].view(-1)
            lbl_mul = batch.get('attack_family_label', torch.zeros_like(lbl_bin))
            if lbl_mul.dim() > 1: lbl_mul = torch.argmax(lbl_mul, dim=1)

            # 1. 记录推理时间
            t0 = time.time()
            with torch.no_grad():
                outputs = self.model(batch)
                features = outputs['multiview_embeddings']
            torch.cuda.synchronize()  # 确保 GPU 计算完成
            times["inference"] += (time.time() - t0)

            batch_size = features.shape[0]

            for i in range(batch_size):
                feat = features[i]

                # 2. 记录漂移检测时间
                t1 = time.time()
                feat_input = feat.unsqueeze(0)
                drift_detected = self.detector.update(self.detector.preprocess(feat_input))
                times["detection"] += (time.time() - t1)

                # 缓存数据用于 Adaptation
                single_sample = {k: (
                    v[i].detach().cpu().clone() if isinstance(v, torch.Tensor) else v[i] if isinstance(v, list) else v)
                                 for k, v in batch.items()}
                self.buffer_features.append(single_sample)
                self.buffer_bin_labels.append(lbl_bin[i].detach().cpu().clone())
                self.buffer_mul_labels.append(lbl_mul[i].detach().cpu().clone())

                if drift_detected:
                    drifts += 1
                    if len(self.buffer_features) >= 32:
                        window = self.adapt_config.get('window', 500)
                        adapt_dicts = self.buffer_features[-window:]
                        adapt_batch = {}
                        for k in adapt_dicts[0].keys():
                            if isinstance(adapt_dicts[0][k], torch.Tensor):
                                adapt_batch[k] = torch.stack([d[k] for d in adapt_dicts])
                            else:
                                adapt_batch[k] = [d[k] for d in adapt_dicts]
                        adapt_bin = torch.stack(self.buffer_bin_labels[-window:])
                        adapt_mul = torch.stack(self.buffer_mul_labels[-window:])

                        # 3. 记录模型自适应微调时间
                        t2 = time.time()
                        self.adapter.adapt(adapt_batch, adapt_bin, adapt_mul)
                        torch.cuda.synchronize()
                        times["adaptation"] += (time.time() - t2)

                    self.detector.reset()

                # 限制 Buffer 大小
                if len(self.buffer_features) > 800:
                    self.buffer_features = self.buffer_features[-800:]
                    self.buffer_bin_labels = self.buffer_bin_labels[-800:]
                    self.buffer_mul_labels = self.buffer_mul_labels[-800:]

        total_time = time.time() - total_start
        peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)  # 转换为 MB

        return {
            "Drifts": drifts,
            "Total_Time_s": total_time,
            "Inference_Time_s": times["inference"],
            "Detection_Time_s": times["detection"],
            "Adaptation_Time_s": times["adaptation"],
            "Peak_VRAM_MB": peak_vram
        }


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # 初始化数据和环境
    dataset_yaml_path = os.path.join(os.path.dirname(__file__),
                                     "../models/flow_bert_multiview/config/datasets/cic_ids_2017.yaml")
    if os.path.exists(dataset_yaml_path):
        dataset_cfg = OmegaConf.load(dataset_yaml_path)
        cfg.data.dataset = dataset_cfg.name
        cfg.data.flow_data_path = resolve_path(dataset_cfg.flow_data_path, dataset_cfg.name)
        cfg.data.session_split.session_split_path = resolve_path(dataset_cfg.session_split_path, dataset_cfg.name)

    cfg.data.split_mode = "flow"
    dm = MultiviewFlowDataModule(cfg)
    dm.setup(stage="fit")
    test_loader = dm.test_dataloader()

    mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
    effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)
    dataset_wrapper = MappingWrapper(mappings, effective_columns)

    ckpt_path = find_valid_checkpoint(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

    # ================= 压测核心配置 =================
    BATCH_SIZE = 256
    WINDOW_BATCHES = [5, 10, 15, 20]
    DETECTORS = ["bndm", "adwin", "kswin"]
    results = []

    logger.info(f"🚀 开始空间与时间开销基准测试...")

    for w_b in WINDOW_BATCHES:
        w_samples = w_b * BATCH_SIZE
        logger.info(f"\n{'=' * 50}\n🌟 当前滑动窗口大小: {w_b} Batches ({w_samples} Samples)\n{'=' * 50}")

        for det in DETECTORS:
            logger.info(f"👉 正在压测算法: {det.upper()}")

            # 清理显存防 OOM
            torch.cuda.empty_cache()

            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)
            runner = BenchmarkRunner(model, cfg, det, w_samples)

            res = runner.run_stream(test_loader)

            # 整理结果
            row = {
                "Detector": det.upper(),
                "Window_Batches": w_b,
                "Window_Samples": w_samples,
                "Drift_Count": res["Drifts"],
                "Total_Time(s)": round(res["Total_Time_s"], 2),
                "Detection_Time(s)": round(res["Detection_Time_s"], 2),
                "Adaptation_Time(s)": round(res["Adaptation_Time_s"], 2),
                "Peak_VRAM(MB)": round(res["Peak_VRAM_MB"], 2)
            }
            results.append(row)

            logger.info(
                f"✅ {det.upper()} 压测完成 | 漂移: {row['Drift_Count']}次 | 耗时: {row['Total_Time(s)']}s | 显存: {row['Peak_VRAM(MB)']}MB")

            del model, runner

    # 保存并打印最终结果
    df = pd.DataFrame(results)
    csv_path = os.path.join(os.path.dirname(__file__), "benchmark_results.csv")
    df.to_csv(csv_path, index=False)

    print("\n" + "=" * 100)
    print("📊 资源开销压测结果总结 (已保存至 benchmark_results.csv)")
    print("=" * 100)
    print(df.to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    main()