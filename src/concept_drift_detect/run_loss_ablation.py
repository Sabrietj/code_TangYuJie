import sys
import os
import glob
import pandas as pd
import numpy as np

# 添加项目根目录到 Path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import torch
import hydra
import logging
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

try:
    from tabulate import tabulate
except ImportError:
    tabulate = None

# 引入项目模块
from src.models.flow_bert_multiview.models.flow_bert_multiview import FlowBertMultiview
from src.models.flow_bert_multiview.data.flow_bert_multiview_dataset import MultiviewFlowDataModule
from src.concept_drift_detect.run_experiment import DriftExperimentRunner, MappingWrapper, resolve_path, \
    find_valid_checkpoint

logging.basicConfig(level=logging.INFO, format='[%(asctime)s %(filename)s:%(lineno)d] [%(levelname)s] %(message)s')
logger = logging.getLogger("BNDM_Ablation")

# ==========================================
# 定义损失函数消融配置 (CDA-SSL 系列)
# ==========================================
ABLATION_SETTINGS = {
    "1. Base Only": {
        "lambda_topo": 0.0,
        "lambda_cons": 0.0,
        "lambda_proto": 0.0
    },
    "2. CDA-SSL-2 (+Topo)": {
        "lambda_topo": 0.1,
        "lambda_cons": 0.0,
        "lambda_proto": 0.0
    },
    "3. CDA-SSL-3 (+Topo+Cons)": {
        "lambda_topo": 0.1,
        "lambda_cons": 0.1,
        "lambda_proto": 0.0
    },
    "4. CDA-SSL (Full)": {
        "lambda_topo": 0.1,
        "lambda_cons": 0.1,
        "lambda_proto": 0.1
    }
}


class AblationExperimentRunner(DriftExperimentRunner):
    """
    继承原有的 Runner，专门用于动态覆盖 loss_weights

    """

    def __init__(self, model, cfg, detector_type="bndm", loss_overrides=None):
        # 强制指定检测器为 bndm
        super().__init__(model, cfg, detector_type=detector_type, enable_adaptation=True)

        if loss_overrides:
            self.adapt_config.update(loss_overrides)
            # 重新实例化 Adapter
            from src.concept_drift_detect.adapter import IncrementalAdapter
            self.adapter = IncrementalAdapter(model, self.adapt_config)

            logger.info(
                f"🔧 BNDM Adapter Weights: Topo={self.adapter.lambda_topo}, Cons={self.adapter.lambda_cons}, Proto={self.adapter.lambda_proto}")


@hydra.main(config_path="../models/flow_bert_multiview/config", config_name="flow_bert_multiview_config",
            version_base="1.2")
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)

    # 固定检测器为 BNDM
    DETECTOR = "bndm"

    logger.info(f"🚀 Starting Loss Ablation Study | Base Detector: {DETECTOR.upper()}")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_name = "cic_ids_2017"
    dataset_yaml_path = os.path.join(current_dir, f"../models/flow_bert_multiview/config/datasets/{dataset_name}.yaml")

    # 加载数据集配置
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

    # 获取标签映射用于报告打印
    if hasattr(dm, 'train_dataset') and dm.train_dataset:
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)
    else:
        _ = dm.train_dataloader()
        mappings = getattr(dm.train_dataset, 'categorical_val2idx_mappings', None)
        effective_columns = getattr(dm.train_dataset, 'categorical_columns_effective', None)

    target_names_mul = None
    if mappings and 'label' in mappings:
        sorted_mappings = sorted(mappings['label'].items(), key=lambda x: x[1])
        target_names_mul = [k for k, v in sorted_mappings]

    dataset_wrapper = MappingWrapper(mappings, effective_columns)
    project_root = os.path.abspath(os.path.join(current_dir, "../../"))
    ckpt_path = find_valid_checkpoint(project_root)

    if not ckpt_path:
        logger.error("❌ CRITICAL: No valid checkpoint found!")
        return

    results = []

    for ablation_name, loss_weights in ABLATION_SETTINGS.items():
        logger.info(f"\n{'=' * 60}\n🔬 Ablation Setting (BNDM): {ablation_name}\n{'=' * 60}")

        try:
            # 每次重新加载模型，确保实验独立性
            model = FlowBertMultiview.load_from_checkpoint(ckpt_path, cfg=cfg, dataset=dataset_wrapper)

            runner = AblationExperimentRunner(
                model,
                cfg,
                detector_type=DETECTOR,
                loss_overrides=loss_weights
            )

            runner.run_stream(test_loader)
            runner.print_detailed_report(target_names_mul)

            # 获取并筛选指定指标
            res = runner.get_results()
            res['Ablation_Strategy'] = ablation_name
            results.append(res)

            del model, runner
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        except Exception as e:
            logger.error(f"❌ Failed during ablation {ablation_name}: {e}")

    # ==================================
    # 打印全局消融报告总结
    # ==================================
    if results:
        df = pd.DataFrame(results)

        # 指标重命名与筛选
        rename_mapping = {
            'Bin_Acc': 'Accuracy(Bin)',
            'Bin_Rec': 'Recall(Bin)',
            'Bin_F1': 'F1-score(Bin)',
            'Mul_MicF1': 'Micro-F1(Mul)'
        }
        df = df.rename(columns=rename_mapping)

        # 严格按照要求的列顺序排列
        columns_order = [
            'Ablation_Strategy',
            'Accuracy(Bin)',
            'Recall(Bin)',
            'F1-score(Bin)',
            'Micro-F1(Mul)',
            'Drifts',
            'Adaptations'
        ]
        df = df[[c for c in columns_order if c in df.columns]]

        print("\n" + "=" * 110)
        print(f"📊 概念漂移适应消融实验结论报告 (Detector: BNDM)")
        print("=" * 110)
        if tabulate:
            print(tabulate(df, headers='keys', tablefmt='pipe', showindex=False, floatfmt=".4f"))
        else:
            print(df.to_string(index=False, float_format="%.4f"))
        print("=" * 110)


if __name__ == "__main__":
    main()