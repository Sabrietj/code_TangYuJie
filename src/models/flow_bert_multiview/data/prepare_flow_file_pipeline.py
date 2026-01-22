import re
import torch
import ast
import os
import pandas as pd
import sys
import logging
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

utils_path = os.path.join(os.path.dirname(__file__),  '..', '..', '..', 'utils')
sys.path.insert(0, utils_path) 
# 设置日志
from logging_config import setup_preset_logging
# 使用统一的日志配置
logger = setup_preset_logging(log_level=logging.INFO)


def is_rank0():
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )

def wait_for_file(path, timeout=3600):
    import time
    start = time.time()
    while not os.path.exists(path):
        if time.time() - start > timeout:
            raise TimeoutError(f"Waiting for {path} timed out")
        time.sleep(1)


def prepare_sampled_data_files(cfg):
    """
    返回:
      cfg.data.flow_data_path
      cfg.data.session_split.session_split_path (可选)
    """
    # ============================================================
    # 1️⃣ 根据配置进行随机采样（控制规模）
    # ============================================================
    if cfg.data.sampling.enabled and cfg.data.sampling.sample_ratio < 1.0: 
        if cfg.data.split_mode == "flow":
            flow_data_path = cfg.data.flow_data_path
            sampled_flow_data_path = flow_data_path.replace(
                ".csv",
                f".sampled_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_flow_data_path):
                if is_rank0():
                    sample_flow_csv(
                        input_path=flow_data_path,
                        output_path=sampled_flow_data_path,
                        ratio=cfg.data.sampling.sample_ratio,
                        seed=cfg.data.random_state,
                    )
                wait_for_file(sampled_flow_data_path)

            cfg.data.flow_data_path = sampled_flow_data_path
            logger.info(f"使用随机下采样后的网络流数据文件: {sampled_flow_data_path}")

        elif cfg.data.split_mode == "session":
            session_data_path = cfg.data.session_split.session_split_path
            sampled_session_data_path = session_data_path.replace(
                ".csv", f".sampled_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_session_data_path):
                if is_rank0():
                    session_df = pd.read_csv(session_data_path, low_memory=False)
                    # session_df = session_df.sample(
                    #     frac=cfg.data.sampling.sample_ratio,
                    #     random_state=cfg.data.random_state,
                    # )
                    if cfg.data.sampling.sample_ratio > 0:
                        step = int(1 / cfg.data.sampling.sample_ratio)
                        step = max(1, step)
                        # 使用切片操作，天然保持顺序
                        session_df = session_df.iloc[::step]
                        logger.info(f"应用时序保留采样 (Systematic Sampling): step={step}, 原始行数已缩减")
                    else:
                        raise ValueError("Sample ratio must be > 0")
                    session_df.to_csv(sampled_session_data_path, index=False)

            wait_for_file(sampled_session_data_path)
            cfg.data.session_split.session_split_path = sampled_session_data_path

            sampled_flow_data_path = cfg.data.flow_data_path.replace(
                ".csv", f".from_session_{cfg.data.sampling.sample_ratio}_{cfg.data.random_state}.csv"
            )

            if not os.path.exists(sampled_flow_data_path):
                if is_rank0():
                    session_df = pd.read_csv(sampled_session_data_path, low_memory=False)
                    sampled_flow_uids = set()
                    for v in session_df[cfg.data.session_split.flow_uid_list_column]:
                        if isinstance(v, str):
                            v = ast.literal_eval(v)
                        sampled_flow_uids.update(v)

                    if len(sampled_flow_uids) == 0:
                        raise ValueError(
                            f"cfg.data.sampling.sample_ratio={cfg.data.sampling.sample_ratio} "
                            f"导致 flow_uids 为空，无法训练"
                        )
                    
                    # 流级过滤（chunk 版，避免一次性读爆内存）
                    filter_flow_csv_by_uid(
                        input_flow_csv=cfg.data.flow_data_path,
                        output_flow_csv=sampled_flow_data_path,
                        keep_uids=sampled_flow_uids,
                    )

            wait_for_file(sampled_flow_data_path)
            cfg.data.flow_data_path = sampled_flow_data_path

        else:
            raise ValueError(f"不支持的 split_mode={cfg.data.split_mode}，无法进行采样")

    # ============================================================
    # 2️⃣ 过滤端口和服务（控制任务语义）
    # ============================================================
    exclude_ports = OmegaConf.select(cfg, "datasets.exclude_ports")
    exclude_services = OmegaConf.select(cfg, "datasets.exclude_services")

    if exclude_ports or exclude_services:
        tag_parts = []
        if exclude_ports:
            tag_parts.append("port_" + "_".join(map(str, sorted(exclude_ports))))
        if exclude_services:
            tag_parts.append("svc_" + "_".join(sorted(exclude_services)))

        tag = "__".join(tag_parts)

        filtered_path = cfg.data.flow_data_path.replace(
            ".csv", f".filtered_{tag}.csv"
        )

        if not os.path.exists(filtered_path):
            if is_rank0():
                filter_flow_csv_by_port_and_service(
                    input_flow_csv=cfg.data.flow_data_path,
                    output_flow_csv=filtered_path,
                    exclude_ports=exclude_ports,
                    exclude_services=exclude_services,
                )
            wait_for_file(filtered_path)

        cfg.data.flow_data_path = filtered_path

    # ============================================================
    # 3️⃣ 过滤 excluded_classes（控制任务语义）
    # （如 CIC-IDS-2017数据集的 Infiltration / Heartbleed 攻击等）
    # ============================================================
    excluded_classes = OmegaConf.select(cfg, "datasets.excluded_classes")
    label_column = OmegaConf.select(cfg, "data.multiclass_label_column")

    if excluded_classes and label_column is not None:
        excluded_tag = "_".join(sorted(excluded_classes))
        excluded_tag = re.sub(r"[^a-zA-Z0-9_]", "_", excluded_tag)    
        filtered_flow_data_path = cfg.data.flow_data_path.replace(
            ".csv", f".filtered_{excluded_tag}.csv"
        )

        if not os.path.exists(filtered_flow_data_path):
            if is_rank0():
                logger.info(f"过滤以下 excluded_classes: {excluded_classes}")
                filter_flow_csv_by_excluded_classes(
                    input_flow_csv=cfg.data.flow_data_path,
                    output_flow_csv=filtered_flow_data_path,
                    label_column=label_column,
                    excluded_classes=list(excluded_classes),
                )
            wait_for_file(filtered_flow_data_path)

        cfg.data.flow_data_path = filtered_flow_data_path
        logger.info(
            f"使用过滤 excluded_classes 后的 flow 数据文件: "
            f"{filtered_flow_data_path}"
        )

    # ============================================================
    # 4️⃣ 合并 merged_classes（label canonicalization）
    # ============================================================
    merged_classes = OmegaConf.select(cfg, "datasets.merged_classes")

    if merged_classes and label_column is not None:
        merged_tag = "_".join(sorted(merged_classes.keys()))
        merged_tag   = re.sub(r"[^a-zA-Z0-9_]", "_", merged_tag)

        merged_flow_data_path = cfg.data.flow_data_path.replace(
            ".csv", f".merged_{merged_tag}.csv"
        )

        if not os.path.exists(merged_flow_data_path):
            if is_rank0():
                logger.info(f"合并 merged_classes: {dict(merged_classes)}")
                merge_flow_csv_classes(
                    input_flow_csv=cfg.data.flow_data_path,
                    output_flow_csv=merged_flow_data_path,
                    label_column=label_column,
                    merged_classes=OmegaConf.to_container(merged_classes),
                )
            wait_for_file(merged_flow_data_path)

        cfg.data.flow_data_path = merged_flow_data_path
        logger.info(
            f"使用 merged_classes 后的 flow 数据文件: {merged_flow_data_path}"
        )    


def sample_flow_csv(
    input_path: str,
    output_path: str,
    ratio: float,
    seed: int,
    chunksize: int = 100_000,
):
    rng = np.random.default_rng(seed)
    first = True
    total_seen = 0
    total_kept = 0

    reader = pd.read_csv(input_path, chunksize=chunksize, low_memory=False)
    if is_rank0():
        reader = tqdm(reader, desc="Sampling flows", unit="chunk")

    for chunk in reader:
        total_seen += len(chunk)
        mask = rng.random(len(chunk)) < ratio
        sampled = chunk[mask]
        kept = len(sampled)
        total_kept += kept

        if kept == 0:
            continue

        sampled.to_csv(
            output_path,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    if first:
        raise RuntimeError(
            f"sample_flow_csv: ratio={ratio} 导致采样结果为空"
        )

    logger.info(
        f"[sample_flow_csv] kept {total_kept} / {total_seen} flows "
        f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
    )


def filter_flow_csv_by_uid(
    input_flow_csv: str,
    output_flow_csv: str,
    keep_uids: set,
    chunksize: int = 100_000,
):
    first = True
    total_seen = 0
    total_kept = 0

    reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
    if is_rank0():
        reader = tqdm(reader, desc="Filtering flows by uid", unit="chunk")

    for chunk in reader:
        total_seen += len(chunk)
        filtered = chunk[chunk["uid"].isin(keep_uids)]
        kept = len(filtered)
        total_kept += kept

        if kept == 0:
            continue

        filtered.to_csv(
            output_flow_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    if first:
        raise RuntimeError(
            "filter_flow_csv_by_uid: 过滤后结果为空，请检查 keep_uids 是否正确"
        )

    logger.info(
        f"[filter_flow_csv_by_uid] kept {total_kept} / {total_seen} flows "
        f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
    )



def filter_flow_csv_by_port_and_service(
    input_flow_csv: str,
    output_flow_csv: str,
    exclude_ports: list = None,
    exclude_services: list = None,
    chunksize: int = 100_000,
):
    exclude_ports = set(exclude_ports or [])
    exclude_services = set(exclude_services or [])

    first = True
    total_kept = 0
    total_seen = 0

    reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)

    # 仅 rank0 显示进度条（避免 DDP / 多进程刷屏）
    if is_rank0():
        reader = tqdm(
            reader,
            desc="Filtering flows by port/service",
            unit="chunk",
        )

    for chunk in reader:
        total_seen += len(chunk)

        mask = pd.Series(True, index=chunk.index)

        if exclude_ports:
            mask &= ~chunk["conn.id.resp_p"].isin(exclude_ports)

        if exclude_services:
            mask &= ~chunk["conn.service"].isin(exclude_services)

        filtered = chunk[mask]
        kept = len(filtered)
        total_kept += kept

        if kept == 0:
            continue

        filtered.to_csv(
            output_flow_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    if first:
        logger.warning(
            "filter_flow_csv_by_port_and_service: 过滤后结果为空"
        )
    else:
        logger.info(
            f"[filter_flow_csv_by_port_and_service] "
            f"kept {total_kept} / {total_seen} flows "
            f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
        )

def filter_flow_csv_by_excluded_classes(
    input_flow_csv: str,
    output_flow_csv: str,
    label_column: str,
    excluded_classes: list,
    chunksize: int = 100_000,
):
    first = True
    total_seen = 0
    total_kept = 0

    reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
    if is_rank0():
        reader = tqdm(reader, desc="Filtering flows by excluded classes", unit="chunk")

    for chunk in reader:
        total_seen += len(chunk)
        filtered = chunk[~chunk[label_column].isin(excluded_classes)]
        kept = len(filtered)
        total_kept += kept

        if kept == 0:
            continue

        filtered.to_csv(
            output_flow_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    if first:
        logger.warning(
            "filter_flow_csv_by_excluded_classes: 未命中任何 excluded_classes，对结果无影响"
        )
    else:
        logger.info(
            f"[filter_flow_csv_by_excluded_classes] "
            f"kept {total_kept} / {total_seen} flows "
            f"({total_kept / max(total_seen, 1) * 100:.2f}%)"
        )


def merge_flow_csv_classes(
    input_flow_csv: str,
    output_flow_csv: str,
    label_column: str,
    merged_classes: dict,
    chunksize: int = 100_000,
):
    """
    将 merged_classes 中的多个原始类别，统一替换为新的类别名
    """
    # 构造反向映射：old_label -> new_label
    merge_map = {}
    for new_label, old_labels in merged_classes.items():
        for old in old_labels:
            merge_map[old] = new_label

    first = True
    total_seen = 0

    reader = pd.read_csv(input_flow_csv, chunksize=chunksize, low_memory=False)
    if is_rank0():
        reader = tqdm(reader, desc="Merging flow classes", unit="chunk")

    for chunk in reader:
        if label_column not in chunk.columns:
            raise KeyError(f"label_column={label_column} 不在 CSV 列中")

        total_seen += len(chunk)

        chunk[label_column] = chunk[label_column].replace(merge_map)

        chunk.to_csv(
            output_flow_csv,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False

    if first:
        logger.warning("merge_flow_csv_classes: 本次数据中未命中任何 merged_classes，对结果无影响")
    else:
        logger.info(f"[merge_flow_csv_classes] processed {total_seen} flows")
