#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_all_flow_dataset_leakage.py
===================================

本脚本用于系统性检测各类特征对目标标签 `is_malicious` 的潜在泄露风险，
用于识别能够“单独预测恶意”的高危特征，避免模型在训练中受到数据泄露污染。

本工具自动分析所有特征列，自动判断特征类型（数值 / 类别 / 字符串 / 嵌入向量），
对每类特征执行对应的泄露评估方法，输出排序后的强泄露特征榜单，并生成可视化图表。

-----------------------------------
1) 特征类型自动识别
-----------------------------------
脚本自动判断特征属于以下类别：

- Numerical（数值特征）
    * int / float
    * 字符串形式数字（如 "123", "3.14"）
- Categorical（类别特征）
    * object / string
    * 离散整数（unique 值数量 < 阈值）
- Embedding（嵌入向量）
    * list/ndarray/字符串形式列表
    * 自动过滤空向量、全 0、常数向量

-----------------------------------
2) 数值型特征泄露检查（Numeric Leakage）
-----------------------------------
以下指标用于衡量数值特征与 `is_malicious` 的可区分性：

- Pearson Correlation（线性相关）
- Mutual Information（非线性依赖）
- AUC Score（特征直接作为分类器时的区分能力）
- Numeric Leakage Score（综合泄露分：由 Pearson/MI/AUC 加权）

强泄露判断标准（参考）：
- Pearson |corr| > 0.5
- MI 高
- AUC > 0.8

-----------------------------------
3) 类别/字符串特征泄露检查（Categorical Leakage）
-----------------------------------
对类别型特征进行以下分析：

- Leakage Ratio（test 中的类别在 train 中出现的占比）
- Conditional Entropy（条件熵，越小越容易泄露）
- NA Ratio（缺失值/无效值占比）
- Categorical Leakage Score（综合泄露分）

强泄露判断标准（参考）：
- Test ↔ Train 的类别共享 > 80%
- 单个类别的标签分布极度偏向（例如 90%+ 都是恶意/正常）
- Conditional Entropy < 0.3

-----------------------------------
4) 输出内容
-----------------------------------
脚本会生成并保存以下内容：

- CSV 报告（数值特征 / 类别特征分别保存）
- 条形图（Bar Plot）
    * 数值特征泄露条形图
    * 类别特征泄露条形图
- 雷达图（Radar Plot）
    * 数值特征泄露雷达图
    * 类别特征泄露雷达图

所有结果会保存到：
    ConfigManager.read_plot_data_path_config() / "leakage_reports"

-----------------------------------
5) Split 支持
-----------------------------------
支持两种数据划分：
- flow-split  （容易泄露，用于诊断问题）
- session-split（实际泛化测试、严谨评估）

-----------------------------------
6) 使用场景
-----------------------------------
- 检查恶意流量检测模型是否因强泄露特征而虚高
- 分析数据集是否存在隐含泄露（如五元组泄露）
- 评估新增特征是否可能导致数据泄露风险
- 在模型训练前执行数据健康检查（Data Health Check）
"""

import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import argparse
import sys
import os
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score
import ipaddress
import json
import ast

import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger("PIL").setLevel(logging.INFO)
logging.getLogger("PIL.PngImagePlugin").setLevel(logging.WARNING)

import matplotlib
matplotlib.rcParams.update({
    "figure.max_open_warning": 0,  # disable "More than 20 figures" warning
})

# 导入配置管理器和相关模块
try:
    # 添加../../utils目录到Python搜索路径
    utils_path = os.path.join(os.path.dirname(__file__), '..', 'utils')
    sys.path.insert(0, utils_path)    
    import config_manager as ConfigManager
    from logging_config import setup_preset_logging
    # 使用统一的日志配置
    logger = setup_preset_logging(log_level=logging.DEBUG)
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保项目结构正确，所有依赖模块可用")
    sys.exit(1)

# ========= 配置 =========

FIVE_TUPLE_COLS = [
    "flowmeter.id.orig_h",
    "flowmeter.id.orig_p",
    "flowmeter.id.resp_h",
    "flowmeter.id.resp_p",
    "flowmeter.proto",
]

TRAIN_RATIO = 0.7
SEED = 42


# ========= 输出辅助 =========

def color(text, c):
    COLORS = {
        "red": "\033[91m", "green": "\033[92m",
        "yellow": "\033[93m", "blue": "\033[94m",
        "end": "\033[0m",
    }
    return f"{COLORS.get(c,'')}{text}{COLORS['end']}"

def info(msg): print(color(f"[INFO] {msg}", "blue"))
def warn(msg): print(color(f"[WARN] {msg}", "yellow"))
def error(msg): print(color(f"[ERROR] {msg}", "red"))


# ========= Step 1: 五元组构建 =========

def build_five_tuple(df):
    df["five_tuple"] = df.apply(
        lambda r: tuple(r[c] for c in FIVE_TUPLE_COLS), axis=1
    )
    return df


# ---------------- Step 2: split 实现 ----------------

#CSV_PATH = "processed_data/CIC-IDS-2017/all_embedded_flow.csv"
FLOW_CSV_PATH = os.path.join(ConfigManager.read_plot_data_path_config(), "all_embedded_flow.csv")
SESSION_CSV_PATH = os.path.join(ConfigManager.read_plot_data_path_config(), "all_split_session.csv")

def split_flow(df):
    """flow-level split"""
    labels = df["is_malicious"].values
    train_df, temp_df = train_test_split(
        df, test_size=1 - TRAIN_RATIO, stratify=labels, random_state=SEED
    )

    temp_labels = temp_df["is_malicious"].values
    val_df, test_df = train_test_split(temp_df, test_size=0.5,
                                      stratify=temp_labels, random_state=SEED)
    return train_df, val_df, test_df


def split_session_by_index(df_flow, session_csv_path=SESSION_CSV_PATH):
    """
    使用 SESSION_CSV_PATH 中的 split 划分结果，
    将 split 标签传播回 flow CSV。
    """
    info(f"Loading session split rules: {session_csv_path}")
    df_sess = pd.read_csv(session_csv_path, low_memory=False)

    split_map = {}
    for _, row in df_sess.iterrows():
        if pd.isna(row["flow_uid_list"]):
            continue
        uids = eval(row["flow_uid_list"])  # 字符串转列表
        for uid in uids:
            split_map[uid] = row["split"]  # train/validate/test

    # 添加 split 列
    if "uid" not in df_flow.columns:
        raise RuntimeError("Flow CSV 中未找到 uid 列，无法执行 session split")

    df_flow["split"] = df_flow["uid"].map(split_map)

    missing = df_flow["split"].isna().mean() * 100
    if missing > 1e-6:
        warn(f"❗ {missing:.2f}% flows missing split mapping")

    # 过滤掉 split 未定义的数据
    df_flow = df_flow[df_flow["split"].notna()]

    train_df = df_flow[df_flow["split"] == "train"]
    val_df   = df_flow[df_flow["split"] == "validate"]
    test_df  = df_flow[df_flow["split"] == "test"]

    info(f"Session split summary:")
    info(f"Train: {len(train_df):,} rows")
    info(f"Val:   {len(val_df):,} rows")
    info(f"Test:  {len(test_df):,} rows")

    return train_df, val_df, test_df


# ========= Step 3: 五元组泄露统计 =========

def check_five_tuple_leakage(train_df, test_df, split_mode="flow"):
    train_set = set(train_df["five_tuple"])
    test_set = set(test_df["five_tuple"])
    shared = train_set & test_set

    leakage_ratio = len(shared) / len(test_set) * 100

    print(f"\n=== Step 2: 五元组泄露情况 ({split_mode}-split) ===")
    print(f"Train 五元组数量: {len(train_set):,}")
    print(f"Test 五元组数量 : {len(test_set):,}")
    print(f"共享五元组数量 : {len(shared):,}")
    print(color(f"🔥 五元组泄露比例: {leakage_ratio:.2f}%", 
                "red" if leakage_ratio > 1 else "green"))

    if len(shared) > 0:
        print("\n示例共享五元组（前 10 项）:")
        print(list(shared)[:10])

    print("""
📌【关于五元组泄露的说明】

"五元组泄露" 意味着：同一个 (srcIP, srcPort, dstIP, dstPort, proto)
同时出现在了 Train 和 Test 中，这将导致模型：
    ✓ 记忆五元组模式而不是泛化
    ✓ Flow-Split 时性能虚高
    ✓ Session-Split 才能真实评价泛化性能
""")


# ========= Step 4: 五元组标签冲突 =========

def check_label_conflicts(train_df):
    group = train_df.groupby("five_tuple")["is_malicious"].unique()
    conflicts = [(k, v.tolist()) for k, v in group.items() if len(v) > 1]

    print("\n=== Step 3: 五元组标签冲突检测（Train 内部） ===")
    print(f"Train 中发现标签冲突的五元组数量: {len(conflicts):,}")

    if conflicts:
        print("\n示例冲突（前 10 项）:")
        print(conflicts[:10])

    print("""
    📌【说明】

    标签冲突 = 同一五元组在训练数据中同时存在恶意与正常样本。
    
    说明：
        • 五元组无法唯一对应恶意行为
        • 模型不能依赖五元组结构记忆
        • 攻击者复用同一 IP/端口进行多类型通信

    冲突越多 → 越说明需要 session 或内容级别特征建模
    """)

    return conflicts


# ========= 工具函数：类别型特征的条件熵 =========

def conditional_entropy(col_series, target_series):
    """
    H(Y | X) = Σ P(X=x) * H(Y | X=x)
    越低 = 泄露越强
    """
    eps = 1e-9
    df = pd.DataFrame({"x": col_series, "y": target_series}).dropna()

    H = 0
    for v, subset in df.groupby("x"):
        p_x = len(subset) / len(df)
        p1 = (subset["y"] == 1).mean()
        p0 = 1 - p1
        h = - (p1 * np.log2(p1 + eps) + p0 * np.log2(p0 + eps))
        H += p_x * h
    return H

class FeaturePatternDetector:

    # ========== 1) 检测是否为 embedding/list-like ==========
    def _safe_sample_series(self, series, max_n=200):
        """从非空值中安全采样一部分，用于做模式识别。"""
        s = series.dropna()
        if s.empty:
            return s
        if len(s) > max_n:
            return s.sample(max_n, random_state=0)
        return s

    def _parse_numeric_array(self, value):
        """尽量把一个值解析成数值数组（用于 sequence/embedding 判别）。"""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None

        # 如果本身就是 list/tuple
        if isinstance(value, (list, tuple)):
            vals = []
            for x in value:
                try:
                    vals.append(float(x))
                except Exception:
                    return None
            return vals if len(vals) > 0 else None

        # 字符串情况
        if isinstance(value, str):
            v = value.strip()
            if v == "":
                return None
            # 尝试 JSON
            try:
                obj = json.loads(v)
                if isinstance(obj, (list, tuple)):
                    vals = []
                    for x in obj:
                        try:
                            vals.append(float(x))
                        except Exception:
                            return None
                    return vals if len(vals) > 0 else None
            except Exception:
                pass
            # 尝试 Python literal
            try:
                obj = ast.literal_eval(v)
                if isinstance(obj, (list, tuple)):
                    vals = []
                    for x in obj:
                        try:
                            vals.append(float(x))
                        except Exception:
                            return None
                    return vals if len(vals) > 0 else None
            except Exception:
                pass
            # 尝试用逗号分割的简单形式
            if "," in v:
                parts = [p.strip() for p in v.split(",")]
                vals = []
                for p in parts:
                    try:
                        vals.append(float(p))
                    except Exception:
                        return None
                return vals if len(vals) > 0 else None

        return None

    def detect_array_semantic(self, col_name: str, series, sample_n: int = 100):
        """
        尝试区分：
        - embedding 向量：长度固定、较小（比如 <=128），名字中常带 freq/embedding/vector 等
        - sequence 序列：长度变化较大，或名字中带 packet/bulk/iat 等
        """
        name = col_name.lower()
        s = self._safe_sample_series(series, max_n=sample_n)

        lengths = []
        ok = 0
        for v in s:
            arr = self._parse_numeric_array(v)
            if arr is None:
                continue
            ok += 1
            lengths.append(len(arr))

        if ok < max(3, len(s) * 0.3):
            # 可解析的太少，认为不是数组型
            return None

        if not lengths:
            return None

        min_len, max_len = min(lengths), max(lengths)
        span = max_len - min_len

        # 名字特征
        name_has_packet = any(k in name for k in ["packet", "bulk", "seq", "sequence", "iat"])
        name_has_embedding = any(k in name for k in ["embedding", "emb", "freq", "vector", "hist"])

        # embedding：长度较小 & 基本固定 & 名字像 embedding/freq
        if max_len <= 256 and span <= max_len * 0.1 and name_has_embedding:
            return "embedding"

        # sequence：长度波动较大 或 名字明显是序列
        if span > max_len * 0.1 or name_has_packet:
            return "sequence"

        # 如果长度固定但没有明显名字线索，默认也偏 embedding
        if max_len <= 256 and span == 0:
            return "embedding"

        # 否则当成 sequence
        return "sequence"

    # ========== 2) 检测是否为 IP 地址 ==========

    def _looks_like_ipv4(self, s: str) -> bool:
        try:
            ipaddress.IPv4Address(s)
            return True
        except Exception:
            return False

    def _looks_like_ipv6(self, s: str) -> bool:
        try:
            ipaddress.IPv6Address(s)
            return True
        except Exception:
            return False

    def looks_like_ip(self, colname, series):
        name = colname.lower()

        # 特征名 heuristic
        name_ip_hint = any(h in name for h in [
            "id.orig_h", "id.resp_h", ".orig_h", ".resp_h"
        ])

        s = self._safe_sample_series(series.astype(str).str.strip())
        if s.empty:
            return False

        valid_cnt = 0
        checked = 0
        for v in s:
            v = v.split(',')[0].strip()

            if v in ["", "nan", "none", "-", "null", "unknown"]:
                continue
            checked += 1

            if self._looks_like_ipv4(v) or self._looks_like_ipv6(v):
                valid_cnt += 1

        if checked == 0:
            return False

        ratio = valid_cnt / checked

        # 只要名称提示 + ≥20% 样本合法即可视为 IP
        if name_ip_hint and ratio >= 0.20:
            return True

        return ratio >= 0.40

    # ========== 3) 检测是否为 domain ==========
    def looks_like_domain(self, colname, series):
        s = self._safe_sample_series(series.astype(str).str.strip().str.lower())
        if s.empty:
            return False
        
        tlds = [".com", ".net", ".org", ".cn", ".io", ".edu", ".gov"]
        cnt = 0
        for v in s:
            if " " in v or "@" in v:
                continue
            if any(tld in v for tld in tlds):
                cnt += 1
        return cnt >= max(3, len(s) * 0.5)

    # ========== 4) 检测是否为 cipher 名称 ==========
    def looks_like_protocol_version(self, colname, series):
        name = colname.lower()
        if "version" not in name:
            return False

        SAMPLE = self._safe_sample_series(series.astype(str))
        cnt = 0
        for v in SAMPLE:
            v_lower = v.lower()
            if v_lower.startswith("tls") or v_lower.startswith("ssl"):
                cnt += 1

        return cnt >= max(3, len(SAMPLE) * 0.5)
    
    def looks_like_cipher(self, series):
        sample = series.dropna().astype(str).head(20)
        cnt = sum("TLS" in v.upper() or "AES" in v.upper() or "CHACHA" in v.upper()
                for v in sample)
        return cnt >= len(sample) * 0.5

    # ========== 5) 检测是否为 port（端口） ==========
    def looks_like_port(self, colname, series):
        name = colname.lower()

        # 必须严格匹配端口字段命名，不允许统计学关键字
        PORT_HINTS = [
            ".orig_p", ".resp_p",
            "id.orig_p", "id.resp_p"
        ]
        if not any(name.endswith(h) for h in PORT_HINTS):
            return False
        
        # 禁止：统计 / 时间 / 包长度等字段被当端口
        BAD_HINTS = [
            "max", "min", "std", "avg", "mean", "tot",
            "size", "window", "payload", "pkts", "bytes"
        ]
        if any(b in name for b in BAD_HINTS):
            return False

        # 数值验证
        s = pd.to_numeric(self._safe_sample_series(series), errors="coerce").dropna()
        if s.empty:
            return False

        return s.min() >= 0 and s.max() <= 65535    
    
    # ========== 6) 数值字符串自动判断 ==========
    def numeric_string_type(self, series):
        s = series.dropna().astype(str).str.strip()
        if len(s) == 0:
            return None
        # int
        if s.str.fullmatch(r"^-?\d+$").all():
            return "int"
        # float
        if s.str.fullmatch(r"^-?\d+\.\d+$").all():
            return "float"
        return None

    # ========== 🏆 主函数：自动类型识别 ==========
    def detect_feature_type(self, colname, series):
        """
        返回:
            dtype_str: string, int64/float64/object/embedding/ip/domain/cipher
            feature_type: numeric / categorical / embedding / ignore / ip / domain / cipher / port
        """
        # ---------- 0) embedding 检测 ----------
        array_sem = self.detect_array_semantic(colname, series)
        if array_sem == "embedding":
            return "embedding", "embedding"
        elif array_sem == "sequence":
            return "sequence", "categorical"

        # ---------- 1) IP 地址 ----------
        if self.looks_like_ip(colname, series):
            return "ip", "categorical"   # IP 适合作为 categorical

        # ---------- 2) Domain ----------
        if self.looks_like_domain(colname, series):
            return "domain", "categorical"

        # ---------- 3) Cipher ----------
        if self.looks_like_protocol_version(colname, series):
            return "protocol_version", "categorical"

        if self.looks_like_cipher(series):
            return "cipher", "categorical"

        # ---------- 4) Port ----------
        if self.looks_like_port(colname, series):
            return "port", "categorical"

        # ---------- 5) 数值型 detection (包含 numeric string) ----------
        dtype = str(series.dtype)

        # 优先识别 object 里的数字字符串
        if series.dtype == object:
            numeric_type = self.numeric_string_type(series)
            if numeric_type == "int":
                return "int64", "numeric"
            if numeric_type == "float":
                return "float64", "numeric"

        # pandas int/float
        if dtype.startswith("int"):
            return "int64", "numeric"
        if dtype.startswith("float"):
            return "float64", "numeric"

        # ---------- 6) 默认作为 categorical ----------
        return "object", "categorical"

# ========= Step 5: 扩展版高危特征泄露扫描 =========

def check_high_risk_feature_leakage(train_df, test_df, top_k=100):
    print("\n=== Step 4: 高危特征泄露检查（扩展版） ===")

    high_risk_results = []

    for col in train_df.columns:
        if col in ["five_tuple", "is_malicious"]:
            continue

        # 跳过 99% 以上为空的列（无意义）
        if train_df[col].isna().mean() > 0.99:
            continue

        filtered_train_df = filter_invalid_embedding_rows(train_df, col)
        if filtered_train_df.empty:
            continue  # 整列都无效，跳过

        filtered_test_df  = filter_invalid_embedding_rows(test_df, col)

        # ========= 类型判断 =========
        detector = FeaturePatternDetector()
        dtype_str, feature_type = detector.detect_feature_type(col, train_df[col])

        # ========= 类别型特征 =========
        is_categorical = (feature_type == "categorical")
        is_numeric =  (feature_type == "numeric")
        if is_categorical:
            # ========= 1）字符串或Categorical特征 =========
            # =============== 计算Leakage Ratio ===============
            # 过滤无效值
            train_valid = filtered_train_df[col].dropna().astype(str).str.strip()
            test_valid  = filtered_test_df[col].dropna().astype(str).str.strip()

            min_support_train = 3  # 可调，建议 3~5
            min_support_test = 1

            # 仅保留出现次数 >= min_support 的类别
            train_valid = train_valid[train_valid.groupby(train_valid).transform("count") >= min_support_train]
            test_valid  = test_valid[test_valid.groupby(test_valid).transform("count") >= min_support_test]

            train_u = set(train_valid[train_valid != ""])
            test_u  = set(test_valid[test_valid != ""])

            if len(test_u) == 0:
                continue

            shared = train_u & test_u
            leakage_ratio = len(shared) / len(test_u) * 100

            # =============== 计算特征列和目标列的条件熵 ===============
            valid_mask = filtered_train_df[col].notna()
            if valid_mask.sum() > 0:
                col_series = filtered_train_df.loc[valid_mask, col].astype(str)
                target_series = filtered_train_df.loc[valid_mask, "is_malicious"]

                if col_series.nunique() > 1:
                    cond_ent = conditional_entropy(col_series, target_series)
                else:
                    cond_ent = None
            else:
                cond_ent = None

            na_ratio = compute_na_ratio(train_df, test_df, col)

            leakage_score = leakage_risk_score(leakage_ratio, cond_ent, na_ratio)
            risk_lv = risk_level(leakage_score)

            high_risk_results.append((col, "categorical", dtype_str, leakage_ratio, cond_ent, na_ratio, leakage_score, risk_lv))
            
        elif is_numeric:
            # ========= 2) 数值型特征（增强版泄露检测） =========
            # Pandas 会把部分列识别成 object，但实际是数字，需要强制转换
            try:
                numeric_col = pd.to_numeric(train_df[col], errors="coerce")
            except Exception:
                continue

            if numeric_col.nunique() <= 1:
                continue

            numeric_valid = numeric_col[train_df[col].notna()]
            label_valid = train_df.loc[numeric_valid.index, "is_malicious"]

            # Pearson
            pearson_corr = abs(numeric_valid.corr(label_valid))
            if np.isnan(pearson_corr):
                continue

            # Mutual Information
            try:
                MI = mutual_info_classif(
                    numeric_valid.values.reshape(-1, 1),
                    label_valid.values,
                    discrete_features=False
                )[0]
                MI_norm = MI / np.log1p(min(50, numeric_valid.nunique()))
            except Exception:
                MI_norm = 0.0

            # AUC Score (用 Logistic/随机分类器)
            try:
                auc_score = roc_auc_score(
                    label_valid,
                    numeric_valid.fillna(numeric_valid.mean())
                )
            except Exception:
                auc_score = 0.5  # 不可区分

            # 数值泄露综合分
            numeric_leak_score = (
                0.2 * pearson_corr +
                0.5 * MI_norm +
                0.3 * auc_score
            )

            risk_lv = risk_level(numeric_leak_score)

            high_risk_results.append((
                col, "numeric", dtype_str,
                pearson_corr, MI_norm, auc_score,
                numeric_leak_score, risk_lv
            ))
    # ============ 输出可配置 top_k 的结果 ============

    # ======== 1）打印类别特征结果 =========
    print(color(f"\n🔥 最强泄露的类别型特征 TOP {top_k}:", "red"))
    print("注意：条件熵越小，特征对标签越有用（相关性更强）")
    cat_feats = sorted(
        [r for r in high_risk_results if r[1] == "categorical"],
        key=lambda x: -x[6])[:top_k] # x[6] 是综合得分 leakage_score

    for col, _, dtype, leak, cond_ent, na, leak_score, risk_lvl in cat_feats:
        ce_str = f"{cond_ent:.4f}" if cond_ent is not None else "N/A"
        print(f"{col:45s} | 类型={dtype:10s} | 泄露率={leak:6.2f}% | 条件熵={ce_str} | N/A率={na:.3f} | 泄露综合分={leak_score:.4f} | 泄露等级={risk_lvl}")

    print("注意：N/A 表示该特征值唯一或几乎唯一无法判断熵")
    print("注意：条件熵越小，特征对标签越有用（相关性更强）")

    # ======== 2）打印数值型特征结果 =========
    print(color(f"\n🔥 最强泄露的数值型特征 TOP {top_k} (按泄露综合分):", "red"))
    print("注意：MI 和 AUC 能捕获非线性与可分类泄露风险")

    num_feats = sorted(
        [r for r in high_risk_results if r[1] == "numeric"],
        key=lambda x: -x[6]  # x[6] 是 numeric_leak_score
    )[:top_k]

    for col, _, dtype, pearson_corr, MI_norm, auc_score, leak_score, risk_lvl in num_feats:
        print(
            f"{col:45s} | 类型={dtype:10s}"
            f" | Pearson={pearson_corr:.4f}"
            f" | MI={MI_norm:.4f}"
            f" | AUC={auc_score:.4f}"
            f" | 泄露综合分={leak_score:.4f}"
            f" | 泄露等级={risk_lvl}"
        )

    return high_risk_results

def leakage_risk_score(leak_ratio, ce, na_ratio):
    """泄露综合得分：泄露率 × (1-CE_norm) × (1-NA_ratio)"""
    if ce is None:
        return 0.0  # 无统计意义
    ce_norm = min(max(ce, 0.0), 1.0)
    return (leak_ratio / 100.0) * (1.0 - ce_norm) * (1.0 - na_ratio)


def compute_na_ratio(df_train, df_test, col, zero_tol=1e-12):
    """
    计算 N/A 比例：
    - 常规列：只把 NaN 视为缺失
    - 类 embedding 列（如 dns.query*_freq / ssl.server_name*_freq）：
        * NaN
        * 解析为向量后 “全 0” 或 “常数向量”
      都视为 N/A
    """
    s_train = df_train[col]
    s_test  = df_test[col]

    total = len(s_train) + len(s_test)
    if total == 0:
        return 1.0

    # 先统计普通 NaN
    na_mask_train = s_train.isna()
    na_mask_test  = s_test.isna()

    def is_zero_or_const_embedding(v):
        """
        仅在 “看起来像向量” 的情况下，额外判断是否全 0 / 常数。
        否则一律返回 False（避免误伤普通字符串或数值列）。
        """
        # NaN 直接交给外层 na_mask 处理
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return False

        # 字符串：必须像 "[0.1, 0.2]" 或 "0.1,0.2" 才当成 embedding
        if isinstance(v, str):
            txt = v.strip()
            if not txt:
                return False
            if not (("[" in txt and "]" in txt) or ("," in txt)):
                return False
            try:
                vals = [float(x) for x in txt.strip("[]").split(",") if x.strip() != ""]
            except Exception:
                return False

        # list / ndarray：直接当向量处理
        elif isinstance(v, (list, np.ndarray)):
            vals = list(v)

        else:
            # 其它类型（纯 float/int/string 无逗号），不当 embedding
            return False

        if len(vals) == 0:
            return True  # 视为无信息

        # 全 0 或常数向量 → 视为 N/A
        if all(abs(x) < zero_tol for x in vals):
            return True
        if len(set(vals)) == 1:
            return True
        return False

    # 只对当前列跑一次 apply，代价还可以接受
    extra_na_train = s_train.apply(is_zero_or_const_embedding)
    extra_na_test  = s_test.apply(is_zero_or_const_embedding)

    na_count = (na_mask_train | extra_na_train).sum() + (na_mask_test | extra_na_test).sum()
    return na_count / total


def risk_level(score):
    if score >= 0.50: return "CRITICAL🔥"
    if score >= 0.30: return "HIGH🚨"
    if score >= 0.10: return "MEDIUM⚠️"
    if score > 0.00:  return "LOW🙂"
    return "NONE"

def filter_invalid_embedding_rows(df, col, zero_tol=1e-12):
    """仅处理 Embedding 列。如果内容不是可解析列表，直接返回原 df[col]"""
    valid_mask = []
    parse_failed = False

    for v in df[col]:
        if isinstance(v, str):
            # 必须同时满足：有括号且有逗号，才视为向量
            if not (("[" in v and "]" in v) or ("," in v)):
                parse_failed = True
                break

            try:
                vals = [float(x) for x in v.strip("[]").split(",") if x]
            except Exception:
                parse_failed = True
                break

        elif isinstance(v, (list, np.ndarray)):
            vals = list(v)

        else:
            # 非向量，不处理
            parse_failed = True
            break

        if len(vals) == 0:
            valid_mask.append(False)
            continue
        
        # 过滤纯零或常数向量
        if all(abs(x) < zero_tol for x in vals) or len(set(vals)) == 1:
            valid_mask.append(False)
        else:
            valid_mask.append(True)

    # 👉 如果该列不是 embedding 列，直接返回原 df
    if parse_failed:
        return df
    
    return df[valid_mask]


def save_csv_reports(split_name, high_risk_results, output_dir="leakage_reports"):
    os.makedirs(output_dir, exist_ok=True)

    # 分离 categorical 和 numeric 特征
    cat_rows = [r for r in high_risk_results if r[1] == "categorical"]
    num_rows = [r for r in high_risk_results if r[1] == "numeric"]

    # 类别型特征表头
    cat_columns = [
        "feature",
        "type",
        "dtype",
        "leak_ratio",
        "conditional_entropy",
        "na_ratio",
        "leakage_score",
        "severity"
    ]
    df_cat = pd.DataFrame(cat_rows, columns=cat_columns)
    cat_path = os.path.join(output_dir,
        f"split_{split_name}_feature_leakage_categorical_top{len(cat_rows)}.csv"
    )
    df_cat.to_csv(cat_path, index=False, encoding="utf-8")
    print(f"📁 分类泄露报告保存: {cat_path}")

    # 数值型特征表头
    num_columns = [
        "feature",
        "type",
        "dtype",
        "pearson_corr",
        "mutual_info_norm",
        "auc_score",
        "leakage_score",
        "severity"
    ]
    df_num = pd.DataFrame(num_rows, columns=num_columns)
    num_path = os.path.join(output_dir, 
        f"split_{split_name}_feature_leakage_numeric_top{len(num_rows)}.csv"
    )
    df_num.to_csv(num_path, index=False, encoding="utf-8")
    print(f"📁 数值泄露报告保存: {num_path}")


def plot_leakage_bar(split_name, high_risk_results, top_k=50, output_dir="leakage_reports"):
    os.makedirs(output_dir, exist_ok=True)

    # ===== Categorical: 用泄露率 =====
    cat_results = sorted(
        [r for r in high_risk_results if r[1] == "categorical"],
        key=lambda x: -x[3]   # leakage_ratio %
    )[:top_k]

    if len(cat_results) > 0:
        feats = [r[0] for r in cat_results]
        leaks = [r[3] for r in cat_results]

        plt.figure(figsize=(12, 6))
        plt.barh(feats, leaks)
        plt.xlabel("Leakage Ratio (%)")
        plt.ylabel("Feature")
        plt.title(f"Categorical Leakage Ratio Top-{len(feats)} ({split_name})")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"split_{split_name}_categorical_leakage_bar.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"📊 Saved: {out_path}")

    else:
        print("⚠ No categorical features for bar plot.")

    # ===== Numeric: 用泄露综合分 =====
    num_results = sorted(
        [r for r in high_risk_results if r[1] == "numeric"],
        key=lambda x: -x[6]  # leakage_score
    )[:top_k]

    if len(num_results) > 0:
        feats = [r[0] for r in num_results]
        scores = [r[6] for r in num_results]

        plt.figure(figsize=(12, 6))
        plt.barh(feats, scores)
        plt.xlabel("Leakage Score")
        plt.ylabel("Feature")
        plt.title(f"Numeric Leakage Score Top-{len(feats)} ({split_name})")
        plt.tight_layout()

        out_path = os.path.join(output_dir, f"split_{split_name}_numeric_leakage_bar.png")
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"📈 Saved: {out_path}")
    else:
        print("⚠ No numeric features for bar plot.")

def plot_leakage_radar(split_name, high_risk_results, top_k=10, output_dir="leakage_reports"):
    os.makedirs(output_dir, exist_ok=True)

    # ---------------- Categorical Radar ---------------- #
    cat_feats = [
        r for r in high_risk_results if r[1] == "categorical"
    ]
    cat_feats = sorted(cat_feats, key=lambda x: -x[3])[:top_k]  # 使用 leakage_ratio

    if len(cat_feats) >= 3:  # 至少需要3个，否则不是有效雷达图
        feats = [r[0] for r in cat_feats]
        leak_vals = [float(r[3]) for r in cat_feats]  # r[3] 泄露率

        N = len(feats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        leak_vals += leak_vals[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, leak_vals, marker="o")
        ax.fill(angles, leak_vals, alpha=0.3)
        ax.set_thetagrids(np.degrees(angles[:-1]), feats)

        plt.title(f"Categorical Leakage Radar (Top-{N}) ({split_name})")
        out_path = os.path.join(output_dir, f"split_{split_name}_categorical_leakage_radar.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"📈 Radar Chart saved to {out_path}")
    else:
        print("⚠ Too few categorical features for radar plot.")

    # ---------------- Numeric Radar ---------------- #
    num_feats = [
        r for r in high_risk_results if r[1] == "numeric"
    ]
    num_feats = sorted(num_feats, key=lambda x: -x[6])[:top_k]  # 使用 numeric_leak_score = r[6]

    if len(num_feats) >= 3:
        feats = [r[0] for r in num_feats]
        leak_vals = [float(r[6]) for r in num_feats]  # r[6] 为 numeric_leak_score

        N = len(feats)
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        angles += angles[:1]
        leak_vals += leak_vals[:1]

        plt.figure(figsize=(8, 8))
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, leak_vals, marker="o")
        ax.fill(angles, leak_vals, alpha=0.3)
        ax.set_thetagrids(np.degrees(angles[:-1]), feats)

        plt.title(f"Numeric Leakage Radar (Top-{N}) ({split_name})")
        out_path = os.path.join(output_dir, f"split_{split_name}_numeric_leakage_radar.png")
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"📈 Radar Chart saved to {out_path}")
    else:
        print("⚠ Too few numeric features for radar plot.")


# ========= 主程序 =========

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_mode", type=str, default="flow",
                        choices=["flow","session"], help="flow or session split")
    parser.add_argument("--topk", type=int, default=200)
    args = parser.parse_args()

    output_dir = os.path.join(ConfigManager.read_plot_data_path_config(), "leakage_reports")
    os.makedirs(output_dir, exist_ok=True)
    log_path = os.path.join(output_dir, f"split_{args.split_mode}_leakage_analysis.log")

    class DualLogger(object):
        def __init__(self, *files):
            self.files = files
        def write(self, msg):
            for f in self.files:
                f.write(msg)
                f.flush()
        def flush(self):
            for f in self.files:
                f.flush()

    log_file = open(log_path, "w", encoding="utf-8")
    sys.stdout = DualLogger(sys.stdout, log_file)
    sys.stderr = DualLogger(sys.stderr, log_file)

    print(f"🔍 所有输出已同时写入日志文件：{log_path}")

    info(f"Loading dataset: {FLOW_CSV_PATH}")
    df = pd.read_csv(FLOW_CSV_PATH, low_memory=False)
    info(f"Loaded {len(df):,} rows")

    # Step 1
    df = build_five_tuple(df)

    # Step 2: split
    if args.split_mode == "flow":
        train_df, val_df, test_df = split_flow(df)
    else:
        train_df, val_df, test_df = split_session_by_index(df)

    # Step 3: leakage
    check_five_tuple_leakage(train_df, test_df, args.split_mode)

    # Step 4: label conflicts
    check_label_conflicts(train_df)

    # Step 5: feature leakage
    high_risk_results = check_high_risk_feature_leakage(train_df, test_df, args.topk)

    print("\n=== Done ===")

    # 当前这个 split 的名字，用来区分输出文件
    split_name = f"{args.split_mode}_split"

    # 保存 CSV
    save_csv_reports(split_name, high_risk_results, output_dir=output_dir)

    # 可视化
    plot_leakage_bar(split_name, high_risk_results, top_k=args.topk, output_dir=output_dir)
    plot_leakage_radar(split_name, high_risk_results, top_k=min(10, args.topk), output_dir=output_dir)


if __name__ == "__main__":
    main()
