import pandas as pd
import numpy as np
import os
import ast
import sys
from tqdm import tqdm
from datetime import datetime

# ================= 配置区域 =================
# 请根据你的实际路径修改这里
BASE_DIR = "/root/autodl-fs/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
FLOW_CSV = os.path.join(BASE_DIR, "all_flow.csv")
SESSION_CSV = os.path.join(BASE_DIR, "all_split_session.csv")

# 采样率 (模拟训练配置)
SAMPLE_RATIO = 0.1

# 列名配置
TIME_COL = "conn.ts"  # 时间戳列名
LABEL_COL = "label"  # 标签列名 (或 multiclass_label)
UID_COL = "uid"


# ===========================================

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def visualize_temporal_labels(df, title, num_buckets=50):
    """
    在时间轴上可视化标签分布，确认是否呈“阶梯状”变化
    """
    if df.empty:
        print(f"⚠️ {title} 数据为空")
        return

    df = df.sort_values(TIME_COL)
    start_ts = df[TIME_COL].min()
    end_ts = df[TIME_COL].max()
    duration = end_ts - start_ts

    print(f"\n📊 {title} - 标签时序分布 (时间跨度: {duration / 3600:.2f} 小时)")
    print(f"   时间范围: {datetime.fromtimestamp(start_ts)} -> {datetime.fromtimestamp(end_ts)}")
    print("-" * 60)
    print(f"{'时间进度':<15} | {'主要标签 (Top 3)':<40}")
    print("-" * 60)

    # 分桶统计
    df['bucket'] = pd.cut(df[TIME_COL], bins=num_buckets, labels=False)

    for i in range(num_buckets):
        bucket_data = df[df['bucket'] == i]
        if bucket_data.empty:
            continue

        # 统计该时间段的标签
        counts = bucket_data[LABEL_COL].value_counts().head(3)
        label_str = ", ".join([f"{k}({v})" for k, v in counts.items()])

        # 计算时间点
        current_time = start_ts + (duration * (i / num_buckets))
        time_str = datetime.fromtimestamp(current_time).strftime('%m-%d %H:%M')

        print(f"{time_str:<15} | {label_str}")


def check_time_order(train_df, val_df, test_df):
    """严格检查时间有序性"""
    print("\n🔍 检查数据集时间有序性 (Train -> Val -> Test)")

    t_max_train = train_df[TIME_COL].max()
    t_min_val = val_df[TIME_COL].min()
    t_max_val = val_df[TIME_COL].max()
    t_min_test = test_df[TIME_COL].min()

    print(f"  Train Max Time: {datetime.fromtimestamp(t_max_train)}")
    print(f"  Val   Min Time: {datetime.fromtimestamp(t_min_val)}")

    if t_max_train > t_min_val:
        print("  ❌ 警告: 训练集和验证集时间重叠！(Flow模式随机打乱会导致此问题)")
    else:
        print("  ✅ 训练集完全在验证集之前")

    print(f"  Val   Max Time: {datetime.fromtimestamp(t_max_val)}")
    print(f"  Test  Min Time: {datetime.fromtimestamp(t_min_test)}")

    if t_max_val > t_min_test:
        print("  ❌ 警告: 验证集和测试集时间重叠！")
    else:
        print("  ✅ 验证集完全在测试集之前")


def load_flow_data_optimized():
    """只读取必要列，节省内存"""
    print(f"正在读取 Flow 数据: {FLOW_CSV} ...")
    cols = [UID_COL, TIME_COL, LABEL_COL]
    df = pd.read_csv(FLOW_CSV, usecols=cols, low_memory=False)
    # 确保按时间排序 (CIC-IDS-2017 应该是大致有序的，这里强制排序以模拟理想情况)
    df = df.sort_values(by=[TIME_COL])
    return df


# ==============================================================================
# 模拟 1: Flow Mode (有序采样 + 有序切分)
# ==============================================================================
def simulate_flow_mode(full_df):
    print_header("模拟 MODE: FLOW (Ordered Sampling + No Shuffle)")

    # 1. 有序采样 (Systematic Sampling)
    step = int(1 / SAMPLE_RATIO)
    sampled_df = full_df.iloc[::step].copy()
    print(f"采样后数据量: {len(sampled_df)} (Ratio: {SAMPLE_RATIO}, Step: {step})")

    # 2. 按顺序切分 (70% Train, 15% Val, 15% Test)
    n = len(sampled_df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)

    train_df = sampled_df.iloc[:train_end]
    val_df = sampled_df.iloc[train_end:val_end]
    test_df = sampled_df.iloc[val_end:]

    # 3. 验证
    check_time_order(train_df, val_df, test_df)
    visualize_temporal_labels(train_df, "Flow-Train")
    visualize_temporal_labels(val_df, "Flow-Validate")
    visualize_temporal_labels(test_df, "Flow-Test")


# ==============================================================================
# 模拟 2: Session Mode (读取 split 文件)
# ==============================================================================
def simulate_session_mode(full_df):
    print_header("模拟 MODE: SESSION (基于 all_split_session.csv)")

    if not os.path.exists(SESSION_CSV):
        print(f"❌ 找不到 {SESSION_CSV}，无法验证 Session 模式")
        return

    print(f"读取 Session Split: {SESSION_CSV} ...")
    session_df = pd.read_csv(SESSION_CSV, usecols=['flow_uid_list', 'split'])

    # 映射 UID -> Label/Time (用于快速查找)
    # full_df 已经只有必要列了

    # 提取 UID 集合
    print("解析 Session UID 列表...")

    def get_uids(split_name):
        subset = session_df[session_df['split'] == split_name]
        uids = set()
        for x in subset['flow_uid_list']:
            try:
                # 兼容字符串或列表格式
                l = ast.literal_eval(x) if isinstance(x, str) else x
                uids.update(l)
            except:
                pass
        return uids

    train_uids = get_uids('train')
    val_uids = get_uids('validate')
    test_uids = get_uids('test')

    print(f"Train UIDs: {len(train_uids)}, Val UIDs: {len(val_uids)}, Test UIDs: {len(test_uids)}")

    # 过滤 Flow
    print("正在根据 UID 过滤 Flow 数据...")
    # 使用 isin 过滤
    train_df = full_df[full_df[UID_COL].isin(train_uids)]
    val_df = full_df[full_df[UID_COL].isin(val_uids)]
    test_df = full_df[full_df[UID_COL].isin(test_uids)]

    # 采样 (Session模式通常也配合采样，这里模拟对 Flow 的结果采样，或者直接看全量)
    # 如果数据量太大，这里仅做可视化采样
    if len(train_df) > 100000:
        train_df = train_df.iloc[::10]
        val_df = val_df.iloc[::10]
        test_df = test_df.iloc[::10]
        print("⚠️ 为加速可视化，仅展示部分数据")

    # 3. 验证
    check_time_order(train_df, val_df, test_df)

    # 重点：展示标签变化
    visualize_temporal_labels(train_df, "Session-Train")
    visualize_temporal_labels(val_df, "Session-Validate")
    visualize_temporal_labels(test_df, "Session-Test")


# ==============================================================================
# 主程序
# ==============================================================================
if __name__ == "__main__":
    if not os.path.exists(FLOW_CSV):
        print(f"❌ 错误: 找不到文件 {FLOW_CSV}")
        sys.exit(1)

    # 1. 加载全量数据 (只加载一次)
    df_all = load_flow_data_optimized()

    # 2. 验证 Flow 模式
    simulate_flow_mode(df_all)

    # 3. 验证 Session 模式
    simulate_session_mode(df_all)

    print("\n✅ 验证完成。请检查上方日志中的时间重叠警告和标签分布。")
    print("   对于 CIC-IDS-2017，你应该看到标签随时间发生明显的类别切换 (例如从 BENIGN 变成 FTP-Patator)。")