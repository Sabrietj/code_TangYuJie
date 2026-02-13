import os
import pandas as pd
import numpy as np
import ast
from tqdm import tqdm

# ================= 配置 =================
# 指向 embed_feature 输出的合并文件目录
DATASET_DIR = "/root/autodl-fs/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"


# =======================================

def verify_merged_sorted():
    print(f"🔍 [Step 2] 开始校验合并后的全局时序: {DATASET_DIR}")

    flow_path = os.path.join(DATASET_DIR, "all_flow.csv")
    session_path = os.path.join(DATASET_DIR, "all_session.csv")

    if not os.path.exists(flow_path) or not os.path.exists(session_path):
        print("❌ 错误: all_flow.csv 或 all_session.csv 不存在，请先运行合并代码！")
        return

    # --- 1. 校验 Flow 顺序 ---
    print("   正在读取 all_flow.csv (只读时间列)...")
    # 尝试读取 conn.ts 或 ts
    df_flow_header = pd.read_csv(flow_path, nrows=0)
    ts_col = 'conn.ts' if 'conn.ts' in df_flow_header.columns else 'ts'

    # 逐块读取以节省内存，检查单调性
    prev_ts = -1.0
    flow_is_sorted = True
    row_count = 0

    chunksize = 500000
    for chunk in tqdm(pd.read_csv(flow_path, usecols=[ts_col], chunksize=chunksize), desc="校验 Flow 时序"):
        ts_values = chunk[ts_col].values

        # 检查当前块内部是否排序
        if not (np.diff(ts_values) >= 0).all():
            flow_is_sorted = False
            break

        # 检查与上一块的连接处
        if row_count > 0:
            if ts_values[0] < prev_ts:
                flow_is_sorted = False
                break

        prev_ts = ts_values[-1]
        row_count += len(chunk)

    if flow_is_sorted:
        print(f"✅ Flow 数据严格按时间升序排列 (共 {row_count} 行)")
    else:
        print(f"❌ Flow 数据存在乱序！请检查 merge_csv_files 中的排序逻辑。")
        return  # Flow 乱序则无需继续检查 Session

    # --- 2. 校验 Session 顺序 ---
    print("\n   正在校验 Session 时序 (这需要加载 UID 映射，稍慢)...")

    # 加载 Flow UID -> TS 映射
    print("   加载 Flow 索引...")
    df_flow = pd.read_csv(flow_path, usecols=['uid', ts_col])
    uid_to_time = dict(zip(df_flow['uid'], df_flow[ts_col]))

    print("   读取 Session 并计算真实时间...")
    df_session = pd.read_csv(session_path)

    def get_start_time(uid_list_str):
        try:
            if isinstance(uid_list_str, str):
                uids = ast.literal_eval(uid_list_str)
            else:
                uids = uid_list_str
            if not uids: return -1
            # 这里的逻辑必须和 merge 代码一致：取 min
            return min([uid_to_time.get(uid, float('inf')) for uid in uids])
        except:
            return -1

    # 计算时间
    tqdm.pandas(desc="计算 Session 时间")
    df_session['calc_ts'] = df_session['flow_uid_list'].progress_apply(get_start_time)

    # 检查单调性
    # 过滤掉无法计算时间的行（通常是 -1 或 inf）
    valid_ts = df_session[df_session['calc_ts'] > 0]['calc_ts'].values

    if (np.diff(valid_ts) >= 0).all():
        print(f"✅ [Step 2] Session 数据严格按时间升序排列！")
        print(f"   Validation Passed. 你的数据已经准备好进行概念漂移实验了。")
    else:
        # 找出错误位置
        diffs = np.diff(valid_ts)
        error_indices = np.where(diffs < 0)[0]
        print(f"❌ [Step 2] Session 数据存在乱序！")
        print(f"   发现 {len(error_indices)} 处时间倒流。")
        print(
            f"   示例错误: 索引 {error_indices[0]} 的时间 {valid_ts[error_indices[0]]} > 索引 {error_indices[0] + 1} 的时间 {valid_ts[error_indices[0] + 1]}")


if __name__ == "__main__":
    verify_merged_sorted()