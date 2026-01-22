import os
import pandas as pd
import numpy as np
import ast
from datetime import datetime

# ================= 配置 =================
DATASET_DIR = "/root/autodl-fs/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
SPLIT_FILE = "all_split_session.csv"  # 或者是 all_embedded_flow.csv 对应的 split
FLOW_FILE = "all_flow.csv"


# =======================================

def verify_split_leakage():
    split_path = os.path.join(DATASET_DIR, SPLIT_FILE)
    flow_path = os.path.join(DATASET_DIR, FLOW_FILE)

    print(f"🔍 [Step 3] 开始校验数据集划分与时间泄露: {split_path}")

    if not os.path.exists(split_path):
        print("❌ 错误: split 文件不存在！")
        return

    df_split = pd.read_csv(split_path, low_memory=False)

    if 'split' not in df_split.columns:
        print("❌ 错误: 文件中没有 'split' 列，无法校验划分！")
        return

    # 为了获取准确时间，我们需要再次关联 Flow 时间 (因为 Session 表里可能没有显式的 start_time 列)
    print("   正在加载 Flow 时间戳以验证精确时间边界...")
    df_flow = pd.read_csv(flow_path, usecols=['uid', 'conn.ts'])
    uid_to_time = dict(zip(df_flow['uid'], df_flow['conn.ts']))

    def get_session_time(row):
        try:
            uids = ast.literal_eval(row['flow_uid_list'])
            if not uids: return np.nan
            return min([uid_to_time.get(u, np.inf) for u in uids])
        except:
            return np.nan

    # 抽样检查或者全量检查 (全量较慢，这里为了安全做全量，你可以改用 sample)
    print("   计算所有 Session 的实际开始时间 (这可能需要几分钟)...")
    # 使用 vectorization 或者 apply
    # 这里为了代码简单直接用 apply，如果太慢可以只取首尾各1000条
    df_split['real_ts'] = df_split.apply(get_session_time, axis=1)

    # 去除无效时间
    df_clean = df_split.dropna(subset=['real_ts'])

    sets = ['train', 'validate', 'test']
    stats = {}

    print("\n📊 划分统计报告:")
    for s in sets:
        subset = df_clean[df_clean['split'] == s]
        if len(subset) == 0:
            print(f"⚠️ 警告: {s} 集为空！")
            stats[s] = {'min': -1, 'max': -1}
            continue

        min_ts = subset['real_ts'].min()
        max_ts = subset['real_ts'].max()

        stats[s] = {'min': min_ts, 'max': max_ts, 'count': len(subset)}

        t_min_str = datetime.fromtimestamp(min_ts).strftime('%Y-%m-%d %H:%M:%S')
        t_max_str = datetime.fromtimestamp(max_ts).strftime('%Y-%m-%d %H:%M:%S')

        print(f"   [{s.upper()}]: {len(subset)} 条")
        print(f"     时间范围: {t_min_str}  --->  {t_max_str}")
        print(f"     Timestamp: {min_ts:.2f} ---> {max_ts:.2f}")

    # --- 核心校验：时间界限 ---
    print("\n🛡️ 时间泄露校验:")
    leakage = False

    # Check Train vs Validate
    if stats['train']['max'] >= stats['validate']['min']:
        print(f"❌ [严重失败] Train 与 Validate 时间重叠！")
        print(f"   Train End ({stats['train']['max']}) >= Val Start ({stats['validate']['min']})")
        leakage = True
    else:
        gap = stats['validate']['min'] - stats['train']['max']
        print(f"✅ Train -> Validate 边界清晰 (间隔 {gap:.2f} 秒)")

    # Check Validate vs Test
    if stats['validate']['max'] >= stats['test']['min']:
        print(f"❌ [严重失败] Validate 与 Test 时间重叠！")
        print(f"   Val End ({stats['validate']['max']}) >= Test Start ({stats['test']['min']})")
        leakage = True
    else:
        gap = stats['test']['min'] - stats['validate']['max']
        print(f"✅ Validate -> Test  边界清晰 (间隔 {gap:.2f} 秒)")

    if not leakage:
        print("\n🎉 [Step 3] 完美！没有发现时间泄露，数据集划分符合概念漂移检测要求。")
    else:
        print("\n🚫 [Step 3] 校验失败，请不要使用此数据进行实验。")


if __name__ == "__main__":
    verify_split_leakage()