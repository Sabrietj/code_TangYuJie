import pandas as pd
import ast
import os
import sys
from tqdm import tqdm

# 配置路径 (请根据实际情况调整)
BASE_DIR = "/root/autodl-fs/processed_data/CIC-IDS-2017-session-srcIP_srcPort_dstIP_dstPort_proto"
FLOW_CSV = os.path.join(BASE_DIR, "all_flow.csv")
SESSION_CSV = os.path.join(BASE_DIR, "all_split_session.csv")


def verify_order():
    print(f"正在读取 Flow 数据: {FLOW_CSV} ...")
    # 只读取 uid 和 时间戳，节省内存
    # 注意：conn.ts 是 Zeek 的标准时间戳字段
    df_flow = pd.read_csv(FLOW_CSV, usecols=['uid', 'conn.ts'])

    # 构建快速查询字典: uid -> timestamp
    print("构建 UID -> 时间戳 映射...")
    uid_to_ts = dict(zip(df_flow['uid'], df_flow['conn.ts']))

    print(f"正在读取 Session 数据: {SESSION_CSV} ...")
    df_session = pd.read_csv(SESSION_CSV, usecols=['session_index', 'flow_uid_list', 'split'])

    print("计算每个 Session 的开始时间...")

    # 辅助函数：获取会话中最早的流时间作为会话时间
    def get_session_start(uid_str):
        try:
            uids = ast.literal_eval(uid_str)
            if not uids: return float('inf')
            # 查找该会话所有流的时间，取最小值
            times = [uid_to_ts.get(uid, float('inf')) for uid in uids]
            return min(times)
        except:
            return float('inf')

    tqdm.pandas(desc="Processing Sessions")
    df_session['start_time'] = df_session['flow_uid_list'].progress_apply(get_session_start)

    # 过滤无效数据
    valid_sessions = df_session[df_session['start_time'] != float('inf')]

    print("\n" + "=" * 50)
    print("数据集时序分布统计")
    print("=" * 50)

    stats = {}
    for split_name in ['train', 'validate', 'test']:
        subset = valid_sessions[valid_sessions['split'] == split_name]
        if subset.empty:
            print(f"⚠️ {split_name} 集为空！")
            continue

        t_min = subset['start_time'].min()
        t_max = subset['start_time'].max()
        count = len(subset)
        stats[split_name] = {'min': t_min, 'max': t_max, 'count': count}

        print(f"[{split_name.upper()}]")
        print(f"  数量: {count}")
        print(f"  时间范围: {t_min:.2f} -> {t_max:.2f}")
        print("-" * 30)

    # 核心验证逻辑
    print("\n正在验证时序严格性 (Train < Validate < Test)...")

    is_ordered = True

    # 验证 Train vs Validate
    if stats['train']['max'] > stats['validate']['min']:
        print(f"❌ 失败: Train 最大时间 ({stats['train']['max']}) > Validate 最小时间 ({stats['validate']['min']})")
        print(f"   存在时间重叠或乱序！")
        is_ordered = False
    else:
        print(f"✅ Train 与 Validate 时序正常 (间隙: {stats['validate']['min'] - stats['train']['max']:.2f}秒)")

    # 验证 Validate vs Test
    if stats['validate']['max'] > stats['test']['min']:
        print(f"❌ 失败: Validate 最大时间 ({stats['validate']['max']}) > Test 最小时间 ({stats['test']['min']})")
        is_ordered = False
    else:
        print(f"✅ Validate 与 Test 时序正常 (间隙: {stats['test']['min'] - stats['validate']['max']:.2f}秒)")

    if is_ordered:
        print("\n🎉 结论: Session 划分是严格按时间有序的！")
    else:
        print("\n⚠️ 结论: Session 划分存在时序问题，请检查数据生成逻辑。")


if __name__ == "__main__":
    verify_order()