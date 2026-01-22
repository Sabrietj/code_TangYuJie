"""
测试修正后的公式
"""
import numpy as np
from scipy.special import betaln, gammaln
import math

def log_binomial(n, k):
    """log( n! / (k! × (n-k)!) )"""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def calculate_B_fixed(n_left_W, n_right_W, n_left_R, n_right_R, alpha=1.0, beta=1.0):
    """使用修正公式计算B"""
    n0_total = n_left_W + n_left_R
    n1_total = n_right_W + n_right_R

    # 对数二项式系数
    log_C_total = log_binomial(n0_total + n1_total, n0_total)
    log_C_W = log_binomial(n_left_W + n_right_W, n_left_W)
    log_C_R = log_binomial(n_left_R + n_right_R, n_left_R)

    # 对数Beta值
    log_beta_total = betaln(alpha + n0_total, beta + n1_total)
    log_beta_W = betaln(alpha + n_left_W, beta + n_right_W)
    log_beta_R = betaln(alpha + n_left_R, beta + n_right_R)
    log_beta_prior = betaln(alpha, beta)

    # H0和H1的对数似然
    log_p_H0 = log_C_total + log_beta_total - log_beta_prior
    log_p_H1 = (log_C_W + log_beta_W - log_beta_prior) + \
               (log_C_R + log_beta_R - log_beta_prior)

    log_B = log_p_H0 - log_p_H1
    B = math.exp(log_B)

    return B, log_B

def fixed_formula():
    print("🧪 测试修正后的公式")
    print("="*60)

    test_cases = [
        (50, 50, 50, 50, "完全相同的计数"),
        (60, 40, 60, 40, "相同比例"),
        (80, 20, 20, 80, "相反分布"),
        (95, 5, 5, 95, "极端相反"),
        (100, 0, 0, 100, "完全相反"),
        # 添加小计数测试
        (5, 5, 5, 5, "小计数相同"),
        (8, 2, 2, 8, "小计数相反"),
    ]

    for n_left_W, n_right_W, n_left_R, n_right_R, desc in test_cases:
        B, log_B = calculate_B_fixed(n_left_W, n_right_W, n_left_R, n_right_R)

        print(f"\n{desc}:")
        print(f"  W: [{n_left_W}, {n_right_W}], R: [{n_left_R}, {n_right_R}]")
        print(f"  log_B = {log_B:.6f}")
        print(f"  B = {B:.6f}")

        # 分析
        if abs(B - 1.0) < 0.1:
            print(f"  ✅ B接近1.0（正确：无漂移）")
        elif B < 1.0:
            print(f"  ✅ B < 1.0（正确：检测到漂移）")
        else:
            print(f"  ❌ B > 1.0（可能仍有问题）")

    # 测试对称性
    print("\n\n🔍 测试对称性")
    print("="*60)

    # 交换W和R应该得到相同的B
    B1, log_B1 = calculate_B_fixed(60, 40, 40, 60)
    B2, log_B2 = calculate_B_fixed(40, 60, 60, 40)

    print(f"W=[60,40], R=[40,60]: B = {B1:.6f}, log_B = {log_B1:.6f}")
    print(f"W=[40,60], R=[60,40]: B = {B2:.6f}, log_B = {log_B2:.6f}")
    print(f"是否相等: {abs(B1 - B2) < 1e-10}")

def analyze_why():
    """深入分析公式"""
    print("\n\n🔬 公式分析")
    print("="*60)

    # 简单情况：n=2的小计数
    n_left_W, n_right_W = 1, 1
    n_left_R, n_right_R = 1, 1

    B, log_B = calculate_B_fixed(n_left_W, n_right_W, n_left_R, n_right_R)

    print(f"小计数测试: W=[1,1], R=[1,1]")
    print(f"  B = {B:.6f}")
    print(f"  log_B = {log_B:.6f}")

    # 手动计算
    log_C_total = log_binomial(4, 2)  # log(4!/(2!×2!)) = log(6)
    log_C_W = log_binomial(2, 1)      # log(2!/(1!×1!)) = log(2)
    log_C_R = log_binomial(2, 1)      # log(2!/(1!×1!)) = log(2)

    print(f"\n手动计算:")
    print(f"  log_C_total = {log_C_total:.6f} (log(6) ≈ 1.791759)")
    print(f"  log_C_W = {log_C_W:.6f} (log(2) ≈ 0.693147)")
    print(f"  log_C_R = {log_C_R:.6f} (log(2) ≈ 0.693147)")
    print(f"  log_C_W + log_C_R = {log_C_W + log_C_R:.6f} (log(4) ≈ 1.386294)")
    print(f"  差值: log_C_total - (log_C_W + log_C_R) = {log_C_total - (log_C_W + log_C_R):.6f}")

if __name__ == "__main__":
    fixed_formula()
    analyze_why()