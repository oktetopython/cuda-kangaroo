#!/usr/bin/env python3
"""
验证Bernstein算法的数学计算
根据论文《Computing small discrete logarithms faster》
"""

def mod_sub(a, b, n):
    """正确的模减法: (a - b) mod n, 结果在 [0, n-1]"""
    a = a % n
    b = b % n
    result = a - b
    if result < 0:
        result += n
    return result

def verify_bernstein_calculation():
    # 从最新测试日志中提取的实际数值 (修正公式)
    range_end = 0x0000000000ffffff       # 区间右端点
    # 从C++日志推算：range_end - table_steps = 0x71cb95d9012846f5
    # 所以：table_steps = range_end - 0x71cb95d9012846f5 = 0x8e346a26fed7b90a
    table_steps = 0x8e346a26fed7b90a     # 表中步数 (从右端点到DP的步数)
    sum_steps = 0x48a2349c01b89938       # 查找步数 (从目标点到DP的步数)
    
    # secp256k1曲线的阶
    n = 0xfffffffffffffffffffffffffffffffebaaedce6af48a03bbfd25e8cd0364141
    
    # 期望的私钥 (24号谜题)
    expected = 0xdc2a04
    
    print("=== Bernstein算法数学验证 (修正公式) ===")
    print(f"区间右端点 (range_end): 0x{range_end:016x}")
    print(f"表中步数 (table_steps): 0x{table_steps:016x}")
    print(f"查找步数 (sum_steps): 0x{sum_steps:016x}")
    print(f"期望私钥: 0x{expected:x}")
    print(f"secp256k1阶 (n): 0x{n:x}")
    print()

    # 修正公式: k_target = (range_end - table_steps) - sum_steps (mod n)
    print("=== 计算步骤 (正确的模减法) ===")

    # 步骤1: range_end - table_steps (mod n)
    temp1 = mod_sub(range_end, table_steps, n)
    print(f"步骤1: mod_sub(range_end, table_steps, n) = 0x{temp1:016x}")

    # 步骤2: temp1 - sum_steps (mod n)
    k_candidate = mod_sub(temp1, sum_steps, n)
    print(f"步骤2: mod_sub(temp1, sum_steps, n) = 0x{k_candidate:016x}")

    # 检查结果是否在区间内
    interval_start = 0x800000
    interval_end = 0xffffff
    if interval_start <= k_candidate <= interval_end:
        print(f"✅ 结果在目标区间内")
    else:
        print(f"❌ 结果不在目标区间 [0x{interval_start:x}, 0x{interval_end:x}] 内")
        # 如果结果不在区间内，可能需要进一步分析
    
    print()
    print("=== 结果比较 ===")
    print(f"计算结果: 0x{k_candidate:016x}")
    print(f"期望结果: 0x{expected:016x}")
    print(f"匹配: {'✅ 是' if k_candidate == expected else '❌ 否'}")
    
    if k_candidate != expected:
        print()
        print("=== 差异分析 ===")
        diff = abs(k_candidate - expected)
        print(f"差值: 0x{diff:016x} ({diff})")
        print(f"计算结果是期望结果的倍数: {k_candidate / expected:.2f}")
        
        # 检查是否在正确的区间内
        interval_start = 0x800000
        interval_end = 0xffffff
        print(f"计算结果是否在区间[0x{interval_start:x}, 0x{interval_end:x}]内: {'是' if interval_start <= k_candidate <= interval_end else '否'}")
        print(f"期望结果是否在区间内: {'是' if interval_start <= expected <= interval_end else '是'}")
    
    return k_candidate == expected

if __name__ == "__main__":
    verify_bernstein_calculation()
