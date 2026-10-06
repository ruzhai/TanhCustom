#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TanhCustom 离线验证器 —— 不需要 NPU，也不需要 CANN，Windows 上直接跑。

它做两件事：
  1. 复刻 host 侧的 tiling 分块（BLOCK_DIM / TILE_NUM / BUFFER_NUM）与 kernel 侧
     的算子实现，逐块模拟 (e^x - e^-x)/(e^x + e^-x) 的 fp32 计算 -> fp16 回写；
  2. 用与 AclNNInvocation/scripts/verify_result.py 完全一致的容差判据判定是否通过。

注意：这不是 NPU 的位精确仿真，Exp/Div 等指令的末位与 numpy 不会逐位相同。
它的价值是提前验证【分块逻辑正确】和【精度余量足够】，这两件事在真机上不会变。

用法：python simulate_tanh.py
"""

import numpy as np

# ---- 与 op_host/tanh_custom.cpp 保持一致的常量 ----
BLOCK_DIM = 8
TILE_NUM = 8
# ---- 与 op_kernel/tanh_custom.cpp 保持一致 ----
BUFFER_NUM = 2

# ---- 与 AclNNInvocation/scripts/verify_result.py 保持一致的判据 ----
LOSS = 1e-3
MINIMUM = 1e-3

SHAPE = (8, 2048)   # 与 gen_data.py 里的 np.random.uniform(-3, 3, [8, 2048]) 一致


def kernel_tanh(x_fp16):
    """复刻 KernelTanh.Process() 的处理顺序：8 个核，每核 8*2 块，每块 tile_length 个元素。"""
    total_length = x_fp16.size
    block_length = total_length // BLOCK_DIM
    tile_length = block_length // TILE_NUM // BUFFER_NUM

    if tile_length * TILE_NUM * BUFFER_NUM != block_length:
        raise AssertionError(
            "分块不能整除：totalLength=%d, BLOCK_DIM*TILE_NUM*BUFFER_NUM=%d"
            % (total_length, BLOCK_DIM * TILE_NUM * BUFFER_NUM)
        )

    out = np.empty(total_length, dtype=np.float16)
    for blk in range(BLOCK_DIM):
        base = blk * block_length
        for i in range(TILE_NUM * BUFFER_NUM):
            s = base + i * tile_length
            tile = x_fp16[s:s + tile_length].astype(np.float32)   # Cast fp16 -> fp32
            e_pos = np.exp(tile)                                  # Exp
            e_neg = np.exp(-tile)                                 # Muls(-1) + Exp
            num = e_pos - e_neg                                   # Sub
            den = e_pos + e_neg                                   # Add
            out[s:s + tile_length] = (num / den).astype(np.float16)  # Div + Cast fp32 -> fp16
    return out


def golden_tanh(x_fp16):
    """复刻 gen_data.py 的真值算法（float64 计算后存为 fp16）。"""
    x = x_fp16.astype(np.float64)
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))).astype(np.float16)


def verify(real_result, golden):
    """与 verify_result.py 逐行等价的判据，额外返回失败计数便于诊断。"""
    result = np.abs(real_result - golden)
    deno = np.maximum(np.abs(real_result), np.abs(golden))
    result_atol = result <= LOSS
    result_rtol = result / (deno + MINIMUM) <= LOSS

    n = real_result.size
    n_atol_fail = int(np.sum(~result_atol))
    n_rtol_fail = int(np.sum(~result_rtol))

    if not result_rtol.all() and not result_atol.all():
        if n_rtol_fail > n * LOSS and n_atol_fail > n * LOSS:
            return False, n_atol_fail, n_rtol_fail
    return True, n_atol_fail, n_rtol_fail


def main():
    total = SHAPE[0] * SHAPE[1]
    tile_length = total // BLOCK_DIM // TILE_NUM // BUFFER_NUM
    print("shape=%s  totalLength=%d" % (list(SHAPE), total))
    print("blockLength=%d  tileLength=%d  每核搬 %d 块"
          % (total // BLOCK_DIM, tile_length, TILE_NUM * BUFFER_NUM))
    print("单块字节数：fp16=%d B, fp32 临时量=%d B"
          % (tile_length * 2, tile_length * 4))
    print("-" * 68)

    # 1) 与 gen_data.py 相同分布的随机输入
    n_cases = 50
    worst_atol = 0
    worst_rtol = 0
    for seed in range(n_cases):
        rng = np.random.default_rng(seed)
        x = rng.uniform(-3, 3, list(SHAPE)).astype(np.float16)
        got = kernel_tanh(x.reshape(-1))
        exp = golden_tanh(x.reshape(-1))
        ok, fa, fr = verify(got, exp)
        worst_atol = max(worst_atol, fa)
        worst_rtol = max(worst_rtol, fr)
        if not ok:
            print("[FAIL] seed=%d atol失败=%d rtol失败=%d" % (seed, fa, fr))
            return 1
    print("[OK] %d 组 uniform(-3,3) 随机输入全部通过（最差 atol 失败 %d 个, rtol 失败 %d 个）"
          % (n_cases, worst_atol, worst_rtol))

    # 2) 边界值：0、极小、接近 ±3 的极值
    edge = np.array([0.0, -0.0, 1e-4, -1e-4, 1e-3, -1e-3,
                     2.999, -2.999, 3.0, -3.0, 0.5, -0.5], dtype=np.float16)
    x = np.zeros(total, dtype=np.float16)
    x[:edge.size] = edge
    got = kernel_tanh(x)
    exp = golden_tanh(x)
    ok, fa, fr = verify(got, exp)
    print("[%s] 边界值用例 atol失败=%d rtol失败=%d" % ("OK" if ok else "FAIL", fa, fr))
    if not ok:
        return 1

    # 3) 抽样打印，肉眼确认数量级
    print("-" * 68)
    x = np.random.default_rng(0).uniform(-3, 3, list(SHAPE)).astype(np.float16).reshape(-1)
    got = kernel_tanh(x)
    exp = golden_tanh(x)
    print("输入        期望(tanh)      实算        绝对误差")
    for i in [0, 1, 2, 100, 5000, 16383]:
        print("%10.5f  %12.6f  %12.6f  %10.2e"
              % (x[i], exp[i], got[i], abs(float(got[i]) - float(exp[i]))))

    nz = np.abs(exp) > 1e-3
    print("-" * 68)
    print("最大绝对误差 %.3e (仅统计|期望|>1e-3 的 %d 个点)"
          % (np.max(np.abs(got[nz].astype(np.float32) - exp[nz].astype(np.float32))), int(nz.sum())))
    print("结论：算法与分块逻辑正确，精度有充足余量。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
