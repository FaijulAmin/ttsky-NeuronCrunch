# SPDX-FileCopyrightText: 2024 Your Name
# SPDX-License-Identifier: Apache-2.0
#
# Comprehensive cocotb testbench for tt_um_matmul
# Covers all 4 modes with edge cases, corner cases, and random regression.

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, ClockCycles
import random

# ─── Golden model (mirrors RTL exactly) ──────────────────────

def to_int8(v):
    v = int(v) & 0xFF
    return v - 256 if v >= 128 else v

def wrap20(v):
    v = int(v) & 0xFFFFF
    return v - 0x100000 if v >= 0x80000 else v

def golden_2x2_relu(A, B):
    C = [[0,0],[0,0]]
    for i in range(2):
        for j in range(2):
            acc = sum(to_int8(A[i][k]) * to_int8(B[k][j]) for k in range(2))
            val = wrap20(acc)
            C[i][j] = 0 if val < 0 else val
    return C

def golden_4x4(A, B, relu=False, C_prev=None, accum=False):
    C = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            acc = sum(to_int8(A[i][k]) * to_int8(B[k][j]) for k in range(4))
            val = wrap20(acc)
            if accum and C_prev is not None:
                val = wrap20(C_prev[i][j] + val)
            if relu and val < 0:
                val = 0
            C[i][j] = val
    return C

# ─── Protocol helpers ─────────────────────────────────────────

async def do_reset(dut):
    dut.rst_n.value  = 0
    dut.ena.value    = 1
    dut.ui_in.value  = 0
    dut.uio_in.value = 0
    await ClockCycles(dut.clk, 4)
    dut.rst_n.value = 1
    await ClockCycles(dut.clk, 2)

def pack_2x2(A, B):
    """Row-major bytes: A[0][0] A[0][1] A[1][0] A[1][1] then B same order"""
    flat = [A[0][0], A[0][1], A[1][0], A[1][1],
            B[0][0], B[0][1], B[1][0], B[1][1]]
    return [int(v) & 0xFF for v in flat]

def pack_4x4(A, B):
    """Row-major 32 bytes: A then B"""
    flat = [A[r][c] for r in range(4) for c in range(4)] + \
           [B[r][c] for r in range(4) for c in range(4)]
    return [int(v) & 0xFF for v in flat]

def decode_output(raw, n_elements):
    """Decode n_elements*3 bytes into signed 20-bit integers"""
    results = []
    for i in range(n_elements):
        b0 = raw[i*3+0]
        b1 = raw[i*3+1]
        b2 = raw[i*3+2] & 0x0F
        val = b0 | (b1 << 8) | (b2 << 16)
        v = val & 0xFFFFF
        results.append(v - 0x100000 if v >= 0x80000 else v)
    return results

async def run_transaction(dut, mode, input_bytes, n_out_elements,
                          accum_clear=False):
    """
    Full protocol transaction:
      1. Pulse start with mode bits set
      2. Load input_bytes serially
      3. Wait for compute (1 or 8 cycles depending on mode)
      4. Read n_out_elements*3 output bytes
    Returns list of signed 20-bit integers (row-major).
    """
    # Assert start with mode and accum_clear
    await RisingEdge(dut.clk)
    ctrl = 0x01 | (mode << 1) | (1 << 3 if accum_clear else 0)
    dut.uio_in.value = ctrl
    dut.ui_in.value  = 0

    # LOAD: deassert start, send bytes
    await RisingEdge(dut.clk)
    dut.uio_in.value = (mode << 1)  # start=0, mode still visible but ignored
    dut.ui_in.value  = input_bytes[0]

    for i in range(1, len(input_bytes)):
        await RisingEdge(dut.clk)
        dut.ui_in.value = input_bytes[i]

    # Wait for COMPUTE cycles (FSM moves to compute then output)
    # After last load byte, FSM needs 1 cycle (2x2) or 8 cycles (4x4) compute
    n_compute = 1 if (mode == 0) else 8
    for _ in range(n_compute):
        await RisingEdge(dut.clk)
    dut.ui_in.value = 0

    # Now in OUTPUT state; read bytes, one per clock
    raw = []
    await RisingEdge(dut.clk)  # first output byte available
    for _ in range(n_out_elements * 3):
        raw.append(int(dut.uo_out.value))
        await RisingEdge(dut.clk)

    return decode_output(raw, n_out_elements)

def flatten_2x2(C):
    return [C[r][c] for r in range(2) for c in range(2)]

def flatten_4x4(C):
    return [C[r][c] for r in range(4) for c in range(4)]

# ─── Test helpers ─────────────────────────────────────────────

async def start_clock(dut):
    clock = Clock(dut.clk, 30, units="ns")  # 33 MHz
    cocotb.start_soon(clock.start())

def assert_eq(got, exp, name):
    if got != exp:
        raise AssertionError(
            f"{name} FAIL:\n  got={got}\n  exp={exp}"
        )

# ═════════════════════════════════════════════════════════════
#  MODE 00: 2x2 + ReLU
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_00_all_zeros(dut):
    """Mode 00: all-zero inputs → all-zero outputs"""
    await start_clock(dut); await do_reset(dut)
    A = [[0,0],[0,0]]; B = [[0,0],[0,0]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "mode00 all_zeros")

@cocotb.test()
async def test_00_identity(dut):
    """Mode 00: A=127*I, B=I → C=127*I (ReLU no-op on positive)"""
    await start_clock(dut); await do_reset(dut)
    A = [[127,0],[0,127]]; B = [[1,0],[0,1]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "mode00 identity")

@cocotb.test()
async def test_00_relu_clamps_negatives(dut):
    """Mode 00: negative results clamped to 0 by ReLU"""
    await start_clock(dut); await do_reset(dut)
    # A=[[0,-1],[0,-1]] B=[[1,0],[1,0]] → C=[[−1,0],[−1,0]] before ReLU → [[0,0],[0,0]]
    A = [[0,-1],[0,-1]]; B = [[1,0],[1,0]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "mode00 relu_clamp")

@cocotb.test()
async def test_00_min_times_min(dut):
    """Mode 00: (-128)*(-128) corner → large positive, ReLU no-op"""
    await start_clock(dut); await do_reset(dut)
    A = [[-128,0],[0,-128]]; B = [[-128,0],[0,-128]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "mode00 min_times_min")

@cocotb.test()
async def test_00_one_hot_each_element(dut):
    """Mode 00: one-hot in A and B, all 8 positions"""
    await start_clock(dut); await do_reset(dut)
    B = [[3,7],[11,5]]
    positions = [(0,0),(0,1),(1,0),(1,1)]
    for r,c in positions:
        A = [[0,0],[0,0]]; A[r][c] = 64
        got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
        exp = flatten_2x2(golden_2x2_relu(A,B))
        assert_eq(got, exp, f"mode00 one_hot_A[{r}][{c}]")
        await ClockCycles(dut.clk, 2)
    A = [[3,7],[11,5]]
    for r,c in positions:
        B = [[0,0],[0,0]]; B[r][c] = 64
        got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
        exp = flatten_2x2(golden_2x2_relu(A,B))
        assert_eq(got, exp, f"mode00 one_hot_B[{r}][{c}]")
        await ClockCycles(dut.clk, 2)

@cocotb.test()
async def test_00_mixed_signs(dut):
    """Mode 00: mixed positive/negative values"""
    await start_clock(dut); await do_reset(dut)
    A = [[10,-20],[30,-40]]; B = [[-50,60],[70,-80]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "mode00 mixed_signs")

@cocotb.test()
async def test_00_random_regression(dut):
    """Mode 00: 100 uniform random tests"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(1001)
    for i in range(100):
        A = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
        exp = flatten_2x2(golden_2x2_relu(A,B))
        assert_eq(got, exp, f"mode00 random[{i}]")
        await ClockCycles(dut.clk, 1)

# ═════════════════════════════════════════════════════════════
#  MODE 01: 4x4 raw
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_01_all_zeros(dut):
    """Mode 01: all zeros"""
    await start_clock(dut); await do_reset(dut)
    A = [[0]*4 for _ in range(4)]; B = [[0]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A,B)), "mode01 all_zeros")

@cocotb.test()
async def test_01_identity(dut):
    """Mode 01: A = 4x4 identity, B = general → C = B"""
    await start_clock(dut); await do_reset(dut)
    A = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A,B)), "mode01 identity")

@cocotb.test()
async def test_01_max_positive(dut):
    """Mode 01: all +127 values"""
    await start_clock(dut); await do_reset(dut)
    A = [[127]*4 for _ in range(4)]; B = [[127]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A,B)), "mode01 max_positive")

@cocotb.test()
async def test_01_min_times_min(dut):
    """Mode 01: all -128 values"""
    await start_clock(dut); await do_reset(dut)
    A = [[-128]*4 for _ in range(4)]; B = [[-128]*4 for _ in range(4)]
    got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
    assert_eq(got, flatten_4x4(golden_4x4(A,B)), "mode01 min_times_min")

@cocotb.test()
async def test_01_one_hot_each_element(dut):
    """Mode 01: one-hot A positions isolating each multiplier path"""
    await start_clock(dut); await do_reset(dut)
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    for r in range(4):
        for c in range(4):
            A = [[0]*4 for _ in range(4)]; A[r][c] = 1
            got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
            exp = flatten_4x4(golden_4x4(A,B))
            assert_eq(got, exp, f"mode01 one_hot_A[{r}][{c}]")
            await ClockCycles(dut.clk, 2)

@cocotb.test()
async def test_01_mixed_extremes(dut):
    """Mode 01: 127 and -128 mixed"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(2002)
    for i in range(20):
        choices = [-128, -1, 0, 1, 127]
        A = [[rng.choice(choices) for _ in range(4)] for _ in range(4)]
        B = [[rng.choice(choices) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
        exp = flatten_4x4(golden_4x4(A,B))
        assert_eq(got, exp, f"mode01 extremes[{i}]")
        await ClockCycles(dut.clk, 2)

@cocotb.test()
async def test_01_random_regression(dut):
    """Mode 01: 50 uniform random tests"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(3003)
    for i in range(50):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
        exp = flatten_4x4(golden_4x4(A,B))
        assert_eq(got, exp, f"mode01 random[{i}]")
        await ClockCycles(dut.clk, 1)

# ═════════════════════════════════════════════════════════════
#  MODE 10: 4x4 + ReLU
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_10_relu_clamps_negatives(dut):
    """Mode 10: verify ReLU clamps all-negative result to 0"""
    await start_clock(dut); await do_reset(dut)
    # A=−I, B=I → C=−I before ReLU → all zeros
    A = [[-1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[1  if r==c else 0 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 2, pack_4x4(A,B), 16)
    exp = flatten_4x4(golden_4x4(A,B, relu=True))
    assert_eq(got, exp, "mode10 relu_clamps_neg")

@cocotb.test()
async def test_10_positive_unchanged(dut):
    """Mode 10: positive results pass through ReLU unchanged"""
    await start_clock(dut); await do_reset(dut)
    A = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 2, pack_4x4(A,B), 16)
    exp = flatten_4x4(golden_4x4(A,B, relu=True))
    assert_eq(got, exp, "mode10 pos_unchanged")

@cocotb.test()
async def test_10_random_regression(dut):
    """Mode 10: 50 random tests"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(4004)
    for i in range(50):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 2, pack_4x4(A,B), 16)
        exp = flatten_4x4(golden_4x4(A,B, relu=True))
        assert_eq(got, exp, f"mode10 random[{i}]")
        await ClockCycles(dut.clk, 1)

# ═════════════════════════════════════════════════════════════
#  MODE 11: 4x4 tiled accumulate
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_11_clear_then_accumulate(dut):
    """Mode 11: clear + tile1, then tile2 accumulates correctly"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(5005)
    A1 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    B1 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    A2 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]
    B2 = [[rng.randint(-64,64) for _ in range(4)] for _ in range(4)]

    # Tile 1: accum_clear=True → C = A1@B1
    got1 = await run_transaction(dut, 3, pack_4x4(A1,B1), 16, accum_clear=True)
    exp1 = flatten_4x4(golden_4x4(A1,B1))
    assert_eq(got1, exp1, "mode11 tile1")
    await ClockCycles(dut.clk, 2)

    # Tile 2: accum_clear=False → C = C + A2@B2
    got2 = await run_transaction(dut, 3, pack_4x4(A2,B2), 16, accum_clear=False)
    C1 = [[got1[r*4+c] for c in range(4)] for r in range(4)]
    exp2 = flatten_4x4(golden_4x4(A2,B2, C_prev=C1, accum=True))
    assert_eq(got2, exp2, "mode11 tile2_accumulated")

@cocotb.test()
async def test_11_three_tile_chain(dut):
    """Mode 11: three sequential tiles accumulate correctly"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(6006)
    tiles = [
        ([[rng.randint(-50,50) for _ in range(4)] for _ in range(4)],
         [[rng.randint(-50,50) for _ in range(4)] for _ in range(4)])
        for _ in range(3)
    ]
    C_running = None
    for idx, (A, B) in enumerate(tiles):
        clr = (idx == 0)
        got = await run_transaction(dut, 3, pack_4x4(A,B), 16, accum_clear=clr)
        C_running = golden_4x4(A, B, C_prev=C_running, accum=(idx>0))
        assert_eq(got, flatten_4x4(C_running), f"mode11 tile{idx}")
        await ClockCycles(dut.clk, 2)

@cocotb.test()
async def test_11_clear_resets_accumulator(dut):
    """Mode 11: accum_clear=True discards previous accumulated state"""
    await start_clock(dut); await do_reset(dut)
    # Load junk tile without clear
    A = [[127]*4 for _ in range(4)]; B = [[127]*4 for _ in range(4)]
    await run_transaction(dut, 3, pack_4x4(A,B), 16, accum_clear=True)
    await ClockCycles(dut.clk, 2)
    # Now clear + fresh simple tile: result should equal just A2@B2
    A2 = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B2 = [[r+1 if c==0 else 0 for c in range(4)] for r in range(4)]
    got = await run_transaction(dut, 3, pack_4x4(A2,B2), 16, accum_clear=True)
    exp = flatten_4x4(golden_4x4(A2, B2))
    assert_eq(got, exp, "mode11 clear_resets")

# ═════════════════════════════════════════════════════════════
#  PROTOCOL / FSM EDGE CASES
# ═════════════════════════════════════════════════════════════

@cocotb.test()
async def test_fsm_back_to_back_mode00(dut):
    """FSM: 3 consecutive 2x2 transactions, no idle gap"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(7007)
    for i in range(3):
        A = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        B = [[rng.randint(-128,127) for _ in range(2)] for _ in range(2)]
        got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
        exp = flatten_2x2(golden_2x2_relu(A,B))
        assert_eq(got, exp, f"back2back_00 tx{i}")

@cocotb.test()
async def test_fsm_back_to_back_mode01(dut):
    """FSM: 2 consecutive 4x4 raw transactions"""
    await start_clock(dut); await do_reset(dut)
    rng = random.Random(8008)
    for i in range(2):
        A = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        B = [[rng.randint(-128,127) for _ in range(4)] for _ in range(4)]
        got = await run_transaction(dut, 1, pack_4x4(A,B), 16)
        exp = flatten_4x4(golden_4x4(A,B))
        assert_eq(got, exp, f"back2back_01 tx{i}")

@cocotb.test()
async def test_fsm_mode_switch(dut):
    """FSM: switch between mode 00 and mode 01 back-to-back"""
    await start_clock(dut); await do_reset(dut)
    # 2x2 first
    A2=[[10,-5],[3,7]]; B2=[[2,0],[1,-1]]
    got2 = await run_transaction(dut, 0, pack_2x2(A2,B2), 4)
    assert_eq(got2, flatten_2x2(golden_2x2_relu(A2,B2)), "mode_switch 2x2")
    # 4x4 immediately after
    A4 = [[1 if r==c else 0 for c in range(4)] for r in range(4)]
    B4 = [[r*4+c+1 for c in range(4)] for r in range(4)]
    got4 = await run_transaction(dut, 1, pack_4x4(A4,B4), 16)
    assert_eq(got4, flatten_4x4(golden_4x4(A4,B4)), "mode_switch 4x4")

@cocotb.test()
async def test_fsm_reset_recovery(dut):
    """FSM: reset mid-operation, verify clean recovery"""
    await start_clock(dut); await do_reset(dut)
    # Begin a transaction then assert reset halfway through load
    dut.uio_in.value = 0x01  # start
    await RisingEdge(dut.clk)
    dut.uio_in.value = 0x00
    dut.ui_in.value  = 0x55   # start loading junk
    for _ in range(4):
        await RisingEdge(dut.clk)
    # Mid-load reset
    dut.rst_n.value = 0
    await ClockCycles(dut.clk, 3)
    dut.rst_n.value = 1
    dut.uio_in.value = 0
    dut.ui_in.value  = 0
    await ClockCycles(dut.clk, 2)
    # Clean transaction should work
    A = [[3,7],[2,-5]]; B = [[4,-1],[6,2]]
    got = await run_transaction(dut, 0, pack_2x2(A,B), 4)
    assert_eq(got, flatten_2x2(golden_2x2_relu(A,B)), "reset_recovery")
