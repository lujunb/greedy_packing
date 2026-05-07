"""
两板方案的后处理：更贴近需求 ④⑤
④ 尽量把零件挪到前板，降低最后一张利用率（随机尝试，不保证仍能挪动）
⑤ 对最后一张做模拟退火式扰动+重排，目标为包络面积更小、更靠左下

示例：  .venv\\Scripts\\python refine_two_boards.py --seed 7 --sa-iters 200 --hole-fill 200
输出图： outputs_refined/sheet_1.png、sheet_2.png
"""
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from solver import (
    BOARD_H,
    BOARD_W,
    PART_PRIORITY,
    PlacedPart,
    board_utilization,
    build_rotated_cache,
    draw_sheet,
    expand_parts,
    load_packing_config,
    pack_with_k,
    try_place_one,
)


def copy_placed_list(parts: List[PlacedPart]) -> List[PlacedPart]:
    return [
        PlacedPart(p.part_name, p.polygon, p.rotation, p.sheet_index) for p in parts
    ]


def last_sheet_tuple(parts: List[PlacedPart]) -> Tuple[float, float, float, float]:
    """越小越好：(包络面积, uminx+uminy, umaxy, umaxx)，与⑤一致。"""
    if not parts:
        return (0.0, 0.0, 0.0, 0.0)
    uminx = min(pp.polygon.bounds[0] for pp in parts)
    uminy = min(pp.polygon.bounds[1] for pp in parts)
    umaxx = max(pp.polygon.bounds[2] for pp in parts)
    umaxy = max(pp.polygon.bounds[3] for pp in parts)
    area = (umaxx - uminx) * (umaxy - uminy)
    return (area, uminx + uminy, umaxy, umaxx)


def greedy_repack_last_sheet(
    remaining: List[PlacedPart],
    pool_names: List[str],
    rotated_cache: Dict,
    rng: random.Random,
    scan_step: float,
    grid_samples: int,
    max_candidates: int,
    hole_fill: int,
    edge_strip: int,
) -> Optional[List[PlacedPart]]:
    """在已有 remaining 障碍下，把 pool 中的件只往「单张板」上摆（几何列表 s，sheet_idx=0）。"""
    order = sorted(
        pool_names, key=lambda n: (PART_PRIORITY.get(n, -1), rng.random()), reverse=True
    )
    s = copy_placed_list(remaining)
    for pn in order:
        placed = try_place_one(
            pn,
            0,
            s,
            rotated_cache,
            rng,
            scan_step,
            grid_samples=grid_samples,
            max_candidates=max_candidates,
            hole_fill_random=hole_fill,
            edge_strip_samples=edge_strip,
            last_sheet_compact=True,
        )
        if placed is None:
            return None
        s.append(placed)
    return s


def push_max_to_sheet0(
    sheets: List[List[PlacedPart]],
    rotated_cache: Dict,
    rng: random.Random,
    scan_step: float,
    max_passes: int,
    push_hole_fill: int,
    push_edge_strip: int,
    push_max_candidates: int,
    push_grid_samples: int,
) -> int:
    """随机抽样尝试把第2板上的件挪到第1板（max_passes 为尝试次数，非嵌套全扫描）。"""
    moves = 0
    no_progress = 0
    for _ in range(max_passes):
        if not sheets[1]:
            break
        p = rng.choice(sheets[1])
        trial = try_place_one(
            p.part_name,
            0,
            sheets[0],
            rotated_cache,
            rng,
            scan_step,
            grid_samples=push_grid_samples,
            max_candidates=push_max_candidates,
            hole_fill_random=push_hole_fill,
            edge_strip_samples=push_edge_strip,
            last_sheet_compact=False,
        )
        if trial is not None:
            sheets[1].remove(p)
            sheets[0].append(trial)
            moves += 1
            no_progress = 0
        else:
            no_progress += 1
            if no_progress > 200:
                break
    return moves


def sa_refine_last_sheet(
    sheets: List[List[PlacedPart]],
    rotated_cache: Dict,
    rng: random.Random,
    scan_step: float,
    iterations: int,
    t0: float,
    t_end: float,
    sa_hole_fill: int,
    sa_edge_strip: int,
    sa_max_candidates: int,
    sa_grid_samples: int,
) -> int:
    """对最后一张做模拟退火式扰动+重排（④ 不增加件数，仅⑤ 更紧凑）。"""
    if len(sheets) < 2 or len(sheets[1]) <= 2:
        return 0

    s1 = copy_placed_list(sheets[1])
    best = copy_placed_list(s1)
    best_key = last_sheet_tuple(best)
    current = copy_placed_list(s1)
    current_key = best_key

    alpha = (t_end / t0) ** (1.0 / max(iterations, 1))
    T = t0
    accepted = 0

    for it in range(iterations):
        if len(current) < 3:
            break
        step = max(1, iterations // 8)
        if it > 0 and it % step == 0:
            print(f"  SA 进度 {it}/{iterations} ...")
        n_remove = rng.randint(2, min(8, len(current) - 1))
        removed = rng.sample(current, n_remove)
        remaining = [x for x in current if x not in removed]
        pool = [p.part_name for p in removed]

        new_s1: Optional[List[PlacedPart]] = None
        for _ in range(5):
            rng.shuffle(pool)
            cand = greedy_repack_last_sheet(
                remaining,
                pool,
                rotated_cache,
                rng,
                scan_step,
                sa_grid_samples,
                sa_max_candidates,
                sa_hole_fill,
                sa_edge_strip,
            )
            if cand is not None:
                new_s1 = cand
                break
        if new_s1 is None:
            T *= alpha
            continue

        new_key = last_sheet_tuple(new_s1)
        if new_key < current_key:
            current = new_s1
            current_key = new_key
            accepted += 1
            if new_key < best_key:
                best = copy_placed_list(new_s1)
                best_key = new_key
        else:
            delta = new_key[0] - current_key[0]
            if T > 1e-12 and rng.random() < math.exp(-delta / T):
                current = new_s1
                current_key = new_key
                accepted += 1
        T *= alpha

    sheets[1] = best
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser(description="两板后处理：④ 前板尽量多装 ⑤ 最后板紧凑团端")
    parser.add_argument("--seed", type=int, default=20260403)
    parser.add_argument("--attempts", type=int, default=1, help="初始 pack_with_k 尝试次数")
    parser.add_argument("--scan-step", type=float, default=45.0)
    parser.add_argument("--hole-fill", type=int, default=350)
    parser.add_argument("--edge-strip", type=int, default=120)
    parser.add_argument("--grid-samples", type=int, default=700)
    parser.add_argument("--max-candidates", type=int, default=900)
    parser.add_argument("--refine-rounds", type=int, default=8)
    parser.add_argument("--push-passes", type=int, default=120, help="④ 随机尝试挪到前板的次数（越大越慢）")
    parser.add_argument(
        "--push-hole-fill",
        type=int,
        default=0,
        help="仅④ 推前板时的孔洞采样（默认0加速；想更积极可设与 --hole-fill 相同）",
    )
    parser.add_argument("--push-edge-strip", type=int, default=0)
    parser.add_argument("--push-max-candidates", type=int, default=450)
    parser.add_argument(
        "--push-grid-samples",
        type=int,
        default=180,
        help="④ 推前板时网格回退点数（小则快）",
    )
    parser.add_argument("--sa-iters", type=int, default=500, help="⑤ 模拟退火迭代次数（越大越慢）")
    parser.add_argument("--sa-t0", type=float, default=5e4, help="SA 初始温度（相对包络面积尺度）")
    parser.add_argument("--sa-tend", type=float, default=1e-3)
    parser.add_argument(
        "--sa-hole-fill",
        type=int,
        default=0,
        help="仅 SA 重排时的孔洞随机次数（默认0可大幅加速；需要可设120~280）",
    )
    parser.add_argument("--sa-edge-strip", type=int, default=0, help="仅 SA 时的四边条带采样，默认0加速")
    parser.add_argument("--sa-max-candidates", type=int, default=380, help="仅 SA 重排时的锚点上限（越小越快）")
    parser.add_argument("--sa-grid-samples", type=int, default=220, help="仅 SA 时网格回退采样点数")
    parser.add_argument("--out-dir", type=str, default="outputs_refined")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="与 solver 相同的 JSON 配置；未指定则使用内置默认",
    )
    args = parser.parse_args()

    cfg_path = Path(args.config) if args.config.strip() else None
    parts = load_packing_config(cfg_path)
    all_parts = expand_parts(parts)
    cache = build_rotated_cache(parts)
    rng = random.Random(args.seed)

    print("=== 初始两板排样（pack_with_k）===")
    sol = pack_with_k(
        all_parts,
        2,
        cache,
        attempts=args.attempts,
        seed=args.seed,
        scan_step=args.scan_step,
        refine_rounds=args.refine_rounds,
        hole_fill_random=args.hole_fill,
        grid_samples=args.grid_samples,
        max_candidates=args.max_candidates,
        edge_strip_samples=args.edge_strip,
        last_sheet_cluster=True,
    )
    if sol is None:
        print("初始 K=2 可行解失败，请增大 --attempts")
        return

    u0 = board_utilization(sol[0])
    u1 = board_utilization(sol[1])
    n1 = len(sol[1])
    print(f"第1板 利用率={u0 * 100:.2f}%, 件数={len(sol[0])}")
    print(f"第2板 利用率={u1 * 100:.2f}%, 件数={n1}")

    print("\n=== ④ 尽量挪到第1板 ===")
    push_moves = push_max_to_sheet0(
        sol,
        cache,
        rng,
        args.scan_step,
        args.push_passes,
        push_hole_fill=args.push_hole_fill,
        push_edge_strip=args.push_edge_strip,
        push_max_candidates=args.push_max_candidates,
        push_grid_samples=args.push_grid_samples,
    )
    u0 = board_utilization(sol[0])
    u1 = board_utilization(sol[1])
    print(f"成功移动 {push_moves} 件到第1板")
    print(f"第1板 利用率={u0 * 100:.2f}%, 件数={len(sol[0])}")
    print(f"第2板 利用率={u1 * 100:.2f}%, 件数={len(sol[1])}")

    print("\n=== ⑤ 最后一张 SA 紧凑重排 ===")
    if args.sa_iters <= 0:
        print("已跳过（--sa-iters<=0）")
        sa_acc = 0
    else:
        sa_acc = sa_refine_last_sheet(
            sol,
            cache,
            rng,
            args.scan_step,
            args.sa_iters,
            args.sa_t0,
            args.sa_tend,
            sa_hole_fill=args.sa_hole_fill,
            sa_edge_strip=args.sa_edge_strip,
            sa_max_candidates=args.sa_max_candidates,
            sa_grid_samples=args.sa_grid_samples,
        )
    u0 = board_utilization(sol[0])
    u1 = board_utilization(sol[1])
    print(f"SA 接受扰动次数(约): {sa_acc}")
    print(f"第1板 利用率={u0 * 100:.2f}%, 件数={len(sol[0])}")
    print(f"第2板 利用率={u1 * 100:.2f}%, 件数={len(sol[1])}")
    print(f"第2板包络指标(越小越好): {last_sheet_tuple(sol[1])}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)
    for idx, sp in enumerate(sol, start=1):
        util = board_utilization(sp) * 100.0
        png = out_dir / f"sheet_{idx}.png"
        draw_sheet(sp, png, title=f"Refined Sheet {idx} | Parts={len(sp)} | Util={util:.2f}%")
    print(f"\n布置图已写入: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
