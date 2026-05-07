import json
import math
import random
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely import affinity
from shapely.geometry import Point, Polygon


BOARD_W = 2000.0
BOARD_H = 4000.0
GAP = 1.0
EPS = 1e-6
# 与 part_library 中圆一致，用于图上标注（非 buffer 近似多边形的顶点）
CIRCLE_RADIUS_MM = 210.0

# 可对圆阵中间、边带等“非角点”空隙采样的零件（边长/尺寸相对小、可旋转多）
_DEFAULT_HOLE_FILL = frozenset({"eq_tri_160", "trap_120_145_h70", "rect_155x70"})
# 四边条带专项采样（贴左右上下边），仅三角/梯形：矩形太宽易拖慢
_DEFAULT_EDGE_STRIP = frozenset({"eq_tri_160", "trap_120_145_h70"})
HOLE_FILL_PART_NAMES: frozenset = _DEFAULT_HOLE_FILL
EDGE_STRIP_PART_NAMES: frozenset = _DEFAULT_EDGE_STRIP

# 由 configure_from_parts / JSON 加载后写入：贪心/LNS 排序（面积大者优先）
PART_PRIORITY: Dict[str, int] = {}
# compact_front：面积小者优先尝试挪板
PART_RANK: Dict[str, int] = {}
# 圆零件名 -> 几何半径(mm)，用于图上标注
CIRCLE_RADIUS_BY_NAME: Dict[str, float] = {}
# 若非空则覆盖 unique_rotations 的硬编码分支
ROTATIONS: Dict[str, List[float]] = {}


@dataclass(frozen=True)
class PartType:
    name: str
    count: int
    polygon: Polygon
    shape_kind: Optional[str] = None


@dataclass
class PlacedPart:
    part_name: str
    polygon: Polygon
    rotation: float
    sheet_index: int
    _gap_zone: Optional[Polygon] = field(default=None, repr=False)

    def gap_zone(self) -> Polygon:
        """原零件向外膨胀 (GAP-EPS)，用于快速判断与另一件是否过近（替代 distance）。"""
        if self._gap_zone is None:
            self._gap_zone = self.polygon.buffer(GAP - EPS)
        return self._gap_zone


def regular_triangle(side: float) -> Polygon:
    h = side * math.sqrt(3) / 2.0
    return Polygon([(0.0, 0.0), (side, 0.0), (side / 2.0, h)])


def symmetric_trapezoid(top: float, bottom: float, height: float) -> Polygon:
    dx = (bottom - top) / 2.0
    return Polygon([(0.0, 0.0), (bottom, 0.0), (bottom - dx, height), (dx, height)])


def parallelogram(base: float, height: float, shift: float) -> Polygon:
    return Polygon([(0.0, 0.0), (base, 0.0), (base + shift, height), (shift, height)])


def polygon_from_shape(shape: str, params: Dict[str, Any]) -> Tuple[Polygon, Optional[float]]:
    """由 JSON 的 shape/params 构造归一化前多边形；圆返回 (buffer 多边形, 半径)。"""
    s = str(shape).lower().strip()
    if s == "rectangle":
        w, h = float(params["width"]), float(params["height"])
        if w <= 0 or h <= 0:
            raise ValueError("rectangle: width/height 必须为正")
        return Polygon([(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]), None
    if s == "circle":
        r = float(params["radius"])
        if r <= 0:
            raise ValueError("circle: radius 必须为正")
        res = int(params.get("resolution", 64))
        return Point(0.0, 0.0).buffer(r, resolution=res), r
    if s == "equilateral_triangle":
        side = float(params["side"])
        if side <= 0:
            raise ValueError("equilateral_triangle: side 必须为正")
        return regular_triangle(side), None
    if s == "symmetric_trapezoid":
        top = float(params["top"])
        bottom = float(params["bottom"])
        height = float(params["height"])
        if min(top, bottom, height) <= 0:
            raise ValueError("symmetric_trapezoid: top/bottom/height 必须为正")
        return symmetric_trapezoid(top, bottom, height), None
    if s == "parallelogram":
        base = float(params["base"])
        height = float(params["height"])
        shift = float(params["shift"])
        if min(base, height) <= 0:
            raise ValueError("parallelogram: base/height 必须为正")
        return parallelogram(base, height, shift), None
    if s == "polygon":
        verts = params.get("vertices")
        if not verts or len(verts) < 3:
            raise ValueError("polygon: vertices 至少需要 3 个点")
        coords = [(float(x), float(y)) for x, y in verts]
        poly = Polygon(coords)
        if not poly.is_valid:
            poly = poly.buffer(0)
        return poly, None
    raise ValueError(f"未知 shape: {shape!r}，支持: rectangle, circle, equilateral_triangle, "
                     f"symmetric_trapezoid, parallelogram, polygon")


def default_rotations_for_shape(shape: str) -> List[float]:
    s = str(shape).lower().strip()
    if s == "circle":
        return [0.0]
    if s == "equilateral_triangle":
        return [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    if s in ("rectangle", "symmetric_trapezoid", "parallelogram", "polygon"):
        return [0.0, 90.0, 180.0, 270.0]
    return [0.0]


def configure_from_parts(parts: List[PartType]) -> None:
    """根据零件列表设置贪心优先级、compact 顺序；内置方案下补充圆半径标注。"""
    global PART_PRIORITY, PART_RANK, CIRCLE_RADIUS_BY_NAME
    if not parts:
        PART_PRIORITY, PART_RANK = {}, {}
        CIRCLE_RADIUS_BY_NAME = {}
        return
    by_area_desc = sorted(((p.name, p.polygon.area) for p in parts), key=lambda x: -x[1])
    PART_PRIORITY = {name: i + 1 for i, (name, _) in enumerate(by_area_desc)}
    by_area_asc = sorted(parts, key=lambda p: p.polygon.area)
    PART_RANK = {p.name: i for i, p in enumerate(by_area_asc)}
    for p in parts:
        if p.name == "circle_r210" and p.name not in CIRCLE_RADIUS_BY_NAME:
            CIRCLE_RADIUS_BY_NAME[p.name] = CIRCLE_RADIUS_MM


def load_packing_config_json(path: Path) -> List[PartType]:
    """从 JSON 读取母板尺寸、间隙、零件定义，并更新全局 BOARD_*/GAP/ROTATIONS 等。"""
    global BOARD_W, BOARD_H, GAP, ROTATIONS, HOLE_FILL_PART_NAMES, EDGE_STRIP_PART_NAMES
    if not path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {path.resolve()}")
    data = json.loads(path.read_text(encoding="utf-8"))
    board = data.get("board") or {}
    BOARD_W = float(board["width"])
    BOARD_H = float(board["height"])
    if BOARD_W <= 0 or BOARD_H <= 0:
        raise ValueError("board.width / board.height 必须为正")
    GAP = float(data.get("gap", 1.0))
    if GAP < 0:
        raise ValueError("gap 不能为负")

    raw_parts = data.get("parts")
    if not isinstance(raw_parts, list) or not raw_parts:
        raise ValueError('JSON 需包含非空 "parts" 数组')

    parts: List[PartType] = []
    rotations: Dict[str, List[float]] = {}
    shape_by_name: Dict[str, str] = {}
    circle_radii: Dict[str, float] = {}

    for i, entry in enumerate(raw_parts):
        if isinstance(entry, dict) and len(entry) == 1 and "_comment" in entry:
            continue
        if not isinstance(entry, dict):
            raise ValueError(f"parts[{i}] 必须是对象")
        name = str(entry.get("name", "")).strip()
        if not name:
            raise ValueError(f"parts[{i}] 缺少 name")
        count = int(entry.get("count", 0))
        if count < 1:
            raise ValueError(f"零件 {name!r} 的 count 必须 >= 1")
        shape = str(entry.get("shape", "")).strip()
        params = entry.get("params")
        if not isinstance(params, dict):
            raise ValueError(f"零件 {name!r} 需要对象 params")
        poly, cr = polygon_from_shape(shape, params)
        shape_by_name[name] = shape.lower().strip()
        if cr is not None:
            circle_radii[name] = cr
        rots = entry.get("rotations")
        if rots is None:
            rots = default_rotations_for_shape(shape)
        else:
            rots = [float(x) for x in rots]
        if not rots:
            raise ValueError(f"零件 {name!r} 的 rotations 不能为空")
        rotations[name] = rots
        parts.append(PartType(name, count, poly, shape_kind=shape_by_name[name]))

    ROTATIONS = rotations

    if "hole_fill_part_names" in data:
        HOLE_FILL_PART_NAMES = frozenset(str(x) for x in data["hole_fill_part_names"])
    else:
        HOLE_FILL_PART_NAMES = frozenset(
            n
            for n, sh in shape_by_name.items()
            if sh in ("equilateral_triangle", "symmetric_trapezoid", "rectangle")
        )

    if "edge_strip_part_names" in data:
        EDGE_STRIP_PART_NAMES = frozenset(str(x) for x in data["edge_strip_part_names"])
    else:
        EDGE_STRIP_PART_NAMES = frozenset(
            n for n, sh in shape_by_name.items() if sh in ("equilateral_triangle", "symmetric_trapezoid")
        )

    global CIRCLE_RADIUS_BY_NAME
    CIRCLE_RADIUS_BY_NAME = {}
    configure_from_parts(parts)
    CIRCLE_RADIUS_BY_NAME.update(circle_radii)
    return parts


def load_packing_config(path: Optional[Path]) -> List[PartType]:
    """未指定 path 时使用内置 part_library() 与默认母板尺寸。"""
    global BOARD_W, BOARD_H, GAP, ROTATIONS, HOLE_FILL_PART_NAMES, EDGE_STRIP_PART_NAMES
    if path is None:
        BOARD_W, BOARD_H, GAP = 2000.0, 4000.0, 1.0
        ROTATIONS = {}
        HOLE_FILL_PART_NAMES = _DEFAULT_HOLE_FILL
        EDGE_STRIP_PART_NAMES = _DEFAULT_EDGE_STRIP
        parts = part_library()
        CIRCLE_RADIUS_BY_NAME.clear()
        configure_from_parts(parts)
        return parts
    return load_packing_config_json(path)


def normalize_polygon(poly: Polygon) -> Polygon:
    minx, miny, _, _ = poly.bounds
    return affinity.translate(poly, xoff=-minx, yoff=-miny)


def unique_rotations(part_name: str) -> List[float]:
    if part_name in ROTATIONS:
        return ROTATIONS[part_name]
    if part_name == "circle_r210":
        return [0.0]
    if part_name == "rect_155x70":
        return [0.0, 90.0]
    if part_name == "eq_tri_160":
        return [0.0, 60.0, 120.0, 180.0, 240.0, 300.0]
    if part_name == "trap_120_145_h70":
        return [0.0, 90.0, 180.0, 270.0]
    if part_name == "para_b240_h100":
        return [0.0, 90.0, 180.0, 270.0]
    return [0.0]


def part_library() -> List[PartType]:
    rect = Polygon([(0.0, 0.0), (155.0, 0.0), (155.0, 70.0), (0.0, 70.0)])
    tri = regular_triangle(160.0)
    circle = Point(0.0, 0.0).buffer(210.0, resolution=64)
    trap = symmetric_trapezoid(120.0, 145.0, 70.0)
    para = parallelogram(240.0, 100.0, shift=80.0)
    return [
        PartType("rect_155x70", 25, rect),
        PartType("eq_tri_160", 30, tri),
        PartType("circle_r210", 40, circle),
        PartType("trap_120_145_h70", 25, trap),
        PartType("para_b240_h100", 50, para),
    ]


def expand_parts(parts: List[PartType]) -> List[str]:
    expanded: List[str] = []
    for p in parts:
        expanded.extend([p.name] * p.count)
    return expanded


def build_rotated_cache(parts: List[PartType]) -> Dict[str, List[Tuple[float, Polygon]]]:
    pmap = {p.name: p.polygon for p in parts}
    cache: Dict[str, List[Tuple[float, Polygon]]] = {}
    for name, base in pmap.items():
        rots = []
        for ang in unique_rotations(name):
            rp = affinity.rotate(base, ang, origin=(0.0, 0.0))
            rots.append((ang, normalize_polygon(rp)))
        cache[name] = rots
    return cache


def generate_candidates(placed: List[PlacedPart]) -> List[Tuple[float, float]]:
    """锚点为零件归一化后的左下角 (cx,cy)。除包络角点外，增加边中点等，便于落入圆与圆之间的空隙。"""
    points = [(0.0, 0.0)]
    for pp in placed:
        minx, miny, maxx, maxy = pp.polygon.bounds
        mx = (minx + maxx) / 2.0
        my = (miny + maxy) / 2.0
        points.append((maxx + GAP, miny))
        points.append((minx, maxy + GAP))
        points.append((maxx + GAP, maxy + GAP))
        # 边中点外延（常见“条带”与四圆中心类空隙）
        points.append((mx, maxy + GAP))
        points.append((mx, miny))
        points.append((maxx + GAP, my))
        points.append((minx, my))
    # 沿母板左、下边铺稀疏锚点（左下角系），帮助贴边放小件
    _be = 85.0
    y = 0.0
    while y <= BOARD_H + EPS:
        points.append((0.0, y))
        y += _be
    x = 0.0
    while x <= BOARD_W + EPS:
        points.append((x, 0.0))
        x += _be
    # 去重并按“左下优先”
    uniq = sorted(set((round(x, 3), round(y, 3)) for x, y in points), key=lambda t: (t[1], t[0]))
    return uniq


def within_board(poly: Polygon) -> bool:
    minx, miny, maxx, maxy = poly.bounds
    return minx >= -EPS and miny >= -EPS and maxx <= BOARD_W + EPS and maxy <= BOARD_H + EPS


def valid_against_sheet(poly: Polygon, placed: List[PlacedPart]) -> bool:
    pminx, pminy, pmaxx, pmaxy = poly.bounds
    for other in placed:
        ominx, ominy, omaxx, omaxy = other.polygon.bounds
        # 先用AABB快速排除“足够远”的零件，减少昂贵的几何运算
        if pmaxx + GAP <= ominx or omaxx + GAP <= pminx or pmaxy + GAP <= ominy or omaxy + GAP <= pminy:
            continue
        if poly.intersects(other.polygon):
            return False
        # 与 other 的 GAP 邻域相交 => 间距 < GAP；比 distance() 快得多
        if poly.intersects(other.gap_zone()):
            return False
    return True


def try_place_one(
    part_name: str,
    sheet_idx: int,
    sheet_parts: List[PlacedPart],
    rotated_cache: Dict[str, List[Tuple[float, Polygon]]],
    rng: random.Random,
    scan_step: float,
    grid_samples: int = 700,
    max_candidates: int = 600,
    hole_fill_random: int = 0,
    edge_strip_samples: int = 0,
    last_sheet_compact: bool = False,
) -> Optional[PlacedPart]:
    candidates = generate_candidates(sheet_parts)
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
    rotations = rotated_cache[part_name][:]
    rng.shuffle(rotations)

    best: Optional[PlacedPart] = None
    best_key: Tuple[float, ...] = (
        (float("inf"), float("inf"), float("inf"), float("inf"))
        if last_sheet_compact
        else (float("inf"), float("inf"), float("inf"))
    )
    if sheet_parts:
        _, _, cur_maxx, cur_maxy = Polygon(
            [(0.0, 0.0)]
            + [pt for pp in sheet_parts for pt in list(pp.polygon.exterior.coords)[:-1]]
        ).bounds
    else:
        cur_maxx, cur_maxy = 0.0, 0.0

    def union_bbox_key(placed_poly: Polygon) -> Tuple[float, float, float, float]:
        """并入新件后，整张板零件并集的轴对齐包络（用于最后一张：团在一端、少浪费外框）。"""
        a, b, c, d = placed_poly.bounds
        if not sheet_parts:
            return (a, b, c, d)
        uminx = min(min(pp.polygon.bounds[0] for pp in sheet_parts), a)
        uminy = min(min(pp.polygon.bounds[1] for pp in sheet_parts), b)
        umaxx = max(max(pp.polygon.bounds[2] for pp in sheet_parts), c)
        umaxy = max(max(pp.polygon.bounds[3] for pp in sheet_parts), d)
        return (uminx, uminy, umaxx, umaxy)

    def consider(placed_poly: Polygon, ang: float) -> None:
        nonlocal best, best_key
        if not within_board(placed_poly):
            return
        if not valid_against_sheet(placed_poly, sheet_parts):
            return
        if last_sheet_compact:
            uminx, uminy, umaxx, umaxy = union_bbox_key(placed_poly)
            bbox_area = (umaxx - uminx) * (umaxy - uminy)
            # 优先：包络面积小（紧凑、空隙少）；再：靠向板左下角一端（umin+umin 小）
            key = (bbox_area, umaxy, umaxx, uminx + uminy)
        else:
            _, _, maxx, maxy = placed_poly.bounds
            sheet_maxx = max(cur_maxx, maxx)
            sheet_maxy = max(cur_maxy, maxy)
            key = (sheet_maxy, sheet_maxx, sheet_maxx * sheet_maxy)
        if key < best_key:
            best_key = key
            best = PlacedPart(part_name, placed_poly, ang, sheet_idx)

    for cx, cy in candidates:
        for ang, norm_poly in rotations:
            placed_poly = affinity.translate(norm_poly, xoff=cx, yoff=cy)
            consider(placed_poly, ang)

    # 三角/梯形/矩形：在每个旋转下均匀随机试左下角锚点，覆盖“多圆围成”的内部与边带空隙
    if hole_fill_random > 0 and part_name in HOLE_FILL_PART_NAMES:
        n_rot = max(1, len(rotations))
        per_rot = max(1, hole_fill_random // n_rot)
        for ang, norm_poly in rotations:
            _, _, pmaxx, pmaxy = norm_poly.bounds
            max_x = BOARD_W - pmaxx
            max_y = BOARD_H - pmaxy
            if max_x < -EPS or max_y < -EPS:
                continue
            for _ in range(per_rot):
                if last_sheet_compact and rng.random() < 0.62:
                    cx = rng.uniform(0.0, max_x * 0.52)
                    cy = rng.uniform(0.0, max_y * 0.52)
                else:
                    cx = rng.uniform(0.0, max_x)
                    cy = rng.uniform(0.0, max_y)
                placed_poly = affinity.translate(norm_poly, xoff=cx, yoff=cy)
                consider(placed_poly, ang)

    # 三角/梯形：四边条带（非最后一张紧凑模式；最后一张要团在一端，避免沿四边摊开）
    if (
        edge_strip_samples > 0
        and part_name in EDGE_STRIP_PART_NAMES
        and not last_sheet_compact
    ):
        n_rot = max(1, len(rotations))
        per_edge = max(1, edge_strip_samples // (4 * n_rot))
        for ang, norm_poly in rotations:
            _, _, pmaxx, pmaxy = norm_poly.bounds
            max_x = BOARD_W - pmaxx
            max_y = BOARD_H - pmaxy
            if max_x < -EPS or max_y < -EPS:
                continue
            for _ in range(per_edge):
                consider(affinity.translate(norm_poly, xoff=0.0, yoff=rng.uniform(0.0, max_y)), ang)
            for _ in range(per_edge):
                consider(affinity.translate(norm_poly, xoff=rng.uniform(0.0, max_x), yoff=0.0), ang)
            for _ in range(per_edge):
                consider(affinity.translate(norm_poly, xoff=max_x, yoff=rng.uniform(0.0, max_y)), ang)
            for _ in range(per_edge):
                consider(affinity.translate(norm_poly, xoff=rng.uniform(0.0, max_x), yoff=max_y), ang)

    # 候选点失败时，做一次网格回退扫描，提升K较小时的可行率（grid_samples<=0 时跳过以加速）
    if best is None and grid_samples > 0:
        for ang, norm_poly in rotations:
            _, _, pmaxx, pmaxy = norm_poly.bounds
            max_x = BOARD_W - pmaxx
            max_y = BOARD_H - pmaxy
            if max_x < -EPS or max_y < -EPS:
                continue

            xs = [i * scan_step for i in range(int(max_x // scan_step) + 1)]
            ys = [j * scan_step for j in range(int(max_y // scan_step) + 1)]
            if not xs or not ys:
                continue
            grid_points = [(x, y) for y in ys for x in xs]
            rng.shuffle(grid_points)
            for x, y in grid_points[:grid_samples]:
                placed_poly = affinity.translate(norm_poly, xoff=x, yoff=y)
                consider(placed_poly, ang)
    return best


def total_area(parts: List[PartType]) -> float:
    return sum(p.count * p.polygon.area for p in parts)


def board_utilization(sheet_parts: List[PlacedPart]) -> float:
    used = sum(pp.polygon.area for pp in sheet_parts)
    return used / (BOARD_W * BOARD_H)


def copy_sheets(sheets: List[List[PlacedPart]]) -> List[List[PlacedPart]]:
    return [
        [PlacedPart(pp.part_name, pp.polygon, pp.rotation, pp.sheet_index) for pp in sp]
        for sp in sheets
    ]


def greedy_place_pool(
    sheets: List[List[PlacedPart]],
    pool_names: List[str],
    rotated_cache: Dict[str, List[Tuple[float, Polygon]]],
    rng: random.Random,
    scan_step: float,
    grid_samples: int = 700,
    max_candidates: int = 600,
    hole_fill_random: int = 0,
    edge_strip_samples: int = 0,
    last_sheet_cluster: bool = True,
    num_sheets: int = 2,
) -> bool:
    """将 pool 中的零件依次贪心放入各板（先试前板）。成功返回 True。"""
    order = list(pool_names)
    order.sort(key=lambda n: (PART_PRIORITY.get(n, -1), rng.random()), reverse=True)

    for pn in order:
        placed = None
        for sidx in range(len(sheets)):
            trial = try_place_one(
                pn,
                sidx,
                sheets[sidx],
                rotated_cache,
                rng,
                scan_step,
                grid_samples=grid_samples,
                max_candidates=max_candidates,
                hole_fill_random=hole_fill_random,
                edge_strip_samples=edge_strip_samples,
                last_sheet_compact=last_sheet_cluster and sidx == num_sheets - 1,
            )
            if trial is not None:
                sheets[sidx].append(trial)
                placed = trial
                break
        if placed is None:
            return False
    return True


def is_better_two_boards(u0_new: float, u1_new: float, u0_old: float, u1_old: float) -> bool:
    """优先提高第1板利用率；相同时降低第2板利用率。"""
    if u0_new > u0_old + 1e-9:
        return True
    if abs(u0_new - u0_old) <= 1e-9 and u1_new < u1_old - 1e-9:
        return True
    return False


def lns_two_boards(
    initial: List[List[PlacedPart]],
    rotated_cache: Dict[str, List[Tuple[float, Polygon]]],
    rng: random.Random,
    scan_step: float,
    iterations: int,
    remove0_min: int,
    remove0_max: int,
    remove1_min: int,
    remove1_max: int,
    compact_rounds: int,
    grid_samples: int,
    max_candidates: int,
    skip_compact: bool = False,
    hole_fill_random: int = 0,
    edge_strip_samples: int = 0,
    last_sheet_cluster: bool = True,
) -> List[List[PlacedPart]]:
    """大邻域搜索：随机抽掉部分零件再贪心回填，冲击第1板利用率。"""
    if len(initial) != 2:
        return initial

    current = copy_sheets(initial)
    best = copy_sheets(initial)
    u0_cur = board_utilization(current[0])
    u1_cur = board_utilization(current[1])
    u0_best, u1_best = u0_cur, u1_cur
    accepted = 0

    def pick_remove(n: int, lo: int, hi: int) -> int:
        if n <= 0:
            return 0
        hi2 = min(hi, n)
        lo2 = min(lo, hi2)
        if lo2 > hi2:
            return hi2
        return rng.randint(lo2, hi2)

    for it in range(iterations):
        if iterations >= 10 and (it + 1) % max(1, iterations // 10) == 0:
            print(f"  LNS 进度 {it + 1}/{iterations} ...")
        trial = copy_sheets(current)
        n0 = pick_remove(len(trial[0]), remove0_min, remove0_max)
        n1 = pick_remove(len(trial[1]), remove1_min, remove1_max)
        if n0 + n1 == 0:
            continue

        removed0 = rng.sample(trial[0], n0) if n0 else []
        removed1 = rng.sample(trial[1], n1) if n1 else []
        for p in removed0:
            trial[0].remove(p)
        for p in removed1:
            trial[1].remove(p)
        pool = [p.part_name for p in removed0 + removed1]

        if not greedy_place_pool(
            trial,
            pool,
            rotated_cache,
            rng,
            scan_step,
            grid_samples=grid_samples,
            max_candidates=max_candidates,
            hole_fill_random=hole_fill_random,
            edge_strip_samples=edge_strip_samples,
            last_sheet_cluster=last_sheet_cluster,
            num_sheets=len(trial),
        ):
            continue

        if not skip_compact:
            trial = compact_front_sheets(
                trial,
                rotated_cache,
                rng,
                scan_step,
                rounds=compact_rounds,
                grid_samples=grid_samples,
                max_candidates=max_candidates,
                hole_fill_random=hole_fill_random,
                edge_strip_samples=edge_strip_samples,
                last_sheet_cluster=last_sheet_cluster,
            )

        u0 = board_utilization(trial[0])
        u1 = board_utilization(trial[1])

        if is_better_two_boards(u0, u1, u0_cur, u1_cur):
            current = trial
            u0_cur, u1_cur = u0, u1
            accepted += 1
            if is_better_two_boards(u0, u1, u0_best, u1_best):
                best = copy_sheets(trial)
                u0_best, u1_best = u0, u1

    print(f"  LNS 接受更优步数: {accepted}/{iterations}")
    return best


def compact_front_sheets(
    sheets: List[List[PlacedPart]],
    rotated_cache: Dict[str, List[Tuple[float, Polygon]]],
    rng: random.Random,
    scan_step: float,
    rounds: int = 2,
    grid_samples: int = 700,
    max_candidates: int = 600,
    hole_fill_random: int = 0,
    edge_strip_samples: int = 0,
    last_sheet_cluster: bool = True,
) -> List[List[PlacedPart]]:
    if len(sheets) <= 1:
        return sheets

    last_idx = len(sheets) - 1

    for _ in range(rounds):
        moved_any = False
        for target_idx in range(len(sheets) - 1):
            for source_idx in range(len(sheets) - 1, target_idx, -1):
                source = sheets[source_idx]
                if not source:
                    continue

                source_sorted = sorted(
                    source,
                    key=lambda p: (
                        PART_RANK.get(p.part_name, 999),
                        p.polygon.area,
                    ),
                )

                for cand in source_sorted:
                    trial = try_place_one(
                        cand.part_name,
                        target_idx,
                        sheets[target_idx],
                        rotated_cache,
                        rng,
                        scan_step,
                        grid_samples=grid_samples,
                        max_candidates=max_candidates,
                        hole_fill_random=hole_fill_random,
                        edge_strip_samples=edge_strip_samples,
                        last_sheet_compact=last_sheet_cluster and target_idx == last_idx,
                    )
                    if trial is None:
                        continue
                    source.remove(cand)
                    sheets[target_idx].append(trial)
                    moved_any = True
        if not moved_any:
            break
    return sheets


def pack_with_k(
    all_parts: List[str],
    k: int,
    rotated_cache: Dict[str, List[Tuple[float, Polygon]]],
    attempts: int = 12,
    seed: int = 42,
    scan_step: float = 35.0,
    refine_rounds: int = 2,
    hole_fill_random: int = 400,
    grid_samples: int = 700,
    max_candidates: int = 900,
    edge_strip_samples: int = 120,
    last_sheet_cluster: bool = True,
) -> Optional[List[List[PlacedPart]]]:
    best_solution: Optional[List[List[PlacedPart]]] = None
    best_score = None

    for t in range(attempts):
        rng = random.Random(seed + t * 9973 + k * 37)
        order = list(all_parts)
        # 大件优先 + 随机扰动（面积大者优先）
        order.sort(key=lambda n: (PART_PRIORITY.get(n, -1), rng.random()), reverse=True)

        sheets: List[List[PlacedPart]] = [[] for _ in range(k)]
        success = True

        for pn in order:
            placed = None
            # 先尽量塞前面的板，形成“前满后空”的结构
            for sidx in range(k):
                trial = try_place_one(
                    pn,
                    sidx,
                    sheets[sidx],
                    rotated_cache,
                    rng,
                    scan_step,
                    grid_samples=grid_samples,
                    max_candidates=max_candidates,
                    hole_fill_random=hole_fill_random,
                    edge_strip_samples=edge_strip_samples,
                    last_sheet_compact=last_sheet_cluster and sidx == k - 1,
                )
                if trial is not None:
                    sheets[sidx].append(trial)
                    placed = trial
                    break
            if placed is None:
                success = False
                break

        if not success:
            continue

        sheets = compact_front_sheets(
            sheets,
            rotated_cache,
            rng,
            scan_step,
            rounds=refine_rounds,
            grid_samples=grid_samples,
            max_candidates=max_candidates,
            hole_fill_random=hole_fill_random,
            edge_strip_samples=edge_strip_samples,
            last_sheet_cluster=last_sheet_cluster,
        )

        utils = [board_utilization(sp) for sp in sheets]
        used_sheets = sum(1 for u in utils if u > 0.0001)
        # 目标：先最少用板，再前满后空
        score = (
            used_sheets,
            -sum(utils[:-1]),   # 越小越好 => 前面利用率越高
            utils[-1],          # 越小越好 => 最后一张利用率越低
        )
        if best_score is None or score < best_score:
            best_score = score
            best_solution = sheets

    return best_solution


def draw_sheet(
    sheet_parts: List[PlacedPart],
    out_png: Path,
    title: str,
    label_every: int = 1,
    label_include_rotation: bool = True,
) -> None:
    """
    生成更高精度的母板排样图：
    - PNG：更高 DPI
    - SVG：矢量图（用于放大/测量）
    - 圆：标注圆心坐标 O=(x,y) 与半径 R（几何圆，非逼近多边形的顶点）
    - 多边形：标注外轮廓各顶点坐标（含旋转后的实际位置）
    - 每张母板：输出统计信息（利用率、件数、各类型数量、包络框）
    """

    # 统计信息（先算再画，避免重复遍历）
    util = board_utilization(sheet_parts) * 100.0
    total_used_area = sum(pp.polygon.area for pp in sheet_parts)
    counts: Dict[str, int] = {}
    for pp in sheet_parts:
        counts[pp.part_name] = counts.get(pp.part_name, 0) + 1

    if sheet_parts:
        minx = min(pp.polygon.bounds[0] for pp in sheet_parts)
        miny = min(pp.polygon.bounds[1] for pp in sheet_parts)
        maxx = max(pp.polygon.bounds[2] for pp in sheet_parts)
        maxy = max(pp.polygon.bounds[3] for pp in sheet_parts)
    else:
        minx = miny = maxx = maxy = 0.0

    # 输出精度：PNG 提高 DPI；SVG 同时保留用于放大与校核
    dpi = 450
    fig, ax = plt.subplots(figsize=(10, 20), dpi=dpi)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, BOARD_W)
    ax.set_ylim(0, BOARD_H)
    ax.set_title(title)
    ax.set_xlabel("mm")
    ax.set_ylabel("mm")

    uniq_names = sorted({pp.part_name for pp in sheet_parts})
    _palette = (
        "#1f77b4 #ff7f0e #2ca02c #d62728 #9467bd #8c564b #e377c2 "
        "#7f7f7f #bcbd22 #17becf"
    ).split()
    color_map = {n: _palette[i % len(_palette)] for i, n in enumerate(uniq_names)}
    abbr = {
        n: (chr(ord("A") + i) if i < 26 else f"Z{i - 25}")
        for i, n in enumerate(uniq_names)
    }

    # 主编号字号；顶点坐标用更小字号
    font_size = 5 if len(sheet_parts) > 110 else (6 if len(sheet_parts) > 60 else 7)
    vtx_font = max(2.5, font_size - 2.5)

    # 每个零件一个递增编号（按“同类型计数”），用于定位标注
    type_seq: Dict[str, int] = {}

    # 画母板边界
    ax.plot([0, BOARD_W, BOARD_W, 0, 0], [0, 0, BOARD_H, BOARD_H, 0], color="black", linewidth=1.0)

    for pp in sheet_parts:
        type_seq[pp.part_name] = type_seq.get(pp.part_name, 0) + 1
        idx_in_type = type_seq[pp.part_name]
        label_core = f"{abbr.get(pp.part_name, '?')}{idx_in_type}"
        theta = pp.rotation

        x, y = pp.polygon.exterior.xy
        pts = list(zip(x, y))
        patch = MplPolygon(
            pts,
            closed=True,
            facecolor=color_map.get(pp.part_name, "#999999"),
            edgecolor="black",
            alpha=0.75,
            linewidth=0.6,
        )
        ax.add_patch(patch)

        if label_every > 1 and (idx_in_type % label_every != 0):
            continue

        if pp.part_name in CIRCLE_RADIUS_BY_NAME:
            # 圆：圆心 + 半径（几何定义）；圆心取质心（与理想圆一致）
            cen = pp.polygon.centroid
            ox, oy = float(cen.x), float(cen.y)
            r_mm = CIRCLE_RADIUS_BY_NAME[pp.part_name]
            lines = [f"{label_core}", f"O=({ox:.2f},{oy:.2f})", f"R={r_mm:.0f}mm"]
            if label_include_rotation:
                lines.append(f"θ={theta:.1f}°")
            ax.text(
                ox,
                oy,
                "\n".join(lines),
                fontsize=font_size,
                ha="center",
                va="center",
                color="black",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.6, pad=0.8),
            )
        else:
            # 多边形：各顶点坐标 + 质心处零件编号与旋转
            poly = pp.polygon
            cent = poly.centroid
            cx, cy = float(cent.x), float(cent.y)
            coords = list(poly.exterior.coords[:-1])
            for i, (vx, vy) in enumerate(coords):
                vx, vy = float(vx), float(vy)
                dx, dy = vx - cx, vy - cy
                norm = math.hypot(dx, dy) or 1.0
                off = 5.0
                tx, ty = vx + off * dx / norm, vy + off * dy / norm
                ax.text(
                    tx,
                    ty,
                    f"V{i + 1}:({vx:.2f},{vy:.2f})",
                    fontsize=vtx_font,
                    ha="center",
                    va="center",
                    color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.45, pad=0.3),
                )
                ax.plot(vx, vy, "k.", markersize=1.2)
            rp = poly.representative_point()
            rpx, rpy = float(rp.x), float(rp.y)
            id_lines = [label_core]
            if label_include_rotation:
                id_lines.append(f"θ={theta:.1f}°")
            ax.text(
                rpx,
                rpy,
                "\n".join(id_lines),
                fontsize=font_size,
                ha="center",
                va="center",
                color="navy",
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.5, pad=0.6),
            )

    # 母板统计文本框
    bbox_area = (maxx - minx) * (maxy - miny) if sheet_parts else 0.0
    stats = (
        f"Utilization: {util:.2f}%\n"
        f"Parts: {len(sheet_parts)}\n"
        f"UsedArea: {total_used_area:,.0f} mm^2\n"
        f"Extent: [{minx:.0f},{miny:.0f}] - [{maxx:.0f},{maxy:.0f}]\n"
        f"ExtentArea: {bbox_area:,.0f} mm^2\n"
        f"Counts: {', '.join(f'{abbr.get(k, k)}={v}' for k, v in sorted(counts.items()))}\n"
        f"Annot: circle O+R(mm); poly vertex V1..Vn (mm); theta=deg"
    )
    ax.text(
        15,
        BOARD_H - 40,
        stats,
        fontsize=8,
        ha="left",
        va="top",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=6.0),
    )

    # 画网格会影响读数；保留坐标刻度即可
    ax.grid(False)
    fig.tight_layout()
    fig.savefig(out_png)
    # 同时输出矢量图（用于高精度查看/放大）
    try:
        out_svg = out_png.with_suffix(".svg")
        fig.savefig(out_svg)
    except Exception:
        pass
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="2D排样求解器（启发式）")
    parser.add_argument("--attempts", type=int, default=12, help="每个母板数K的随机尝试次数")
    parser.add_argument("--max-extra-k", type=int, default=4, help="在面积下界基础上最多额外尝试多少张母板")
    parser.add_argument("--scan-step", type=float, default=35.0, help="候选失败时网格回退步长(mm)，越小越慢但更易找到可行解")
    parser.add_argument("--refine-rounds", type=int, default=2, help="后板向前板回填优化轮数")
    parser.add_argument("--fixed-k", type=int, default=0, help="固定母板数进行优化（>0时跳过K搜索）")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="JSON 配置文件路径（母板尺寸、间隙、零件形状/数量）；未指定则使用内置默认",
    )
    parser.add_argument("--seed", type=int, default=20260401, help="随机种子")
    parser.add_argument("--lns-iters", type=int, default=0, help="两板大邻域搜索迭代次数，0表示关闭")
    parser.add_argument("--lns-remove0-min", type=int, default=8, help="LNS每轮从第1板最少移除零件数")
    parser.add_argument("--lns-remove0-max", type=int, default=22, help="LNS每轮从第1板最多移除零件数")
    parser.add_argument("--lns-remove1-min", type=int, default=0, help="LNS每轮从第2板最少移除零件数")
    parser.add_argument("--lns-remove1-max", type=int, default=12, help="LNS每轮从第2板最多移除零件数")
    parser.add_argument("--lns-compact-rounds", type=int, default=2, help="LNS每轮成功后的回填轮数（越小越快）")
    parser.add_argument(
        "--lns-grid-samples",
        type=int,
        default=0,
        help="LNS内网格回退采样点数，0表示关闭网格仅锚点（最快，失败率更高）",
    )
    parser.add_argument("--lns-max-candidates", type=int, default=220, help="LNS内候选锚点上限（越小越快）")
    parser.add_argument(
        "--lns-no-compact",
        action="store_true",
        help="LNS 每轮成功后跳过 compact_front_sheets（快很多，探索更随机）",
    )
    parser.add_argument(
        "--hole-fill",
        type=int,
        default=400,
        help="三角/梯形/矩形在全板随机试左下角锚点的次数，用于填圆阵中间与边带空隙（0关闭，越大越慢）",
    )
    parser.add_argument("--grid-samples", type=int, default=700, help="候选点失败时网格回退采样点数")
    parser.add_argument("--max-candidates", type=int, default=900, help="锚点候选数量上限（含边中点）")
    parser.add_argument(
        "--edge-strip",
        type=int,
        default=120,
        help="三角/梯形沿母板四边条带的采样次数预算（0关闭，越大越易贴边，略慢）",
    )
    parser.add_argument(
        "--no-last-sheet-cluster",
        action="store_true",
        help="关闭最后一张母板约束（默认开启：团在左下、包络尽量紧）",
    )
    args = parser.parse_args()
    last_sheet_cluster = not args.no_last_sheet_cluster

    cfg_path = Path(args.config) if args.config.strip() else None
    parts = load_packing_config(cfg_path)
    all_parts = expand_parts(parts)
    cache = build_rotated_cache(parts)

    board_area = BOARD_W * BOARD_H
    lb = math.ceil(total_area(parts) / board_area)

    print("=== 问题信息 ===")
    print(f"母板尺寸: {BOARD_W:.0f} x {BOARD_H:.0f} mm")
    print(f"总零件数: {len(all_parts)}")
    print(f"总零件面积: {total_area(parts):.2f} mm^2")
    print(f"面积下界母板数: {lb}")

    best_k = None
    best_solution = None

    if args.fixed_k > 0:
        k_values = [args.fixed_k]
    else:
        k_values = list(range(lb, lb + args.max_extra_k + 1))

    for k in k_values:
        print(f"\n尝试母板数 K={k} ...")
        solution = pack_with_k(
            all_parts,
            k,
            cache,
            attempts=args.attempts,
            seed=args.seed,
            scan_step=args.scan_step,
            refine_rounds=args.refine_rounds,
            hole_fill_random=args.hole_fill,
            grid_samples=args.grid_samples,
            max_candidates=args.max_candidates,
            edge_strip_samples=args.edge_strip,
            last_sheet_cluster=last_sheet_cluster,
        )
        if solution is not None:
            best_k = k
            best_solution = solution
            print(f"K={k} 可行。")
            break
        print(f"K={k} 未找到可行解。")

    if best_solution is None or best_k is None:
        print("\n未在搜索范围内找到可行解，请提高 attempts 或扩大 K 搜索范围。")
        return

    if best_k == 2 and args.lns_iters > 0:
        print("\n=== LNS 大邻域搜索（两板） ===")
        u0_before = board_utilization(best_solution[0])
        u1_before = board_utilization(best_solution[1])
        print(f"LNS 前: 第1板利用率={u0_before * 100:.2f}%, 第2板={u1_before * 100:.2f}%")
        rng_lns = random.Random(args.seed ^ 0x9E3779B9)
        best_solution = lns_two_boards(
            best_solution,
            cache,
            rng_lns,
            args.scan_step,
            args.lns_iters,
            args.lns_remove0_min,
            args.lns_remove0_max,
            args.lns_remove1_min,
            args.lns_remove1_max,
            args.lns_compact_rounds,
            args.lns_grid_samples,
            args.lns_max_candidates,
            skip_compact=args.lns_no_compact,
            hole_fill_random=args.hole_fill,
            edge_strip_samples=args.edge_strip,
            last_sheet_cluster=last_sheet_cluster,
        )
        u0_after = board_utilization(best_solution[0])
        u1_after = board_utilization(best_solution[1])
        print(f"LNS 后: 第1板利用率={u0_after * 100:.2f}%, 第2板={u1_after * 100:.2f}%")

    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)

    print("\n=== 最终结果 ===")
    print(f"最少母板数（在当前求解参数下）: {best_k}")
    used_sheets = 0
    for idx, sp in enumerate(best_solution, start=1):
        util = board_utilization(sp) * 100.0
        count = len(sp)
        by_type: Dict[str, int] = {}
        for pp in sp:
            by_type[pp.part_name] = by_type.get(pp.part_name, 0) + 1
        if count > 0:
            used_sheets += 1
        print(f"第{idx}张板: 零件数={count}, 利用率={util:.2f}%")
        print(f"  类型分布: {by_type}")
        png = out_dir / f"sheet_{idx}.png"
        draw_sheet(sp, png, title=f"Sheet {idx} | Parts={count} | Util={util:.2f}%")

    print(f"\n实际使用母板数: {used_sheets}")
    print(f"布置图输出目录: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
