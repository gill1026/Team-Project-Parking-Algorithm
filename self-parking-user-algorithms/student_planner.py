from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------- 차량/플래너 파라미터 -----------------
L_WHEELBASE = 2.6
LF = 1.6
LR = 1.4
HALF_W = 0.8
MAX_STEER = math.radians(35.0)
R_MIN = L_WHEELBASE / math.tan(MAX_STEER)      # 최소 회전 반경 ≈ 3.71
BODY_OFFSET = (LF - LR) * 0.5                   # 0.1: 뒷축→차체중심(노즈 방향)

# 차체 근사 원(차체 좌표계, 뒷축 원점, 노즈=+x). 차체를 완전히 덮음.
CIRCLE_XC = (-0.9, 0.1, 1.1)
CIRCLE_R = 0.95

# Hybrid A*
DS = 0.7
SUBSTEPS = 3
STEER_SET = (-1.0, -0.5, 0.0, 0.5, 1.0)
XY_RES = 0.5
YAW_BINS = 24
REVERSE_PENALTY = 1.6
GEAR_SWITCH_PENALTY = 6.0
STEER_PENALTY = 0.6
GOAL_XY_TOL = 0.6
GOAL_YAW_TOL = math.radians(18.0)
# 종료는 결정론적인 확장 횟수로 제어한다(머신 부하와 무관 → 재현성 보장).
# 실측 최악(stage3) ~5000 확장 → 12000 캡은 2배 이상 여유이며, 모든 맵이 캡에
# 닿기 전에 목표를 찾는다. 벽시계는 예기치 못한 행(hang) 방지용 안전장치로만 둔다.
MAX_EXPANSIONS = 12000
PLAN_TIME_SAFETY = 3.0

# 충돌/휴리스틱 그리드
COLL_RES = 0.2
OCC_BASE = 0.22   # 점유 슬롯 추가 여유(추적오차 흡수 → dense 맵 충돌 방지)
LINE_BASE = 0.05
LINE_HALF_WIDTH = 0.25
HEUR_RES = 0.5
HEUR_BASE = 0.10

# Dubins shot
SHOT_RANGE = 14.0     # 목표까지 이 거리 이내일 때만 시도
SHOT_THROTTLE = 2
SHOT_MAX_WORDS = 2    # 최단 단어 몇 개만 검사(실패 비용 절감)

ENTRY_LEN = 3.4

# 제어
LD_MIN = 1.3
LD_MAX = 3.0
LD_BASE = 1.0
LD_GAIN = 0.7
STEER_SMOOTH = 0.35   # 직전 조향과 블렌딩(저역통과) → 진동·반전 감소
# 곡률 인지 속도 프로파일
V_STRAIGHT = 2.9      # 긴 직선 전진 상한
SPEED_REV = 1.2       # 후진 구간 상한
SPEED_ENTRY = 0.60    # 슬롯 직선 진입 상한(낮을수록 중심에서 더 느리게 정지)
V_CURVE_MIN = 1.1     # 곡선 최저 속도
A_LAT = 0.42          # 허용 횡가속 → 곡률→속도 (R_min에서 ≈1.25 m/s)
A_DECEL = 2.5         # 프로파일 감속도(여유: maxBrake 7)
A_ACCEL = 1.9         # 프로파일 가속 램프
SEG_END_TOL = 0.30
STOP_SPEED = 0.06
SEG_LINK_SPEED = 0.5   # 비-마지막 구간 끝 도착 속도(0 으로 기면 끝점에서 맴돌다 정체)
LAST_HARD_BRAKE = 0.5  # 마지막 구간: 끝 0.5m 전부터 강제 제동(IoU 유지·저속 정지)
STEER_DEADZONE = math.radians(3.0)  # 미세 진동 제거 → 조향 반전 횟수 감소
DENSIFY = 0.2


LAST_EXPANSIONS = 0


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _mod2pi(a: float) -> float:
    return a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi))


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)


# ----------------- Dubins -----------------
def _dubins_words(alpha, beta, d):
    sa, sb = math.sin(alpha), math.sin(beta)
    ca, cb = math.cos(alpha), math.cos(beta)
    c_ab = math.cos(alpha - beta)
    out = []

    # LSL
    tmp0 = d + sa - sb
    p2 = 2 + d * d - 2 * c_ab + 2 * d * (sa - sb)
    if p2 >= 0:
        tmp1 = math.atan2((cb - ca), tmp0)
        t = _mod2pi(-alpha + tmp1); p = math.sqrt(p2); q = _mod2pi(beta - tmp1)
        out.append((t + p + q, (t, p, q), ("L", "S", "L")))
    # RSR
    tmp0 = d - sa + sb
    p2 = 2 + d * d - 2 * c_ab + 2 * d * (sb - sa)
    if p2 >= 0:
        tmp1 = math.atan2((ca - cb), tmp0)
        t = _mod2pi(alpha - tmp1); p = math.sqrt(p2); q = _mod2pi(-beta + tmp1)
        out.append((t + p + q, (t, p, q), ("R", "S", "R")))
    # LSR
    p2 = -2 + d * d + 2 * c_ab + 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp2 = math.atan2((-ca - cb), (d + sa + sb)) - math.atan2(-2.0, p)
        t = _mod2pi(-alpha + tmp2); q = _mod2pi(-_mod2pi(beta) + tmp2)
        out.append((t + p + q, (t, p, q), ("L", "S", "R")))
    # RSL
    p2 = d * d - 2 + 2 * c_ab - 2 * d * (sa + sb)
    if p2 >= 0:
        p = math.sqrt(p2)
        tmp2 = math.atan2((ca + cb), (d - sa - sb)) - math.atan2(2.0, p)
        t = _mod2pi(alpha - tmp2); q = _mod2pi(_mod2pi(beta) - tmp2)
        out.append((t + p + q, (t, p, q), ("R", "S", "L")))
    # RLR
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (sa - sb)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(alpha - math.atan2(ca - cb, d - sa + sb) + p / 2.0)
        q = _mod2pi(alpha - beta - t + p)
        out.append((t + p + q, (t, p, q), ("R", "L", "R")))
    # LRL
    tmp = (6.0 - d * d + 2 * c_ab + 2 * d * (-sa + sb)) / 8.0
    if abs(tmp) <= 1.0:
        p = _mod2pi(2 * math.pi - math.acos(tmp))
        t = _mod2pi(-alpha - math.atan2(ca - cb, d + sa - sb) + p / 2.0)
        q = _mod2pi(_mod2pi(beta) - alpha - t + p)
        out.append((t + p + q, (t, p, q), ("L", "R", "L")))

    out.sort(key=lambda w: w[0])
    return out


def _dubins_sample(start, lengths, modes, R, step=0.25):
    """forward Dubins 경로 샘플 (x,y,yaw) 리스트 반환 (start 포함, goal 끝점 포함)."""
    x, y, yaw = start
    pts = [(x, y, yaw)]
    for seg_len_norm, mode in zip(lengths, modes):
        seg_len = seg_len_norm * R
        n = max(1, int(math.ceil(seg_len / step)))
        ds = seg_len / n
        if mode == "S":
            for _ in range(n):
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
                pts.append((x, y, yaw))
        else:
            curv = 1.0 / R if mode == "L" else -1.0 / R
            for _ in range(n):
                yaw = _wrap(yaw + curv * ds)
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
                pts.append((x, y, yaw))
    return pts


def dubins_shot(start, goal, R, env):
    """충돌 없는 최단 forward Dubins 경로 샘플 반환, 없으면 None."""
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    dx, dy = gx - sx, gy - sy
    Dd = math.hypot(dx, dy)
    if Dd < 1e-6:
        return None
    d = Dd / R
    th = math.atan2(dy, dx)
    alpha = _wrap(syaw - th)
    beta = _wrap(gyaw - th)
    max_len = 3.0 * Dd + 5.0   # 과도한 우회 루프는 dense 맵에서 항상 충돌 → 건너뜀
    for total, lengths, modes in _dubins_words(alpha, beta, d)[:SHOT_MAX_WORDS]:
        if total * R > max_len:
            continue
        pts = _dubins_sample(start, lengths, modes, R)
        ok = True
        for (x, y, yaw) in pts:
            if env.collide(x, y, yaw):
                ok = False
                break
        if ok:
            return pts
    return None


# ----------------- 충돌 환경 -----------------
class CollisionEnv:
    """점유 슬롯 + 주차 라인 + 경계로 충돌/휴리스틱 그리드를 구성.
    차체는 3원 근사로 검사(보수적)."""

    def __init__(self, extent, occ_rects, line_rects):
        self.xmin, self.xmax, self.ymin, self.ymax = extent

        self.cres = COLL_RES
        self.cnx = max(1, int((self.xmax - self.xmin) / self.cres) + 1)
        self.cny = max(1, int((self.ymax - self.ymin) / self.cres) + 1)
        self.cgrid = np.zeros((self.cny, self.cnx), dtype=bool)
        for r in occ_rects:
            self._mark(self.cgrid, self.cres, r, CIRCLE_R + OCC_BASE)
        for r in line_rects:
            self._mark(self.cgrid, self.cres, r, CIRCLE_R + LINE_BASE)

        self.hres = HEUR_RES
        self.hnx = max(1, int((self.xmax - self.xmin) / self.hres) + 1)
        self.hny = max(1, int((self.ymax - self.ymin) / self.hres) + 1)
        self.hblock = np.zeros((self.hny, self.hnx), dtype=bool)
        for r in occ_rects:
            self._mark(self.hblock, self.hres, r, CIRCLE_R + HEUR_BASE)
        for r in line_rects:
            self._mark(self.hblock, self.hres, r, CIRCLE_R + HEUR_BASE)

    def _mark(self, grid, res, rect, infl):
        x0, x1, y0, y1 = rect
        c0 = max(0, int((x0 - infl - self.xmin) / res))
        c1 = min(grid.shape[1] - 1, int(math.ceil((x1 + infl - self.xmin) / res)))
        r0 = max(0, int((y0 - infl - self.ymin) / res))
        r1 = min(grid.shape[0] - 1, int(math.ceil((y1 + infl - self.ymin) / res)))
        if c1 >= c0 and r1 >= r0:
            grid[r0:r1 + 1, c0:c1 + 1] = True

    def collide(self, x, y, yaw) -> bool:
        c = math.cos(yaw); s = math.sin(yaw)
        xmin = self.xmin; ymin = self.ymin
        bxmax = self.xmax - CIRCLE_R; bymax = self.ymax - CIRCLE_R
        bxmin = xmin + CIRCLE_R; bymin = ymin + CIRCLE_R
        cres = self.cres; grid = self.cgrid
        cnx = self.cnx; cny = self.cny
        for xc in CIRCLE_XC:
            wx = x + c * xc
            wy = y + s * xc
            if wx < bxmin or wx > bxmax or wy < bymin or wy > bymax:
                return True
            col = int((wx - xmin) / cres)
            row = int((wy - ymin) / cres)
            if 0 <= row < cny and 0 <= col < cnx and grid[row, col]:
                return True
        return False

    def heuristic_map(self, goal_xy) -> np.ndarray:
        gx, gy = goal_xy
        gc = min(max(int((gx - self.xmin) / self.hres), 0), self.hnx - 1)
        gr = min(max(int((gy - self.ymin) / self.hres), 0), self.hny - 1)
        dist = np.full((self.hny, self.hnx), np.inf, dtype=np.float64)
        if self.hblock[gr, gc]:
            free = np.argwhere(~self.hblock)
            if free.size:
                d2 = (free[:, 0] - gr) ** 2 + (free[:, 1] - gc) ** 2
                gr, gc = free[d2.argmin()]
        dist[gr, gc] = 0.0
        pq = [(0.0, int(gr), int(gc))]
        nbrs = ((1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
                (1, 1, 1.4142), (1, -1, 1.4142), (-1, 1, 1.4142), (-1, -1, 1.4142))
        H, W = self.hny, self.hnx
        hres = self.hres
        while pq:
            dval, r, c = heapq.heappop(pq)
            if dval > dist[r, c]:
                continue
            for dr, dc, w in nbrs:
                nr = r + dr; nc = c + dc
                if 0 <= nr < H and 0 <= nc < W and not self.hblock[nr, nc]:
                    nd = dval + w * hres
                    if nd < dist[nr, nc]:
                        dist[nr, nc] = nd
                        heapq.heappush(pq, (nd, nr, nc))
        return dist

    def heur_lookup(self, dist, x, y, goal_xy) -> float:
        c = int((x - self.xmin) / self.hres)
        r = int((y - self.ymin) / self.hres)
        eucl = math.hypot(x - goal_xy[0], y - goal_xy[1])
        if 0 <= r < self.hny and 0 <= c < self.hnx:
            dv = dist[r, c]
            if math.isfinite(dv):
                return max(dv, eucl)
        return eucl


# ----------------- Hybrid A* -----------------
def hybrid_astar(env: CollisionEnv, start, goal, t_deadline):
    """start/goal=(x,y,yaw). 성공 시 (x,y,yaw,gear) 리스트(시작→목표) 반환, 실패 None."""
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    dist_map = env.heuristic_map((gx, gy))
    yaw_step = 2.0 * math.pi / YAW_BINS

    def key(x, y, yaw):
        return (int(round(x / XY_RES)), int(round(y / XY_RES)),
                int(round(_wrap(yaw) / yaw_step)) % YAW_BINS)

    nx_ = [sx]; ny_ = [sy]; nyaw = [syaw]; ngear = ['D']; nsteer = [0.0]
    nparent = [-1]; ng = [0.0]
    nshot = [None]   # dubins tail attached at this node (list of (x,y,yaw)) or None
    h0 = env.heur_lookup(dist_map, sx, sy, (gx, gy))
    pq = [(h0, 0)]
    best_g = {key(sx, sy, syaw): 0.0}

    expansions = 0
    found = -1
    ss = DS / SUBSTEPS

    while pq:
        expansions += 1
        if expansions > MAX_EXPANSIONS:
            break
        # 벽시계는 안전장치로만(정상 케이스에선 절대 발동하지 않음 → 결정론 유지)
        if t_deadline is not None and expansions % 4000 == 0 and time.time() > t_deadline:
            break
        _, i = heapq.heappop(pq)
        cx, cy, cyaw, cgear, csteer, cg = nx_[i], ny_[i], nyaw[i], ngear[i], nsteer[i], ng[i]

        dpos = math.hypot(cx - gx, cy - gy)
        if dpos <= GOAL_XY_TOL and abs(_wrap(cyaw - gyaw)) <= GOAL_YAW_TOL:
            found = i
            break

        # Dubins 해석적 확장
        if dpos <= SHOT_RANGE and expansions % SHOT_THROTTLE == 0:
            tail = dubins_shot((cx, cy, cyaw), goal, R_MIN, env)
            if tail is not None:
                nshot[i] = tail
                found = i
                break

        for gear in ('D', 'R'):
            direction = 1.0 if gear == 'D' else -1.0
            for st in STEER_SET:
                steer = st * MAX_STEER
                x, y, yaw = cx, cy, cyaw
                ok = True
                for _ in range(SUBSTEPS):
                    yaw = _wrap(yaw + direction * ss / L_WHEELBASE * math.tan(steer))
                    x += direction * ss * math.cos(yaw)
                    y += direction * ss * math.sin(yaw)
                    if env.collide(x, y, yaw):
                        ok = False
                        break
                if not ok:
                    continue
                add = DS * (REVERSE_PENALTY if gear == 'R' else 1.0)
                if gear != cgear:
                    add += GEAR_SWITCH_PENALTY
                add += STEER_PENALTY * abs(steer - csteer)
                ngc = cg + add
                k = key(x, y, yaw)
                if k in best_g and best_g[k] <= ngc:
                    continue
                best_g[k] = ngc
                ni = len(nx_)
                nx_.append(x); ny_.append(y); nyaw.append(yaw)
                ngear.append(gear); nsteer.append(steer)
                nparent.append(i); ng.append(ngc); nshot.append(None)
                f = ngc + env.heur_lookup(dist_map, x, y, (gx, gy))
                heapq.heappush(pq, (f, ni))

    global LAST_EXPANSIONS
    LAST_EXPANSIONS = expansions
    if found < 0:
        return None

    path = []
    i = found
    tail = nshot[i]
    while i >= 0:
        path.append((nx_[i], ny_[i], nyaw[i], ngear[i]))
        i = nparent[i]
    path.reverse()
    if tail is not None:
        # dubins tail 은 모두 전진(D)
        for (x, y, yaw) in tail[1:]:
            path.append((x, y, yaw, 'D'))
    return path


# ----------------- 경로 → 세그먼트 -----------------
def _path_to_segments(path):
    segs = []
    if not path:
        return segs
    cur_gear = path[0][3]
    pts = [(path[0][0], path[0][1])]
    for (x, y, yaw, gear) in path[1:]:
        if gear != cur_gear:
            segs.append({"gear": cur_gear, "pts": pts})
            pts = [pts[-1]]
            cur_gear = gear
        pts.append((x, y))
    segs.append({"gear": cur_gear, "pts": pts})
    out = []
    for s in segs:
        dp = _densify(s["pts"], DENSIFY)
        if len(dp) >= 2 and _polyline_len(dp) > 0.25:
            out.append({"gear": s["gear"], "pts": np.array(dp, dtype=float),
                        "vmax": V_STRAIGHT if s["gear"] == 'D' else SPEED_REV})
    return out


def _densify(pts, step):
    if len(pts) < 2:
        return list(pts)
    out = [pts[0]]
    for i in range(1, len(pts)):
        x0, y0 = out[-1]
        x1, y1 = pts[i]
        d = math.hypot(x1 - x0, y1 - y0)
        if d < 1e-6:
            continue
        n = max(1, int(d / step))
        for k in range(1, n + 1):
            t = k / n
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    return out


def _polyline_len(pts):
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
               for i in range(1, len(pts)))


def _curve_speed(kappa, vmax):
    if kappa < 1e-4:
        return vmax
    return max(V_CURVE_MIN, min(vmax, math.sqrt(A_LAT / kappa)))


def _speed_profile(pts, vmax, end_v):
    """곡률·감속(backward)·가속(forward) 패스로 점별 속도 한계 배열을 만든다.
    lookahead 감속이라 곡선/구간끝 전에 미리 줄어 오버슈트가 없다(start_v=0: 구간
    시작은 항상 정지 상태)."""
    n = len(pts)
    if n == 1:
        return [0.0]
    ds = [0.0] * n
    for i in range(1, n):
        ds[i] = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    kap = [0.0] * n
    for i in range(1, n - 1):
        a1 = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
        a2 = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        dl = 0.5 * (ds[i] + ds[i + 1])
        kap[i] = abs(_wrap(a2 - a1)) / dl if dl > 1e-6 else 0.0
    if n > 2:
        kap[0] = kap[1]; kap[-1] = kap[-2]
    v = [_curve_speed(kap[i], vmax) for i in range(n)]
    v[-1] = min(v[-1], end_v)
    # backward pass: 곡선·구간끝 전에 미리 감속(룩어헤드). 가속 램프는 제어의
    # 물리 가속이 담당하므로 forward pass 는 두지 않는다(시작점 0 → 정지 버그 방지).
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * A_DECEL * ds[i + 1]))
    return v


# ----------------- 플래너 -----------------
@dataclass
class StudentPlanner:
    map_data: Optional[Dict[str, Any]] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    occ_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)
    line_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)
    lines_raw: Any = None
    expected_orientation: Optional[str] = None
    env: Optional[CollisionEnv] = None

    segments: Optional[List[Dict[str, Any]]] = None
    seg_i: int = 0
    seg_pi: int = 0
    done: bool = False
    plan_target: Optional[Tuple[float, float]] = None
    prev_t: float = 1e9
    prev_steer: float = 0.0

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        self.map_data = map_payload
        self.extent = tuple(map(float, map_payload.get("extent", (0, 0, 0, 0))))
        slots = map_payload.get("slots") or []
        occ = map_payload.get("occupied_idx") or []
        self.occ_rects = [tuple(map(float, slots[i]))
                          for i in range(len(slots)) if i < len(occ) and occ[i]]
        self.lines_raw = map_payload.get("lines") or []
        self.line_rects = self._compute_line_rects(self.lines_raw)
        self.expected_orientation = map_payload.get("expected_orientation")
        self.env = None
        pretty_print_map_summary(map_payload)
        self._reset_path()

    def _reset_path(self):
        self.segments = None
        self.seg_i = 0
        self.seg_pi = 0
        self.done = False
        self.plan_target = None

    def _compute_line_rects(self, lines):
        if self.extent is None:
            return []
        xmin, xmax, ymin, ymax = self.extent
        hw = LINE_HALF_WIDTH
        out = []
        for ln in lines:
            x1, y1, x2, y2 = map(float, ln)
            if abs(x1 - x2) < 1e-6:
                xa, xb = min(x1, x2) - hw, max(x1, x2) + hw
                ya, yb = min(y1, y2), max(y1, y2)
            elif abs(y1 - y2) < 1e-6:
                xa, xb = min(x1, x2), max(x1, x2)
                ya, yb = min(y1, y2) - hw, max(y1, y2) + hw
            else:
                xa, xb = min(x1, x2) - hw, max(x1, x2) + hw
                ya, yb = min(y1, y2) - hw, max(y1, y2) + hw
            xa = max(xa, xmin); xb = min(xb, xmax)
            ya = max(ya, ymin); yb = min(yb, ymax)
            if xb > xa and yb > ya:
                out.append((xa, xb, ya, yb))
        return out

    def _entry_geometry(self, target_slot):
        x0, x1, y0, y1 = target_slot
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        xmin, xmax, ymin, ymax = self.extent

        low_blocked = self._horiz_line_near(cx, cy, side="low")
        high_blocked = self._horiz_line_near(cx, cy, side="high")
        if low_blocked and not high_blocked:
            open_high = True
        elif high_blocked and not low_blocked:
            open_high = False
        elif not low_blocked and not high_blocked:
            open_high = (cy - ymin) <= (ymax - cy)
        else:
            open_high = True

        ty = -1.0 if open_high else +1.0   # 슬롯 안으로 들어가는 진행 부호

        if self.expected_orientation == "front_in":
            hy = +1.0
        elif self.expected_orientation == "rear_in":
            hy = -1.0
        else:
            hy = ty
        yaw = math.pi / 2.0 if hy > 0 else -math.pi / 2.0
        entry_gear = 'D' if hy * ty > 0 else 'R'
        goal_rear = (cx, cy - BODY_OFFSET * hy)

        # 사전진입(pre-entry): 열린 변 쪽 인접 통로의 "중앙"에 배치한다.
        # (통로 가장자리에 두면 Dubins shot 이 라인에 막혀 수렴이 어렵다.)
        open_edge = y1 if open_high else y0   # 열린 변의 슬롯 경계(resized)
        far = self._next_line_on_open_side(cx, open_edge, open_high, ymin, ymax)
        if open_high:
            aisle_far = far - 0.3
            pre_y = 0.5 * (open_edge + aisle_far)
            pre_y = min(max(pre_y, open_edge + (LR + 0.4)), aisle_far - (LF + 0.4))
            pre_y = max(pre_y, open_edge + 1.0)
        else:
            aisle_far = far + 0.3
            pre_y = 0.5 * (open_edge + aisle_far)
            pre_y = max(min(pre_y, open_edge - (LR + 0.4)), aisle_far + (LF + 0.4))
            pre_y = min(pre_y, open_edge - 1.0)
        pre_rear = (cx, pre_y)
        return {"cx": cx, "cy": cy, "yaw": yaw, "hy": hy, "ty": ty,
                "entry_gear": entry_gear, "goal_rear": goal_rear, "pre_rear": pre_rear}

    def _next_line_on_open_side(self, cx, open_edge, open_high, ymin, ymax):
        """열린 변 너머(통로 건너편) 가장 가까운 수평 라인 y. 없으면 맵 경계."""
        best = ymax if open_high else ymin
        for ln in self.lines_raw:
            x1, y1, x2, y2 = map(float, ln)
            if abs(y1 - y2) > 1e-6:
                continue
            ly = y1
            if not (min(x1, x2) - 0.3 <= cx <= max(x1, x2) + 0.3):
                continue
            if open_high and ly > open_edge + 0.4:
                best = min(best, ly)
            elif (not open_high) and ly < open_edge - 0.4:
                best = max(best, ly)
        return best

    def _horiz_line_near(self, cx, cy, side):
        for ln in self.lines_raw:
            x1, y1, x2, y2 = map(float, ln)
            if abs(y1 - y2) > 1e-6:
                continue
            ly = y1
            if not (min(x1, x2) - 0.3 <= cx <= max(x1, x2) + 0.3):
                continue
            if side == "low" and (cy - 3.2) <= ly <= (cy + 0.3):
                return True
            if side == "high" and (cy - 0.3) <= ly <= (cy + 3.2):
                return True
        return False

    def _plan(self, obs):
        st = obs.get("state", {})
        start = (float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                 float(st.get("yaw", 0.0)))
        target_slot = tuple(map(float, obs.get("target_slot", (0, 0, 0, 0))))
        geo = self._entry_geometry(target_slot)
        self.plan_target = (geo["cx"], geo["cy"])

        if self.env is None:
            self.env = CollisionEnv(self.extent, self.occ_rects, self.line_rects)
        env = self.env

        goal_pose = (geo["pre_rear"][0], geo["pre_rear"][1], geo["yaw"])
        t0 = time.time()
        path = hybrid_astar(env, start, goal_pose, time.time() + PLAN_TIME_SAFETY)

        entry_pts = _densify([geo["pre_rear"], geo["goal_rear"]], DENSIFY)
        segments = _path_to_segments(path) if path else None

        if segments:
            # 접근 세그먼트 스무딩: 출렁임 제거 → 조향반전·거리 감소 (충돌 시 되돌림)
            for s in segments:
                s["pts"] = self._smooth_seg(s["pts"], s["gear"])
            segments.append({"gear": geo["entry_gear"],
                             "pts": np.array(entry_pts, dtype=float),
                             "vmax": SPEED_ENTRY})
            self.segments = segments
            print(f"[algo] plan OK in {time.time()-t0:.2f}s: {len(segments)} segs, "
                  f"entry={geo['entry_gear']}, yaw={math.degrees(geo['yaw']):.0f}, "
                  f"target=({geo['cx']:.1f},{geo['cy']:.1f})")
        else:
            self.segments = self._fallback_plan(start, geo, entry_pts)
            print(f"[algo] plan FALLBACK in {time.time()-t0:.2f}s -> geometric")

        self._assign_speed_profiles(self.segments)
        self.seg_i = 0
        self.seg_pi = 0
        self.done = False
        self.prev_steer = 0.0

    def _smooth_seg(self, pts, gear, iters=60, w_data=0.4, w_smooth=0.25):
        """경로점을 평활화(직선화). 끝점 고정, 충돌나는 이동은 되돌림."""
        n = len(pts)
        if n < 4 or self.env is None:
            return pts
        new = [list(p) for p in pts]
        orig = [list(p) for p in pts]
        for _ in range(iters):
            for i in range(1, n - 1):
                ox, oy = new[i][0], new[i][1]
                for j in (0, 1):
                    new[i][j] += (w_data * (orig[i][j] - new[i][j])
                                  + w_smooth * (new[i - 1][j] + new[i + 1][j] - 2.0 * new[i][j]))
                yaw = math.atan2(new[i + 1][1] - new[i - 1][1], new[i + 1][0] - new[i - 1][0])
                if gear == 'R':
                    yaw = _wrap(yaw + math.pi)
                if self.env.collide(new[i][0], new[i][1], yaw):
                    new[i][0], new[i][1] = ox, oy
        return np.array(new, dtype=float)

    def _assign_speed_profiles(self, segments):
        """각 세그먼트에 점별 속도 프로파일을 부여.
        - 비-마지막 구간: 끝에서 정지(end_v=0, 기어 전환 위해).
        - 마지막(진입) 구간: end_v=vmax 로 끝까지 순항 후 hard-brake 로 정지
          → IoU 유지하며 낮은 final_speed."""
        last = len(segments) - 1
        for k, seg in enumerate(segments):
            vmax = seg.get("vmax", V_STRAIGHT)
            # 마지막(진입) 구간은 중심에서 거의 정지하도록 종단속도를 0 근처로
            end_v = STOP_SPEED if k == last else SEG_LINK_SPEED
            seg["vel"] = _speed_profile(seg["pts"], vmax, end_v)

    def _fallback_plan(self, start, geo, entry_pts):
        sx, sy, _ = start
        cx = geo["cx"]
        pre = geo["pre_rear"]
        aisle_y = pre[1]
        wp = [(sx, sy), (sx, aisle_y), (cx, aisle_y)]
        approach = {"gear": "D", "pts": np.array(_densify(wp, DENSIFY), dtype=float),
                    "vmax": V_STRAIGHT}
        entry = {"gear": geo["entry_gear"], "pts": np.array(entry_pts, dtype=float),
                 "vmax": SPEED_ENTRY}
        return [approach, entry]

    def compute_path(self, obs):
        self._plan(obs)

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        t = float(obs.get("t", 0.0))
        st = obs.get("state", {})
        x = float(st.get("x", 0.0)); y = float(st.get("y", 0.0))
        yaw = float(st.get("yaw", 0.0)); v = float(st.get("v", 0.0))

        target_slot = tuple(map(float, obs.get("target_slot", (0, 0, 0, 0))))
        tcx = 0.5 * (target_slot[0] + target_slot[1])
        tcy = 0.5 * (target_slot[2] + target_slot[3])

        need_plan = False
        if self.segments is None:
            need_plan = True
        elif t < self.prev_t - 0.1:
            need_plan = True
        elif self.plan_target is None or math.hypot(tcx - self.plan_target[0],
                                                    tcy - self.plan_target[1]) > 1.0:
            need_plan = True
        self.prev_t = t

        if need_plan and self.extent is not None:
            self._plan(obs)

        if not self.segments:
            return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
        if self.done:
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        seg = self.segments[self.seg_i]
        gear = seg["gear"]
        pts = seg["pts"]
        self.seg_pi = self._advance_index(pts, x, y, self.seg_pi)
        dist_end = math.hypot(pts[-1][0] - x, pts[-1][1] - y)
        is_last = (self.seg_i == len(self.segments) - 1)
        # 끝점 도착: 거리 기준 또는 경로 인덱스가 끝에 도달(오버슈트 대비)
        arrived = dist_end < SEG_END_TOL or self.seg_pi >= len(pts) - 1

        if arrived:
            if is_last:
                if abs(v) < STOP_SPEED + 0.02:
                    self.done = True
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": gear}
            else:
                if abs(v) > STOP_SPEED:
                    return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": gear}
                self.seg_i += 1
                self.seg_pi = 0
                seg = self.segments[self.seg_i]
                gear = seg["gear"]
                pts = seg["pts"]
                dist_end = math.hypot(pts[-1][0] - x, pts[-1][1] - y)
                is_last = (self.seg_i == len(self.segments) - 1)

        steer = self._pure_pursuit(pts, self.seg_pi, x, y, yaw, v, gear)
        # 저역통과(직전 명령 블렌딩) + 데드존 → 0 부근 진동/반전 제거
        steer = (1.0 - STEER_SMOOTH) * steer + STEER_SMOOTH * self.prev_steer
        self.prev_steer = steer
        if abs(steer) < STEER_DEADZONE:
            steer = 0.0

        # 곡률 인지 속도 프로파일(룩어헤드: 곡선·끝 전에 미리 감속)
        vel = seg.get("vel")
        idx = self.seg_pi if self.seg_pi < len(vel) else len(vel) - 1
        target_mag = vel[idx]

        vmag = abs(v)
        accel = 0.0; brake = 0.0
        moving_wrong = (gear == 'D' and v < -0.05) or (gear == 'R' and v > 0.05)
        if is_last and dist_end < LAST_HARD_BRAKE:
            # 슬롯 깊이 도달 후 한 번에 정지 → IoU 유지·낮은 final_speed
            brake = 1.0
        elif moving_wrong:
            brake = 1.0
        elif vmag < target_mag - 0.06:
            accel = min(1.0, 2.0 * (target_mag - vmag) + 0.2)
        elif vmag > target_mag + 0.10:
            brake = min(1.0, 1.4 * (vmag - target_mag))
        else:
            accel = 0.07

        return {"steer": float(steer), "accel": float(accel),
                "brake": float(brake), "gear": gear}

    def _advance_index(self, pts, x, y, start_i):
        n = len(pts)
        best_i = start_i
        best_d = float("inf")
        lo = max(0, start_i - 2)
        for i in range(lo, n):
            dx = pts[i][0] - x; dy = pts[i][1] - y
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _pure_pursuit(self, pts, i, x, y, yaw, v, gear):
        Ld = max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * abs(v)))
        n = len(pts)
        tx, ty = pts[-1]
        acc = 0.0
        px, py = pts[min(i, n - 1)]
        for j in range(min(i, n - 1) + 1, n):
            nx, ny = pts[j]
            acc += math.hypot(nx - px, ny - py)
            px, py = nx, ny
            if acc >= Ld:
                tx, ty = nx, ny
                break
        ldist = math.hypot(tx - x, ty - y)
        if ldist < 1e-6:
            return 0.0
        if gear == 'D':
            alpha = _wrap(math.atan2(ty - y, tx - x) - yaw)
            delta = math.atan2(2.0 * L_WHEELBASE * math.sin(alpha), ldist)
        else:
            # 후진: 기준 헤딩을 반대(yaw+π)로 두고 조향 부호를 반전한다.
            alpha = _wrap(math.atan2(ty - y, tx - x) - (yaw + math.pi))
            delta = -math.atan2(2.0 * L_WHEELBASE * math.sin(alpha), ldist)
        return max(-MAX_STEER, min(MAX_STEER, delta))


planner = StudentPlanner()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    try:
        planner.set_map(map_payload)
    except Exception as exc:
        print(f"[algo] set_map error: {exc}")


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        import traceback
        print(f"[algo] planner_step error: {exc}")
        traceback.print_exc()
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
