"""자율주차 알고리즘 (학생 구현부).

전체 흐름:
  set_map(payload)      : 맵 1회 수신 → 충돌 격자 + 휴리스틱(거리) 격자 준비
  compute_control(obs)  : 매 틱 호출.
        ├ 필요하면 _plan() 으로 경로를 (재)생성
        │     ① _entry_geometry : 목표 자세/진입방식(전·후진) 결정
        │     ② hybrid_astar   : 시작→진입점 충돌없는 경로 탐색 (+Dubins shot)
        │     ③ _smooth_seg    : 경로 평활화
        │     ④ 진입 직선구간 추가 + 속도 프로파일 부여
        └ 경로를 Pure Pursuit + 속도제어로 따라가 steer/accel/brake/gear 반환

자세한 설명은 같은 폴더의 ALGORITHM.md 참고.
좌표/모델은 시뮬레이터(demo_self_parking_sim.py)와 동일한 자전거 모델을 가정한다.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ----------------- 차량/플래너 파라미터 -----------------
L_WHEELBASE = 2.6                               # 휠베이스(앞축~뒤축 거리) [m]
LF = 1.6                                        # 차체 reference→앞범퍼 [m]
LR = 1.4                                        # 차체 reference→뒷범퍼 [m]
HALF_W = 0.8                                    # 차폭 절반 [m]
MAX_STEER = math.radians(35.0)                  # 최대 조향각 [rad]
R_MIN = L_WHEELBASE / math.tan(MAX_STEER)       # 최소 회전 반경 ≈ 3.71 m (이보다 급히 못 돎)
BODY_OFFSET = (LF - LR) * 0.5                   # 0.1: 뒷축원점→차체중심(노즈 방향) 보정

# 차체 충돌 근사: 사각형 대신 "원 3개"로 덮는다(빠르고 보수적).
# 차체 좌표계(뒷축 원점, 노즈=+x) 기준 원 중심들의 x오프셋과 공통 반경.
CIRCLE_XC = (-0.9, 0.1, 1.1)
CIRCLE_R = 0.95

# --- Hybrid A* 탐색 파라미터 ---
DS = 0.7                                        # 한 번 확장 시 진행 거리 [m]
SUBSTEPS = 3                                    # DS 를 잘게 나눠 충돌검사하는 횟수
STEER_SET = (-1.0, -0.5, 0.0, 0.5, 1.0)         # 조향 후보(×MAX_STEER): 좌최대~우최대
XY_RES = 0.5                                    # 방문 격자(closed set) 위치 해상도 [m]
YAW_BINS = 24                                   # 방문 격자 방향 분해능(15°)
REVERSE_PENALTY = 1.6                           # 후진 1m 비용 배율(전진 선호)
GEAR_SWITCH_PENALTY = 6.0                       # 전↔후진 전환 1회 비용(전환 최소화)
STEER_PENALTY = 0.6                             # 조향 변화 비용(부드러운 경로 유도)
GOAL_XY_TOL = 0.6                               # 목표 도달 위치 허용 [m]
GOAL_YAW_TOL = math.radians(18.0)               # 목표 도달 방향 허용 [rad]
# 종료는 결정론적인 확장 횟수로 제어한다(머신 부하와 무관 → 재현성 보장).
# 실측 최악(stage3) ~5000 확장 → 12000 캡은 2배 이상 여유이며, 모든 맵이 캡에
# 닿기 전에 목표를 찾는다. 벽시계는 예기치 못한 행(hang) 방지용 안전장치로만 둔다.
MAX_EXPANSIONS = 12000
PLAN_TIME_SAFETY = 3.0                          # 계획 최대 벽시계 시간 [s] (안전장치)

# --- 충돌/휴리스틱 격자 ---
COLL_RES = 0.2                                  # 충돌 격자 해상도 [m] (정밀)
OCC_BASE = 0.22   # 점유 슬롯 추가 여유(추적오차 흡수 → dense 맵 충돌 방지)
LINE_BASE = 0.05  # 주차선 추가 여유(작게: 통로가 막히지 않도록)
LINE_HALF_WIDTH = 0.25                          # 주차선을 사각형으로 만들 때 두께 절반(시뮬레이터 동일)
HEUR_RES = 0.5                                  # 휴리스틱 거리 격자 해상도 [m] (성김=빠름)
HEUR_BASE = 0.10                                # 휴리스틱 격자 장애물 여유

# --- Dubins shot(해석적 확장) ---
SHOT_RANGE = 14.0     # 목표까지 이 거리 이내일 때만 시도
SHOT_THROTTLE = 2     # 매 확장마다 하지 않고 N번에 1번만 시도(비용 절감)
SHOT_MAX_WORDS = 2    # 최단 단어 몇 개만 검사(실패 비용 절감)

ENTRY_LEN = 3.4       # (참고용) 진입 직선 길이 기준

# --- 제어(Controller) 파라미터 ---
LD_MIN = 1.3          # Pure Pursuit 전방주시거리 최소 [m]
LD_MAX = 3.0          # 〃 최대
LD_BASE = 1.0         # 〃 기본값 (Ld = BASE + GAIN·속도)
LD_GAIN = 0.7         # 〃 속도 비례 계수
STEER_SMOOTH = 0.35   # 직전 조향과 블렌딩(저역통과) → 진동·반전 감소
# 곡률 인지 속도 프로파일
V_STRAIGHT = 2.9      # 긴 직선 전진 상한 [m/s]
SPEED_REV = 1.2       # 후진 구간 상한
SPEED_ENTRY = 0.60    # 슬롯 직선 진입 상한(낮을수록 중심에서 더 느리게 정지)
V_CURVE_MIN = 1.1     # 곡선 최저 속도
A_LAT = 0.42          # 허용 횡가속 → 곡률→속도 (R_min에서 ≈1.25 m/s)
A_DECEL = 2.5         # 프로파일 감속도(여유: maxBrake 7)
A_ACCEL = 1.9         # 프로파일 가속 램프(참고)
SEG_END_TOL = 0.30    # 세그먼트 끝점 도달 판정 거리 [m]
STOP_SPEED = 0.06     # 정지로 간주하는 속도 [m/s]
SEG_LINK_SPEED = 0.5   # 비-마지막 구간 끝 도착 속도(0 으로 기면 끝점에서 맴돌다 정체)
LAST_HARD_BRAKE = 0.5  # 마지막 구간: 끝 0.5m 전부터 강제 제동(IoU 유지·저속 정지)
STEER_DEADZONE = math.radians(3.0)  # 미세 진동 제거 → 조향 반전 횟수 감소
DENSIFY = 0.2         # 경로점 촘촘히 다시 찍는 간격 [m]


LAST_EXPANSIONS = 0   # (디버그용) 마지막 계획의 탐색 노드 수


def _wrap(a: float) -> float:
    """각도를 (-π, π] 범위로 정규화."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _mod2pi(a: float) -> float:
    """각도를 [0, 2π) 범위로 정규화 (Dubins 계산용)."""
    return a - 2.0 * math.pi * math.floor(a / (2.0 * math.pi))


def pretty_print_map_summary(map_payload: Dict[str, Any]) -> None:
    """맵 수신 시 콘솔에 요약 출력(디버그용)."""
    extent = map_payload.get("extent") or [None, None, None, None]
    slots = map_payload.get("slots") or []
    occupied = map_payload.get("occupied_idx") or []
    free_slots = len(slots) - sum(1 for v in occupied if v)
    print("[algo] map extent :", extent)
    print("[algo] total slots:", len(slots), "/ free:", free_slots)


# ----------------- Dubins 곡선 -----------------
# Dubins: "전진만 하는 차"가 두 자세(x,y,yaw) 사이를 최소회전반경으로 잇는 최단 경로.
# 6가지 패턴(LSL/RSR/LSR/RSL/RLR/LRL) 중 가장 짧은 것을 고른다.
# L=좌회전, R=우회전, S=직진. 여기서는 정규화 거리(d=거리/R)로 계산한다.
def _dubins_words(alpha, beta, d):
    """6개 Dubins 패턴의 (총길이, 세그먼트길이들, 모드들)을 길이순으로 반환."""
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

    out.sort(key=lambda w: w[0])   # 짧은 경로 우선
    return out


def _dubins_sample(start, lengths, modes, R, step=0.25):
    """Dubins 경로(세그먼트들)를 따라 (x,y,yaw)를 step 간격으로 샘플링."""
    x, y, yaw = start
    pts = [(x, y, yaw)]
    for seg_len_norm, mode in zip(lengths, modes):
        seg_len = seg_len_norm * R     # 정규화 길이 → 실제 길이
        n = max(1, int(math.ceil(seg_len / step)))
        ds = seg_len / n
        if mode == "S":                # 직진
            for _ in range(n):
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
                pts.append((x, y, yaw))
        else:                          # 좌/우 회전
            curv = 1.0 / R if mode == "L" else -1.0 / R
            for _ in range(n):
                yaw = _wrap(yaw + curv * ds)
                x += ds * math.cos(yaw)
                y += ds * math.sin(yaw)
                pts.append((x, y, yaw))
    return pts


def dubins_shot(start, goal, R, env):
    """start→goal 을 잇는 충돌 없는 최단 Dubins(전진) 경로를 반환, 없으면 None.
    Hybrid A* 가 목표 근처에서 '한 방에' 목표 자세로 연결하는 용도."""
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
        for (x, y, yaw) in pts:        # 경로 전체가 충돌 없어야 채택
            if env.collide(x, y, yaw):
                ok = False
                break
        if ok:
            return pts
    return None


# ----------------- 충돌 환경 -----------------
class CollisionEnv:
    """점유 슬롯 + 주차 라인 + 경계로 충돌/휴리스틱 그리드를 구성.
    차체는 3원 근사로 검사(보수적).

    - cgrid : 정밀 충돌 격자 (collide() 가 매 확장마다 조회)
    - hblock/heuristic_map : 목표까지 '장애물을 돌아가는 실제 거리'를 주는 휴리스틱
    """

    def __init__(self, extent, occ_rects, line_rects):
        self.xmin, self.xmax, self.ymin, self.ymax = extent

        # (1) 정밀 충돌 격자: 점유 슬롯/주차선을 차체반경만큼 부풀려 칠한다.
        self.cres = COLL_RES
        self.cnx = max(1, int((self.xmax - self.xmin) / self.cres) + 1)
        self.cny = max(1, int((self.ymax - self.ymin) / self.cres) + 1)
        self.cgrid = np.zeros((self.cny, self.cnx), dtype=bool)
        for r in occ_rects:
            self._mark(self.cgrid, self.cres, r, CIRCLE_R + OCC_BASE)
        for r in line_rects:
            self._mark(self.cgrid, self.cres, r, CIRCLE_R + LINE_BASE)

        # (2) 휴리스틱용 성긴 격자: 장애물만 표시(거리계산은 heuristic_map 에서).
        self.hres = HEUR_RES
        self.hnx = max(1, int((self.xmax - self.xmin) / self.hres) + 1)
        self.hny = max(1, int((self.ymax - self.ymin) / self.hres) + 1)
        self.hblock = np.zeros((self.hny, self.hnx), dtype=bool)
        for r in occ_rects:
            self._mark(self.hblock, self.hres, r, CIRCLE_R + HEUR_BASE)
        for r in line_rects:
            self._mark(self.hblock, self.hres, r, CIRCLE_R + HEUR_BASE)

    def _mark(self, grid, res, rect, infl):
        """사각형 rect 를 infl 만큼 부풀려 격자 셀들을 True(장애물)로 칠한다."""
        x0, x1, y0, y1 = rect
        c0 = max(0, int((x0 - infl - self.xmin) / res))
        c1 = min(grid.shape[1] - 1, int(math.ceil((x1 + infl - self.xmin) / res)))
        r0 = max(0, int((y0 - infl - self.ymin) / res))
        r1 = min(grid.shape[0] - 1, int(math.ceil((y1 + infl - self.ymin) / res)))
        if c1 >= c0 and r1 >= r0:
            grid[r0:r1 + 1, c0:c1 + 1] = True

    def collide(self, x, y, yaw) -> bool:
        """차체(원 3개)가 경계를 벗어나거나 장애물 격자와 겹치면 True."""
        c = math.cos(yaw); s = math.sin(yaw)
        xmin = self.xmin; ymin = self.ymin
        # 원 중심이 경계에서 반경 이내로 들어오면 경계 충돌
        bxmax = self.xmax - CIRCLE_R; bymax = self.ymax - CIRCLE_R
        bxmin = xmin + CIRCLE_R; bymin = ymin + CIRCLE_R
        cres = self.cres; grid = self.cgrid
        cnx = self.cnx; cny = self.cny
        for xc in CIRCLE_XC:               # 앞/중/뒤 3원
            wx = x + c * xc                # 원 중심의 월드 좌표
            wy = y + s * xc
            if wx < bxmin or wx > bxmax or wy < bymin or wy > bymax:
                return True                # 경계 충돌
            col = int((wx - xmin) / cres)
            row = int((wy - ymin) / cres)
            if 0 <= row < cny and 0 <= col < cnx and grid[row, col]:
                return True                # 장애물 충돌
        return False

    def heuristic_map(self, goal_xy) -> np.ndarray:
        """목표 셀에서 시작하는 격자 Dijkstra 로 '장애물을 돌아가는 최단거리' 맵 생성.
        → A* 휴리스틱으로 쓰면 직선거리보다 똑똑해서 통로를 따라 탐색한다."""
        gx, gy = goal_xy
        gc = min(max(int((gx - self.xmin) / self.hres), 0), self.hnx - 1)
        gr = min(max(int((gy - self.ymin) / self.hres), 0), self.hny - 1)
        dist = np.full((self.hny, self.hnx), np.inf, dtype=np.float64)
        # 목표가 장애물 셀이면 가장 가까운 빈 셀로 이동
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
        """위치 (x,y)의 휴리스틱 값 = max(격자 Dijkstra 거리, 직선거리)."""
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
    """start/goal=(x,y,yaw). 성공 시 (x,y,yaw,gear) 리스트(시작→목표) 반환, 실패 None.

    일반 A*와 달리 상태가 (x,y,yaw)이고, 이웃은 '실제 차가 갈 수 있는 짧은 호'.
    매 노드에서 (전진/후진)×(조향5단계)=10개로 뻗고, Dubins shot 으로 목표에 직결도 시도."""
    sx, sy, syaw = start
    gx, gy, gyaw = goal
    dist_map = env.heuristic_map((gx, gy))         # 휴리스틱 거리 맵(1회)
    yaw_step = 2.0 * math.pi / YAW_BINS

    def key(x, y, yaw):
        """방문 격자 키(같은 칸·같은 방향이면 같은 상태로 취급)."""
        return (int(round(x / XY_RES)), int(round(y / XY_RES)),
                int(round(_wrap(yaw) / yaw_step)) % YAW_BINS)

    # 노드 정보를 병렬 리스트로 보관(메모리/속도). 인덱스 i 가 노드 식별자.
    nx_ = [sx]; ny_ = [sy]; nyaw = [syaw]; ngear = ['D']; nsteer = [0.0]
    nparent = [-1]; ng = [0.0]                     # 부모 인덱스, 누적비용 g
    nshot = [None]   # 이 노드에 붙은 dubins 꼬리(목표까지). 있으면 그 노드가 목표 직전.
    h0 = env.heur_lookup(dist_map, sx, sy, (gx, gy))
    pq = [(h0, 0)]                                 # 우선순위큐: (f=g+h, 노드인덱스)
    best_g = {key(sx, sy, syaw): 0.0}              # 상태별 최선 g

    expansions = 0
    found = -1
    ss = DS / SUBSTEPS                             # 충돌검사용 미세 전진거리

    while pq:
        expansions += 1
        if expansions > MAX_EXPANSIONS:            # 결정론적 종료(재현성)
            break
        # 벽시계는 안전장치로만(정상 케이스에선 절대 발동하지 않음 → 결정론 유지)
        if t_deadline is not None and expansions % 4000 == 0 and time.time() > t_deadline:
            break
        _, i = heapq.heappop(pq)
        cx, cy, cyaw, cgear, csteer, cg = nx_[i], ny_[i], nyaw[i], ngear[i], nsteer[i], ng[i]

        # 목표 도달?
        dpos = math.hypot(cx - gx, cy - gy)
        if dpos <= GOAL_XY_TOL and abs(_wrap(cyaw - gyaw)) <= GOAL_YAW_TOL:
            found = i
            break

        # Dubins 해석적 확장: 목표 근처면 곧장 연결 시도(탐색 가속 + 깔끔한 도착)
        if dpos <= SHOT_RANGE and expansions % SHOT_THROTTLE == 0:
            tail = dubins_shot((cx, cy, cyaw), goal, R_MIN, env)
            if tail is not None:
                nshot[i] = tail
                found = i
                break

        # 10개 모션 프리미티브로 이웃 생성
        for gear in ('D', 'R'):
            direction = 1.0 if gear == 'D' else -1.0   # 전진 +1 / 후진 -1
            for st in STEER_SET:
                steer = st * MAX_STEER
                x, y, yaw = cx, cy, cyaw
                ok = True
                # 자전거 모델로 DS 만큼 전진하며 중간중간 충돌검사
                for _ in range(SUBSTEPS):
                    yaw = _wrap(yaw + direction * ss / L_WHEELBASE * math.tan(steer))
                    x += direction * ss * math.cos(yaw)
                    y += direction * ss * math.sin(yaw)
                    if env.collide(x, y, yaw):
                        ok = False
                        break
                if not ok:
                    continue
                # 비용: 거리(+후진배율) + 기어전환 + 조향변화 (채점 항목과 연동)
                add = DS * (REVERSE_PENALTY if gear == 'R' else 1.0)
                if gear != cgear:
                    add += GEAR_SWITCH_PENALTY
                add += STEER_PENALTY * abs(steer - csteer)
                ngc = cg + add
                k = key(x, y, yaw)
                if k in best_g and best_g[k] <= ngc:   # 이미 더 싸게 방문한 상태면 skip
                    continue
                best_g[k] = ngc
                ni = len(nx_)                          # 새 노드 등록
                nx_.append(x); ny_.append(y); nyaw.append(yaw)
                ngear.append(gear); nsteer.append(steer)
                nparent.append(i); ng.append(ngc); nshot.append(None)
                f = ngc + env.heur_lookup(dist_map, x, y, (gx, gy))
                heapq.heappush(pq, (f, ni))

    global LAST_EXPANSIONS
    LAST_EXPANSIONS = expansions
    if found < 0:
        return None                                # 실패

    # 부모 포인터를 거꾸로 따라가 경로 복원(시작→목표 순서)
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
    """경로를 '같은 기어(전진/후진)' 구간들로 쪼갠다. 구간 사이 = 방향전환(정지 필요)."""
    segs = []
    if not path:
        return segs
    cur_gear = path[0][3]
    pts = [(path[0][0], path[0][1])]
    for (x, y, yaw, gear) in path[1:]:
        if gear != cur_gear:               # 기어 바뀌면 구간 끊기
            segs.append({"gear": cur_gear, "pts": pts})
            pts = [pts[-1]]                # 다음 구간은 이어지는 점부터
            cur_gear = gear
        pts.append((x, y))
    segs.append({"gear": cur_gear, "pts": pts})
    # 너무 짧은 구간은 버리고, 점을 촘촘히 채워(densify) 추종 안정화
    out = []
    for s in segs:
        dp = _densify(s["pts"], DENSIFY)
        if len(dp) >= 2 and _polyline_len(dp) > 0.25:
            out.append({"gear": s["gear"], "pts": np.array(dp, dtype=float),
                        "vmax": V_STRAIGHT if s["gear"] == 'D' else SPEED_REV})
    return out


def _densify(pts, step):
    """점들 사이를 step 간격으로 보간해 촘촘한 점열로."""
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
    """점열의 총 길이."""
    return sum(math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
               for i in range(1, len(pts)))


def _curve_speed(kappa, vmax):
    """곡률 kappa 에서의 허용 속도 = √(허용횡가속/곡률), 단 [V_CURVE_MIN, vmax]."""
    if kappa < 1e-4:
        return vmax
    return max(V_CURVE_MIN, min(vmax, math.sqrt(A_LAT / kappa)))


def _speed_profile(pts, vmax, end_v):
    """곡률·감속(backward) 패스로 점별 속도 한계 배열을 만든다.
    lookahead 감속이라 곡선/구간끝 전에 미리 줄어 오버슈트가 없다(start_v=0: 구간
    시작은 항상 정지 상태)."""
    n = len(pts)
    if n == 1:
        return [0.0]
    # 각 점 사이 거리
    ds = [0.0] * n
    for i in range(1, n):
        ds[i] = math.hypot(pts[i][0] - pts[i - 1][0], pts[i][1] - pts[i - 1][1])
    # 각 점의 곡률(앞뒤 방향 변화 / 거리)
    kap = [0.0] * n
    for i in range(1, n - 1):
        a1 = math.atan2(pts[i][1] - pts[i - 1][1], pts[i][0] - pts[i - 1][0])
        a2 = math.atan2(pts[i + 1][1] - pts[i][1], pts[i + 1][0] - pts[i][0])
        dl = 0.5 * (ds[i] + ds[i + 1])
        kap[i] = abs(_wrap(a2 - a1)) / dl if dl > 1e-6 else 0.0
    if n > 2:
        kap[0] = kap[1]; kap[-1] = kap[-2]
    v = [_curve_speed(kap[i], vmax) for i in range(n)]
    v[-1] = min(v[-1], end_v)                  # 구간 끝 도착 속도
    # backward pass: 곡선·구간끝 전에 미리 감속(룩어헤드). 가속 램프는 제어의
    # 물리 가속이 담당하므로 forward pass 는 두지 않는다(시작점 0 → 정지 버그 방지).
    for i in range(n - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2.0 * A_DECEL * ds[i + 1]))
    return v


# ----------------- 플래너 -----------------
@dataclass
class StudentPlanner:
    # --- 맵에서 얻은 정보 ---
    map_data: Optional[Dict[str, Any]] = None
    extent: Optional[Tuple[float, float, float, float]] = None
    occ_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)   # 점유 슬롯 사각형
    line_rects: List[Tuple[float, float, float, float]] = field(default_factory=list)  # 주차선 사각형
    lines_raw: Any = None                       # 원본 라인 세그먼트(진입 기하 판단용)
    expected_orientation: Optional[str] = None  # "front_in" / "rear_in"
    env: Optional[CollisionEnv] = None          # 충돌/휴리스틱 환경(지연 생성)

    # --- 계획/추종 상태 ---
    segments: Optional[List[Dict[str, Any]]] = None   # 현재 경로(구간 리스트)
    seg_i: int = 0          # 현재 따라가는 구간 인덱스
    seg_pi: int = 0         # 구간 내 현재(최근접) 점 인덱스
    done: bool = False      # 주차 완료(정지 유지)
    plan_target: Optional[Tuple[float, float]] = None  # 마지막으로 계획한 목표 중심
    prev_t: float = 1e9     # 직전 obs 시간(라운드 리셋 감지용)
    prev_steer: float = 0.0 # 직전 조향(저역통과 필터용)

    def set_map(self, map_payload: Dict[str, Any]) -> None:
        """맵 1회 수신: 장애물(점유슬롯/주차선)과 요구 방향을 저장하고 경로 상태 초기화."""
        self.map_data = map_payload
        self.extent = tuple(map(float, map_payload.get("extent", (0, 0, 0, 0))))
        slots = map_payload.get("slots") or []
        occ = map_payload.get("occupied_idx") or []
        # 점유된 슬롯만 장애물 사각형으로
        self.occ_rects = [tuple(map(float, slots[i]))
                          for i in range(len(slots)) if i < len(occ) and occ[i]]
        self.lines_raw = map_payload.get("lines") or []
        self.line_rects = self._compute_line_rects(self.lines_raw)
        self.expected_orientation = map_payload.get("expected_orientation")
        self.env = None                         # 다음 계획 때 생성
        pretty_print_map_summary(map_payload)
        self._reset_path()

    def _reset_path(self):
        """경로/추종 상태 리셋."""
        self.segments = None
        self.seg_i = 0
        self.seg_pi = 0
        self.done = False
        self.plan_target = None

    def _compute_line_rects(self, lines):
        """주차선 세그먼트 [x1,y1,x2,y2] 를 두께(LINE_HALF_WIDTH) 가진 사각형으로 변환.
        (시뮬레이터의 주차선 충돌과 동일하게 맞춤)."""
        if self.extent is None:
            return []
        xmin, xmax, ymin, ymax = self.extent
        hw = LINE_HALF_WIDTH
        out = []
        for ln in lines:
            x1, y1, x2, y2 = map(float, ln)
            if abs(x1 - x2) < 1e-6:            # 수직선
                xa, xb = min(x1, x2) - hw, max(x1, x2) + hw
                ya, yb = min(y1, y2), max(y1, y2)
            elif abs(y1 - y2) < 1e-6:          # 수평선
                xa, xb = min(x1, x2), max(x1, x2)
                ya, yb = min(y1, y2) - hw, max(y1, y2) + hw
            else:                              # 대각(드묾)
                xa, xb = min(x1, x2) - hw, max(x1, x2) + hw
                ya, yb = min(y1, y2) - hw, max(y1, y2) + hw
            xa = max(xa, xmin); xb = min(xb, xmax)
            ya = max(ya, ymin); yb = min(yb, ymax)
            if xb > xa and yb > ya:
                out.append((xa, xb, ya, yb))
        return out

    def _entry_geometry(self, target_slot):
        """목표 슬롯으로부터 '도착 자세 + 진입 방식'을 계산.
        반환: 중심(cx,cy), 최종 yaw, 머리방향 hy, 진입부호 ty, 기어(D/R),
              뒷축 목표(goal_rear), 사전진입점(pre_rear)."""
        x0, x1, y0, y1 = target_slot
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        xmin, xmax, ymin, ymax = self.extent

        # (1) 슬롯이 어느 쪽(위/아래)으로 열려 있나? = 주차선이 '없는' 쪽이 입구.
        low_blocked = self._horiz_line_near(cx, cy, side="low")
        high_blocked = self._horiz_line_near(cx, cy, side="high")
        if low_blocked and not high_blocked:
            open_high = True
        elif high_blocked and not low_blocked:
            open_high = False
        elif not low_blocked and not high_blocked:
            open_high = (cy - ymin) <= (ymax - cy)   # 둘 다 안 막히면 더 넓은 쪽
        else:
            open_high = True

        ty = -1.0 if open_high else +1.0   # 슬롯 안으로 들어가는 진행 부호(+y/-y)

        # (2) 최종 차 머리 방향: 요구사항(front_in/rear_in)이 결정.
        if self.expected_orientation == "front_in":
            hy = +1.0                       # 코가 +y(위)
        elif self.expected_orientation == "rear_in":
            hy = -1.0                       # 코가 -y(아래)
        else:
            hy = ty
        yaw = math.pi / 2.0 if hy > 0 else -math.pi / 2.0
        # (3) 머리방향과 진입방향이 같으면 전진, 다르면 후진으로 들어가야 함.
        entry_gear = 'D' if hy * ty > 0 else 'R'
        # 뒷축 기준점 목표(차체중심이 슬롯중심에 오도록 BODY_OFFSET 보정)
        goal_rear = (cx, cy - BODY_OFFSET * hy)

        # (4) 사전진입(pre-entry): 열린 변 쪽 인접 통로의 "중앙"에 배치한다.
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
            if abs(y1 - y2) > 1e-6:           # 수평선만
                continue
            ly = y1
            if not (min(x1, x2) - 0.3 <= cx <= max(x1, x2) + 0.3):  # cx 위를 지나는 선만
                continue
            if open_high and ly > open_edge + 0.4:
                best = min(best, ly)
            elif (not open_high) and ly < open_edge - 0.4:
                best = max(best, ly)
        return best

    def _horiz_line_near(self, cx, cy, side):
        """슬롯 중심 위/아래 가까이에 수평 주차선(=막힌 변)이 있는지."""
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
        """경로 생성: 진입 기하 → Hybrid A*(시작→사전진입점) → 스무딩 → 진입직선 → 속도프로파일."""
        st = obs.get("state", {})
        start = (float(st.get("x", 0.0)), float(st.get("y", 0.0)),
                 float(st.get("yaw", 0.0)))
        target_slot = tuple(map(float, obs.get("target_slot", (0, 0, 0, 0))))
        geo = self._entry_geometry(target_slot)
        self.plan_target = (geo["cx"], geo["cy"])

        if self.env is None:                          # 충돌환경 1회 생성(맵 바뀌면 재생성)
            self.env = CollisionEnv(self.extent, self.occ_rects, self.line_rects)
        env = self.env

        # 1) 시작 → 사전진입점(pre_rear) 까지 Hybrid A*
        goal_pose = (geo["pre_rear"][0], geo["pre_rear"][1], geo["yaw"])
        t0 = time.time()
        path = hybrid_astar(env, start, goal_pose, time.time() + PLAN_TIME_SAFETY)

        # 2) 진입 직선구간(사전진입점 → 슬롯중심)
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
            # Hybrid A* 실패 시: 기하학적 ㄱ자 경로로 폴백
            self.segments = self._fallback_plan(start, geo, entry_pts)
            print(f"[algo] plan FALLBACK in {time.time()-t0:.2f}s -> geometric")

        self._assign_speed_profiles(self.segments)
        self.seg_i = 0
        self.seg_pi = 0
        self.done = False
        self.prev_steer = 0.0

    def _smooth_seg(self, pts, gear, iters=60, w_data=0.4, w_smooth=0.25):
        """경로점을 평활화(직선화). 끝점 고정, 충돌나는 이동은 되돌림.
        w_data: 원본 유지, w_smooth: 이웃 평균으로 당김(직선화)."""
        n = len(pts)
        if n < 4 or self.env is None:
            return pts
        new = [list(p) for p in pts]
        orig = [list(p) for p in pts]
        for _ in range(iters):
            for i in range(1, n - 1):              # 양 끝점은 고정
                ox, oy = new[i][0], new[i][1]
                for j in (0, 1):
                    new[i][j] += (w_data * (orig[i][j] - new[i][j])
                                  + w_smooth * (new[i - 1][j] + new[i + 1][j] - 2.0 * new[i][j]))
                # 이동 후 충돌하면 원위치로(안전 보장)
                yaw = math.atan2(new[i + 1][1] - new[i - 1][1], new[i + 1][0] - new[i - 1][0])
                if gear == 'R':
                    yaw = _wrap(yaw + math.pi)
                if self.env.collide(new[i][0], new[i][1], yaw):
                    new[i][0], new[i][1] = ox, oy
        return np.array(new, dtype=float)

    def _assign_speed_profiles(self, segments):
        """각 세그먼트에 점별 속도 프로파일을 부여.
        - 비-마지막 구간: 끝 도착속도 SEG_LINK_SPEED (기어 전환 위해 거의 정지).
        - 마지막(진입) 구간: 종단속도 STOP_SPEED → 슬롯 중심에서 거의 정지(정지 점수↑)."""
        last = len(segments) - 1
        for k, seg in enumerate(segments):
            vmax = seg.get("vmax", V_STRAIGHT)
            # 마지막(진입) 구간은 중심에서 거의 정지하도록 종단속도를 0 근처로
            end_v = STOP_SPEED if k == last else SEG_LINK_SPEED
            seg["vel"] = _speed_profile(seg["pts"], vmax, end_v)

    def _fallback_plan(self, start, geo, entry_pts):
        """Hybrid A* 가 실패했을 때의 단순 기하 경로: 직진→통로→슬롯 앞."""
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
        """(외부에서 명시 호출용) 경로 재생성."""
        self._plan(obs)

    def compute_control(self, obs: Dict[str, Any]) -> Dict[str, float]:
        """매 틱 호출. 필요 시 재계획 후, 현재 구간을 추종해 제어값 반환."""
        t = float(obs.get("t", 0.0))
        st = obs.get("state", {})
        x = float(st.get("x", 0.0)); y = float(st.get("y", 0.0))
        yaw = float(st.get("yaw", 0.0)); v = float(st.get("v", 0.0))

        target_slot = tuple(map(float, obs.get("target_slot", (0, 0, 0, 0))))
        tcx = 0.5 * (target_slot[0] + target_slot[1])
        tcy = 0.5 * (target_slot[2] + target_slot[3])

        # --- 재계획이 필요한가? ---
        need_plan = False
        if self.segments is None:                       # 아직 경로 없음
            need_plan = True
        elif t < self.prev_t - 0.1:                     # 시간이 되감김 = 새 라운드
            need_plan = True
        elif self.plan_target is None or math.hypot(tcx - self.plan_target[0],
                                                    tcy - self.plan_target[1]) > 1.0:
            need_plan = True                            # 목표가 바뀜
        self.prev_t = t

        if need_plan and self.extent is not None:
            self._plan(obs)

        if not self.segments:                           # 경로 못 만들면 정지
            return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
        if self.done:                                   # 주차 완료 → 계속 정지
            return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": "D"}

        seg = self.segments[self.seg_i]
        gear = seg["gear"]
        pts = seg["pts"]
        self.seg_pi = self._advance_index(pts, x, y, self.seg_pi)   # 최근접 점 갱신
        dist_end = math.hypot(pts[-1][0] - x, pts[-1][1] - y)
        is_last = (self.seg_i == len(self.segments) - 1)
        # 끝점 도착: 거리 기준 또는 경로 인덱스가 끝에 도달(오버슈트 대비)
        arrived = dist_end < SEG_END_TOL or self.seg_pi >= len(pts) - 1

        if arrived:
            if is_last:                                 # 마지막 구간 끝 = 주차 완료 처리
                if abs(v) < STOP_SPEED + 0.02:
                    self.done = True
                return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": gear}
            else:                                       # 중간 구간 끝 = 멈춘 뒤 다음 구간으로
                if abs(v) > STOP_SPEED:
                    return {"steer": 0.0, "accel": 0.0, "brake": 1.0, "gear": gear}
                self.seg_i += 1
                self.seg_pi = 0
                seg = self.segments[self.seg_i]
                gear = seg["gear"]
                pts = seg["pts"]
                dist_end = math.hypot(pts[-1][0] - x, pts[-1][1] - y)
                is_last = (self.seg_i == len(self.segments) - 1)

        # --- 조향: Pure Pursuit + 저역통과 + 데드존 ---
        steer = self._pure_pursuit(pts, self.seg_pi, x, y, yaw, v, gear)
        # 저역통과(직전 명령 블렌딩) + 데드존 → 0 부근 진동/반전 제거
        steer = (1.0 - STEER_SMOOTH) * steer + STEER_SMOOTH * self.prev_steer
        self.prev_steer = steer
        if abs(steer) < STEER_DEADZONE:
            steer = 0.0

        # --- 속도: 곡률 인지 프로파일 목표값에 맞춰 가/감속 ---
        vel = seg.get("vel")
        idx = self.seg_pi if self.seg_pi < len(vel) else len(vel) - 1
        target_mag = vel[idx]                           # 이 점에서 원하는 속도 크기

        vmag = abs(v)
        accel = 0.0; brake = 0.0
        moving_wrong = (gear == 'D' and v < -0.05) or (gear == 'R' and v > 0.05)
        if is_last and dist_end < LAST_HARD_BRAKE:
            # 슬롯 깊이 도달 후 한 번에 정지 → IoU 유지·낮은 final_speed
            brake = 1.0
        elif moving_wrong:                              # 기어와 반대로 굴러가면 제동
            brake = 1.0
        elif vmag < target_mag - 0.06:                  # 느리면 가속
            accel = min(1.0, 2.0 * (target_mag - vmag) + 0.2)
        elif vmag > target_mag + 0.10:                  # 빠르면 제동
            brake = min(1.0, 1.4 * (vmag - target_mag))
        else:                                           # 유지(살짝 가속)
            accel = 0.07

        return {"steer": float(steer), "accel": float(accel),
                "brake": float(brake), "gear": gear}

    def _advance_index(self, pts, x, y, start_i):
        """현재 위치에서 경로상 가장 가까운 점 인덱스(앞으로만 진행)."""
        n = len(pts)
        best_i = start_i
        best_d = float("inf")
        lo = max(0, start_i - 2)               # 약간 뒤까지만 보고(되돌아가지 않게)
        for i in range(lo, n):
            dx = pts[i][0] - x; dy = pts[i][1] - y
            d = dx * dx + dy * dy
            if d < best_d:
                best_d = d
                best_i = i
        return best_i

    def _pure_pursuit(self, pts, i, x, y, yaw, v, gear):
        """전방주시점을 향하도록 조향각 계산(Pure Pursuit). 후진이면 기준헤딩 반전."""
        Ld = max(LD_MIN, min(LD_MAX, LD_BASE + LD_GAIN * abs(v)))   # 속도 비례 주시거리
        n = len(pts)
        # 현재 점에서 경로를 따라 Ld 만큼 앞의 점을 목표로
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
        if gear == 'D':                        # 전진: 표준 Pure Pursuit
            alpha = _wrap(math.atan2(ty - y, tx - x) - yaw)
            delta = math.atan2(2.0 * L_WHEELBASE * math.sin(alpha), ldist)
        else:
            # 후진: 기준 헤딩을 반대(yaw+π)로 두고 조향 부호를 반전한다.
            alpha = _wrap(math.atan2(ty - y, tx - x) - (yaw + math.pi))
            delta = -math.atan2(2.0 * L_WHEELBASE * math.sin(alpha), ldist)
        return max(-MAX_STEER, min(MAX_STEER, delta))


# ----------------- 통신 모듈 인터페이스 -----------------
# ipc_client.py 가 아래 두 함수를 호출한다(이 부분은 건드리지 않음).
planner = StudentPlanner()


def handle_map_payload(map_payload: Dict[str, Any]) -> None:
    """맵 패킷 수신 시 호출."""
    try:
        planner.set_map(map_payload)
    except Exception as exc:
        print(f"[algo] set_map error: {exc}")


def planner_step(obs: Dict[str, Any]) -> Dict[str, Any]:
    """매 스텝 관측 수신 시 호출 → 제어 명령 반환."""
    try:
        return planner.compute_control(obs)
    except Exception as exc:
        import traceback
        print(f"[algo] planner_step error: {exc}")
        traceback.print_exc()
        return {"steer": 0.0, "accel": 0.0, "brake": 0.5, "gear": "D"}
