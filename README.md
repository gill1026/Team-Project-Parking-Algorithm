# 🅿️ Team Project — 자율주차 알고리즘 (Rule-Based)

한양대 ERICA · 자료구조와 알고리즘 · Final Project
**주차 시뮬레이터를 분석하고, 규칙 기반(Rule-Based) 자율주차 알고리즘을 설계·구현한 프로젝트입니다.**

- 시뮬레이터: [`self-parking-sim-main/`](self-parking-sim-main) (원본 그대로, 수정 X)
- 우리 알고리즘: [`self-parking-user-algorithms/student_planner.py`](self-parking-user-algorithms/student_planner.py) ← **우리가 구현한 부분**
- 알고리즘 상세 문서: [`self-parking-user-algorithms/ALGORITHM.md`](self-parking-user-algorithms/ALGORITHM.md)
- 설치/실행 가이드: [`SETUP_GUIDE.md`](SETUP_GUIDE.md)

---

## 1. 문제 정의

시뮬레이터가 매 틱 차량 상태를 주면, 우리는 **조향(steer)·가속(accel)·제동(brake)·기어(D/R)** 를 돌려줘서
차를 **목표 주차칸에 정확히·안전하게 주차**시켜야 한다.

- **입력**: 차량 상태 `(x, y, yaw, v)`, 목표 주차칸 `target_slot`, 맵(점유 슬롯·주차선·경계·요구 방향)
- **출력**: `{steer, accel, brake, gear}`
- **목표**: 충돌 없이 + 빠르게 + 슬롯 중앙에 정렬해 완전 정지

---

## 2. 시뮬레이터 분석 🔍

코드를 직접 뜯어 **채점/충돌이 실제로 어떻게 동작하는지**부터 파악했다. (이게 설계의 출발점)

### 2-1. 차량 모델 (자전거 모델)
```
x += v·cos(yaw)·dt ,  y += v·sin(yaw)·dt ,  yaw += (v/L)·tan(steer)·dt
```
- 휠베이스 `L=2.6m`, 최대 조향 `35°` → **최소 회전 반경 R_min ≈ 3.7m** (제자리 회전 불가)
- 차체 약 `3.0m × 1.6m`, **기어 R로 후진 가능**

### 2-2. 충돌 판정 — *직접 분석해서 알아낸 핵심*
시뮬레이터가 실제로 충돌로 치는 것은:

| 켜짐 | 항목 |
|------|------|
| ✅ | (A) 맵 경계 밖 |
| ✅ | (B) **점유된 주차칸**(주차된 차) 사각형 |
| ✅ | (C) **주차선(lane line)** |
| ❌ | (D) 외벽 격자(stationary) — **비활성** |

> ⚠️ 이걸 모르고 처음엔 "외벽 격자"로 충돌 검사를 했다가 **벽·차를 통과하는 버그**를 겪었다.
> 분석 후 **"점유 슬롯 + 주차선"** 으로 바로잡아 해결.
> 또한 슬롯에 **차체가 완전히 들어가면 충돌 면제**(목표칸 내부 허용) 규칙도 확인.

### 2-3. 채점 구조
- **성공 게이트(이걸 못 넘으면 0점)**: `IoU ≥ 0.30` AND `|속도| ≤ 0.2 m/s` AND 주차방향 판별 가능
- 성공 시: 안전점수 50 + (시간·거리·평균속도·조향반전·IoU·방향·정지)의 가중합
- **IoU 천장**: 차(1.6×3.0)를 슬롯(2.2×4.2)에 완벽히 넣어도 **IoU ≈ 0.52가 최대**(기하학적 한계)

### 2-4. 요구 주차 방향 (front_in / rear_in)
- 맵 패킷의 `expected_orientation` 으로 전달됨 (화면 상단 빨간 글씨).
- 이 값은 **"최종 차 머리 방향"** 만 정하고, 실제로 앞으로 넣을지/후진으로 넣을지는 **슬롯 입구 위치에 따라 달라진다**는 점을 분석으로 확인.

---

## 3. 알고리즘 설계 🧠

자율주행 표준 파이프라인(**계획 → 제어**)을 따른다.

```
set_map(맵 1회)  →  충돌 격자 + 휴리스틱(거리) 격자 준비
        │
compute_control(매 틱)
   ├─ _entry_geometry : 목표 자세(yaw)·전/후진 진입·사전진입점 결정
   ├─ hybrid_astar    : 시작 → 진입점 충돌없는 경로 탐색 (+ Dubins shot)
   ├─ _smooth_seg     : 경로 평활화(직선화)
   ├─ 직선 도킹구간    : 진입점 → 슬롯 중심
   ├─ _speed_profile  : 곡률 따라 점별 속도 미리 계산
   └─ Pure Pursuit + 속도제어 → steer/accel/brake/gear
```

### 핵심 알고리즘
| 기법 | 역할 | 왜 |
|------|------|-----|
| **Hybrid A\*** | 시작→진입점 경로 탐색 | 일반 A*와 달리 `(x,y,yaw)`에 회전반경·후진까지 고려 → **차가 실제로 갈 수 있는 경로** |
| **Dubins shot** | 목표 근처에서 한 방에 연결 | 격자로 목표 자세에 정확히 닿기 어려움 → **빠르게+정확히** 마무리 |
| **3원 충돌 근사** | 차체를 원 3개로 검사 | 사각형 SAT보다 **빠르고**(점 조회) 회전에 강함 |
| **Dijkstra 휴리스틱** | 목표까지 "돌아가는 실제 거리" | 직선거리보다 똑똑해 통로를 따라 탐색 |
| **진입 기하 분석** | 입구 방향·전후진·진입점 | front/rear와 슬롯 입구를 보고 **전진/후진 자동 결정** |
| **Pure Pursuit** | 경로 추종(조향) | 단순·강건·부드러움, 차량 모델 불필요 (후진도 지원) |
| **곡률 속도 프로파일** | 점별 목표 속도 | 곡선 전에 미리 감속 → 오버슈트 없이 부드럽게, 슬롯 중앙서 정지 |

> 각 기법의 자세한 원리·이유·그림은 **[ALGORITHM.md](self-parking-user-algorithms/ALGORITHM.md)** 참고.

### 에이전트가 입력을 쓰는 방식
- **맵(1회)** → 충돌 격자 + 거리 휴리스틱 구성
- **목표 슬롯** → 도착 자세(위치+방향) + 진입 방식 결정
- **차량 상태(매 틱)** → 계획의 출발점 + 경로 추종(조향·속도·정지 판단)

---

## 4. 결과 📊

GUI 없이 시뮬레이터와 **동일한 충돌 모델**로 전 슬롯을 자동 테스트(`test_planner.py`):

| 맵 | 요구 방향 | 성공률 | 평균 IoU |
|----|-----------|--------|----------|
| Default Lot | front_in | **25 / 25** | 0.52 (≈최대) |
| Crowded Lot | front_in | **13 / 13** | 0.52 (≈최대) |
| Full House Lot | rear_in | **1 / 1** | 0.53 |

→ **전 슬롯 충돌 없이 주차 성공**, IoU는 기하학적 한계치에 도달. (전진/후진 주차 모두 처리)

---

## 5. 저장소 구조

```
.
├── self-parking-sim-main/              # 시뮬레이터 (원본, 수정 안 함)
│   └── demo_self_parking_sim.py
├── self-parking-user-algorithms/
│   ├── student_planner.py              # ⭐ 우리 알고리즘 (구현부)
│   ├── ALGORITHM.md                    # 알고리즘 상세 설명
│   ├── test_planner.py                 # 헤드리스 성능 테스트(개발용)
│   └── my_agent.py, ipc_client.py      # 통신부 (원본)
├── SETUP_GUIDE.md                      # 설치/실행 가이드
└── README.md
```

## 6. 실행 방법 (요약)

```bash
# 1) 파일 받기 (Code → Download ZIP) 후 압축 해제
# 2) 시뮬레이터 폴더에서 가상환경 + 패키지
cd self-parking-sim-main
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 3) 실행
.\.venv\Scripts\python.exe demo_self_parking_sim.py
#    → GUI 우측 패널에서 self-parking-user-algorithms/my_agent.py 선택 후 시작
```
자세한 절차·문제해결은 **[SETUP_GUIDE.md](SETUP_GUIDE.md)** 참고.

> 알고리즘 수정은 **`self-parking-user-algorithms/student_planner.py`** 한 파일만 건드리면 된다.
