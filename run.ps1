# ============================================================
#  Self-Parking 시뮬레이터 빠른 실행 런처 (PowerShell)
#  사용법:  .\run.ps1            # 기본(ipc) 모드로 시뮬레이터 실행
#           .\run.ps1 --mode wasd   # 키보드 조작 디버그 모드
#
#  시뮬레이터 GUI 우측 패널에서 학생 에이전트
#  (self-parking-user-algorithms/my_agent.py)를 선택해 실행하세요.
#  에이전트 서브프로세스도 자동으로 같은 venv python을 사용합니다.
# ============================================================
$ErrorActionPreference = "Stop"
$SimDir = Join-Path $PSScriptRoot "self-parking-sim-main"
$VenvPy = Join-Path $SimDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "[오류] 가상환경을 찾을 수 없습니다: $VenvPy" -ForegroundColor Red
    Write-Host "       먼저 setup이 완료되었는지 확인하세요."
    exit 1
}

Push-Location $SimDir
try {
    Write-Host "[실행] 시뮬레이터를 시작합니다..." -ForegroundColor Cyan
    & $VenvPy "demo_self_parking_sim.py" @args
}
finally {
    Pop-Location
}
