# AI_Project: 고정익 항공기 비행 데이터 실시간 이상 탐지 (CMU ALFA + GDN)

고정익 무인항공기 비행 데이터셋인 **CMU ALFA (AirLab Failure and Anomaly) Dataset**과 **GDN (Graph Deviation Network, AAAI 2021)**을 활용하여, 정상 비행 데이터만으로 학습하고 엔진·엘리베이터·러더·에일러론 등 조종면 결함을 실시간으로 탐지 및 원인 센서(Root Cause)까지 지목하는 비지도 이상 탐지 프로젝트입니다.

---

## 1. 프로젝트 원칙 및 디렉터리 구조

1. **오픈소스 베이스라인 무수정 원칙 (Zero Baseline Modification)**
   - `GDN/` 및 `Time-Series-Library/` 내부의 원본 코드는 단 한 줄도 수정하지 않고 그대로 보존합니다.
   - 모든 전처리, PyTorch Geometric 2.x 호환 어댑터, 에피소드 윈도윙, 잔차 극소화 아키텍처 어댑터, 학습/평가 및 대시보드 시각화는 `GDN_custom/` 내에서 원본 모듈을 `import`하여 수행합니다.
2. **디렉터리 구조**
   ```text
   C:\AI Project\
   ├── dataset/ALFA/                 # CMU ALFA 원본 데이터셋 안내 및 47개 비행 시퀀스 데이터
   ├── GDN/                          # Graph Deviation Network (AAAI 2021) 공식 레포지토리 (원본 유지)
   ├── Time-Series-Library/          # 시계열 이상 탐지 공식 오픈소스 레포지토리 (원본 유지)
   └── GDN_custom/                   # 커스텀 전처리·학습·평가 워크스페이스
       ├── prepare_alfa.py           # ALFA 47개 비행 시퀀스 10Hz 동기화 + 도메인 엔지니어링 변환 스크립트
       ├── run_alfa_gdn.py           # GDN 학습/평가 러너 (ConcatDataset, GELU 잔차 제거, EWMA, 대시보드 생성)
       ├── data/alfa/                # 변환 완료된 GDN 입력 데이터셋 (train.csv, test.csv, list.txt 등)
       ├── pretrained/alfa/          # 학습된 Best 모델 가중치 (.pt)
       └── results/alfa/             # 실험별 성능 지표(.json), 시점별 점수(.csv), 시각화 대시보드(_dashboard.png)
   ```

---

## 2. 데이터셋 구성 (CMU ALFA Dataset)

- **원본 출처:** Carnegie Mellon University AirLab (KiltHub / Figshare `processed.zip`)
- **비행 시퀀스 구성:** 총 47개 자율 비행 시퀀스 (사용: 46개 시퀀스)
  - **Train (정상 비행):** `*_no_failure` 10개 비행 시퀀스 (총 `5,530` 타임스텝, 10Hz 기준 `553.0초`)
  - **Test (결함 비행):** `*_failure` 36개 비행 시퀀스 (총 `41,426` 타임스텝, 엔진/엘리베이터/러더/에일러론 고장 포함)
  - **제외:** `carbonZ_2018-07-18-12-10-11_no_ground_truth` (정답 라벨 미존재 1개 비행)
- **샘플링 동기화:** 비동기 토픽(`20~25Hz` 자세 제어 토픽 및 `3~5Hz` IMU/HUD 토픽)을 공통 시간축인 **`10Hz` (`0.1초` 간격)** 그리드로 선형 보간(`np.interp`) 정렬.
- **고장 시점 라벨링 (`attack`):** 각 비행 폴더의 `*-failure_status-{engines,elevator,aileron,rudder}.csv`에서 `field.data > 0`이 최초 발생한 나노초(`%time`) 시점부터 비행 종료 시까지 `attack = 1.0`으로 부여.

---

## 3. 단계별 핵심 기술 개선 내역

### [개선 1] 비행 에피소드별 독립 슬라이딩 윈도우 (`ConcatDataset`) 및 에피소드 균등 검증 분할
- **문제점:** 원본 GDN은 단일 연속 시계열(SWaT/WADI)을 가정하여 전체 CSV를 통째로 슬라이딩 윈도우로 자릅니다. 다중 비행 데이터인 ALFA에서 이를 그대로 쓰면 앞 비행의 종료 데이터와 다음 비행의 이륙 데이터가 하나의 윈도우(`slide_win=15`)에 섞이는 경계 오염이 발생합니다. 또한 단일 구간만 잘라 Validation으로 쓰면 특정 1개 비행에만 편중되어 조기 종료(Early Stopping)가 발생합니다.
- **해결 (`run_alfa_gdn.py`):**
  1. `sequence_summary.csv`의 비행별 샘플 수를 기준으로 각 비행을 독립된 `TimeDataset`으로 생성한 뒤 PyTorch `ConcatDataset`으로 결합하여 비행 간 경계 침범을 원천 차단했습니다.
  2. **Episode-wise Temporal Block Split (`get_loaders`):** 시계열 연속성(Data Leakage 방지)을 지키면서 10개 정상 비행 전체가 검증에 골고루 반영되도록, 각 정상 비행의 앞쪽 90% 구간을 Train으로, 마지막 10% 구간을 Validation으로 균등 분할했습니다.

### [개선 2] 항공 비행역학 도메인 피처 엔지니어링 및 결측 채널 정제 (`22개 센서 노드`)
- **문제점:**
  1. 절대 해발 고도(`altitude`)는 비행일/기압마다 초기 오프셋이 달라 정상 비행끼리도 스케일이 어긋남.
  2. 방위각(`yaw`)은 $\pm 180^\circ$ 경계에서 불연속 점프(Wrap-around)가 발생하여 정상 선회 중에도 큰 예측 오차를 유발함.
  3. 북쪽/동쪽 속도(`vel_x`, `vel_y`)는 기수 방향(Heading)에 따라 부호가 반전됨.
  4. **피토관(Pitot tube) 결측 규명:** 2018-09-11 이후 촬영된 33개 비행에서 피토관 센서 비활성화로 `airspeed_meas = 0.0`으로 고정되어 있고 `airspeed_cmd`는 `15/16` 상수로만 기록되어 전체 오경보의 60%를 유발함.
- **해결 (`prepare_alfa.py`):**
  1. **상대 고도 변환:** 각 비행 시작 초반 2초 중앙값을 기준으로 상대 고도 변화량(`altitude_rel`) 산출.
  2. **방위각 연속화:** `yaw_meas`를 연속 삼각함수(`yaw_sin`, `yaw_cos`)로 분해하고, 최단 회전 오차 `yaw_err = ((yaw_cmd - yaw_meas + 180) % 360) - 180` 반영.
  3. **조종 명령-응답 추종 잔차(Tracking Residual) 직접 주입:** 조종면 결함 시 즉각 벌어지는 `roll_err`, `pitch_err`, `yaw_err`, `vel_err_z`를 파생 피처로 추가하고, 결측된 `airspeed` 3개 채널 및 방위 종속 수평 속도를 제거하여 최종 **22개 핵심 물리 노드**(`data/alfa/list.txt`)로 정제.

### [개선 3] 현업 표준 EWMA (지수 가중 이동 평균) 기반 실시간 이상 점수 산출
- **해결 (`run_alfa_gdn.py` - `compute_ewma_err_scores`):**
  - 원본 GDN의 Median/IQR 정규화 공식을 유지하되, 단순 이동 평균(SMA)이 유발하는 탐지 지연을 없애기 위해 각 비행 에피소드별로 **EWMA ($\alpha = 0.3$)**를 적용했습니다:
    $$S_t = \alpha \cdot e_t + (1 - \alpha) \cdot S_{t-1}$$

### [개선 4] 정상 상태 예측 잔차(Residual) 제거를 위한 GDN 아키텍처 병목 해결 및 고찰
- **원인 규명:**
  1. **Dead ReLU Clipping:** 원본 `GDN.py`(179행)의 `F.relu(self.bn_outlayer_in(out))`는 배치 정규화 이후 평균 이하(골짜기 구간)의 음수 특징값을 전부 `0`으로 절단하여, 예측 파형이 아래쪽 골짜기(`0.42`)로 내려가지 못하고 `0.55` 부근에서 수평으로 잘리는 현상을 유발했습니다.
  2. **출력 직전 Dropout 손실:** `self.dp = nn.Dropout(0.2)`가 회귀 출력 직전 뉴런 20%를 꺼버려 고점/저점 진폭을 20% 축소시켰습니다.
- **해결 (`_enable_cross_sensor_gdn_forward`):**
  - 원본 `GDN/` 파일 수정 없이 런타임 어댑터를 통해 출력 직전 `Dropout`을 `Identity()`로 비활성화하고, `ReLU`를 양방향 미분이 가능한 **`GELU` 활성화 함수 + Multi-Head Graph Attention(`heads=4`) + 3층 MLP + `ReduceLROnPlateau` 스케줄러**로 교체했습니다.
  - **결과:** 학습 MSE Loss가 `0.0156` $\rightarrow$ **`0.0013` (11배 감소)**, 검증 Loss가 `0.0184` $\rightarrow$ **`0.0024` (7.4배 감소)**로 수렴하며 정상 구간(0~116.3초)의 고점과 저점 모두 예측선이 실제선과 빈틈없이 일치했습니다 (`results/alfa/11_05-02_15_36_dashboard.png`).
- **시계열 이상 탐지에서의 핵심 고찰 (One-Step Self-Loop 현상):**
  - GDN은 매 0.1초마다 직전 1.5초(`t-1.5s ~ t-0.1s`)의 실제 측정값을 입력받아 다음 1스텝(`t`)을 예측하며, 그래프 내부에 **자기 자신 노드를 참조하는 화살표(`add_self_loops`)**를 포함합니다.
  - 따라서 표현력이 극대화된 모델이 정상 구간 오차를 0으로 줄이는 과정에서 자기 자신의 직전 시점 값($x_{t-1}$)에 높은 가중치를 두게 되면, 고장 발생 후에도 직전 0.1초 전의 고장 난 값을 그대로 추종하여 이상 구간까지 잘 예측해 버리는 **Over-generalization(과도한 일반화)** 현상이 나타납니다.
  - 이를 극복하기 위해서는 그래프 어텐션에서 자기 자신 노드를 참조하는 **Self-Loop를 마스킹(제거)**하여 오직 타 물리 센서들과의 교차 관계(Cross-Sensor Physics)만으로 예측하도록 강제하는 설계가 핵심임을 규명했습니다.

---

## 4. 실험 결과 및 성능 비교 요약

| 실험 단계 | 노드 수 | Val MSE Loss | Global F1 | Global AUROC | 비행별 평균 AUROC | 탐지 비행 수 (평균 지연) | `elevator_failure` 특징 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **초기 베이스라인** (단순 결합) | 24 | `0.0241` | `0.5617` | `0.7227` | - | - | `pitch_meas` 지목 (정상 구간 오경보 다수) |
| **1번 개선** (비행별 윈도우 분리) | 24 | `0.0229` | `0.6143` | `0.7429` | `0.8767` | `34 / 36` (`0.79s`) | 비행 경계 오염 제거 |
| **2번 개선** (`airspeed` 결측 제거 전) | 25 | `0.0265` | `0.4255` | `0.6810` | `0.8106` | `36 / 36` (`1.58s`) | `pitch_err` 정확 지목 (`airspeed` 오경보 규명) |
| **3번 개선** (22노드 정제 + EWMA) | 22 | `0.0184` | **`0.6138`** | **`0.7679`** | **`0.8810`** | `34 / 36` (`1.93s`) | `elevator_failure` F1 **`0.7404`**, 순항 오경보 0회 |
| **4번 개선** (`GELU` + 잔차 극소화) | 22 | **`0.0024`** | `0.4364` | `0.7151` | `0.8224` | **`36 / 36` (`1.24s`)** | 정상 구간 실제선-예측선 완벽 일치 (잔차 제거) |

---

## 5. 실행 방법

### 1) 데이터 전처리 재생성 (필요 시)
```powershell
& "C:\AI Project\.venv\Scripts\python.exe" "C:\AI Project\GDN_custom\prepare_alfa.py"
```

### 2) GDN 모델 학습 및 실시간 이상 탐지 대시보드 생성
VS Code에서 `GDN_custom/run_alfa_gdn.py`를 열고 **`F5`** (또는 우측 상단 `▶ Run Python File`)를 누르거나 터미널에서 아래 명령어를 실행합니다:
```powershell
& "C:\AI Project\.venv\Scripts\python.exe" "C:\AI Project\GDN_custom\run_alfa_gdn.py" -epoch 50
```
실행이 완료되면 `GDN_custom/results/alfa/` 폴더에 아래 3개 파일이 자동 생성됩니다:
1. `<타임스탬프>.json`: 전체 및 36개 개별 비행별 F1, AUROC, 탐지 지연 시간(초), 학습된 센서 그래프 연결 구조(`learned_graph_topk`)
2. `<타임스탬프>_scores.csv`: 전체 테스트 시점별 이상 점수, 정답 라벨, 예측 결과, 실시간 원인 센서(`root_cause_sensor`)
3. `<타임스탬프>_dashboard.png`: Loss 수렴 곡선 + 단일 비행 실제값 vs 예측값 + 실시간 이상 점수 및 임계값 3단 대시보드
