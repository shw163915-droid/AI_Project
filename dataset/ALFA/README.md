# CMU ALFA (AirLab Failure and Anomaly) Dataset

고정익 무인기(Carbon Z T-28)의 실제 자율 비행 중 발생한 추진계 및 조종면(Control Surface) 결함 데이터셋입니다.  
모델에 종속되지 않도록 프로젝트 최상위(`C:\AI Project\dataset\ALFA`)에 위치하며, 데이터 파싱 및 전처리 파이프라인은 `Time-Series-Library` 베이스라인 코드를 활용합니다.

---

## 1. 디렉터리 구조

```text
C:\AI Project\
├── dataset/
│   └── ALFA/
│       ├── raw/                  # 원본 ALFA CSV 또는 ROS bag 추출 파일 저장 폴더
│       │   ├── normal/           # 정상 비행 시퀀스 (비지도 학습용)
│       │   └── fault/            # 결함 주입 비행 시퀀스 (실시간 이상 탐지 평가용)
│       ├── processed/            # Time-Series-Library 로더 연동용 데이터
│       └── README.md             # 데이터셋 설명 문서
├── Time-Series-Library/          # 데이터 파싱/전처리 및 베이스라인 프레임워크
├── Anomaly_Transformer/
└── TCN/
```

---

## 2. 공식 데이터셋 다운로드 링크

CMU AirLab에서 배포하는 공식 **Processed CSV (또는 MAT/ROSbag)** 파일은 아래 경로에서 다운로드하여 `raw/` 폴더에 배치할 수 있습니다.

* **공식 프로젝트 페이지:** [https://theairlab.org/alfa-dataset/](https://theairlab.org/alfa-dataset/)
* **CMU KiltHub (Figshare 직접 다운로드):** [https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963](https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963)
* **공식 툴킷 GitHub:** [https://github.com/castacks/alfa-dataset-tools](https://github.com/castacks/alfa-dataset-tools)

---

## 3. 주요 센서 변수 (16개 핵심 물리 노드)

| 노드 번호 | 변수명 (`feature_name`) | 물리적 의미 | 단위 | 역할 구분 |
| :---: | :--- | :--- | :---: | :--- |
| 0 | `roll_cmd` | 목표 롤(Roll) 명령각 | rad | 조종 명령 (Command) |
| 1 | `pitch_cmd` | 목표 피치(Pitch) 명령각 | rad | 조종 명령 (Command) |
| 2 | `yaw_cmd` | 목표 요(Yaw) 명령각 | rad | 조종 명령 (Command) |
| 3 | `throttle_cmd` | 엔진 스로틀 명령 | % (0~1) | 조종 명령 (Command) |
| 4 | `roll` | 실제 롤 자세각 | rad | 기체 자세 응답 (Attitude) |
| 5 | `pitch` | 실제 피치 자세각 | rad | 기체 자세 응답 (Attitude) |
| 6 | `yaw` | 실제 요 자세각 | rad | 기체 자세 응답 (Attitude) |
| 7 | `gx` (`roll_rate`) | 롤 각속도 ($p$) | rad/s | 각속도 응답 (IMU Gyro) |
| 8 | `gy` (`pitch_rate`) | 피치 각속도 ($q$) | rad/s | 각속도 응답 (IMU Gyro) |
| 9 | `gz` (`yaw_rate`) | 요 각속도 ($r$) | rad/s | 각속도 응답 (IMU Gyro) |
| 10 | `ax` | 종축 선형 가속도 | $\text{m/s}^2$ | 가속도 응답 (IMU Accel) |
| 11 | `ay` | 횡축 선형 가속도 | $\text{m/s}^2$ | 가속도 응답 (IMU Accel) |
| 12 | `az` | 수직 선형 가속도 | $\text{m/s}^2$ | 가속도 응답 (IMU Accel) |
| 13 | `airspeed` | 대기속도 (Pitot tube) | m/s | 비행 속도 상태 |
| 14 | `altitude` | 기압/GPS 고도 | m | 비행 고도 상태 |
| 15 | `vertical_speed` | 고도 상승/하강률 | m/s | 비행 동역학 상태 |

---

## 4. 결함 시나리오 목록 (Ground Truth Fault Types)

1. **Engine Full Power Loss:** 비행 중 엔진 완전 정지 (추력 상실, 대기속도/고도 급감)
2. **Elevator Stuck:** 승강타 고착 (`pitch_cmd` 변화에도 `pitch_rate`, `pitch` 무반응)
3. **Left / Right / Both Aileron Stuck:** 보조익 고착 (`roll_cmd`와 `roll_rate` 간 그래프 연결 붕괴)
4. **Rudder Stuck Left / Right:** 방향타 고착 (`yaw_cmd` 및 횡가속도 `ay` 비정상 편차 발생)

