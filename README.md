# AI_Project

고정익 항공기 비행 데이터(CMU ALFA Dataset) 기반 실시간 이상 탐지 프로젝트 저장소입니다.

## 디렉터리 구조

- `dataset/ALFA/`: CMU ALFA 원본 데이터셋 안내 및 로컬 저장 경로
- `GDN/`: Graph Deviation Network (AAAI 2021) 공식 오픈소스 베이스라인 (원본 유지)
- `Time-Series-Library/`: 시계열 분석/이상 탐지 공식 오픈소스 베이스라인 (원본 유지)
- `GDN_custom/`: 베이스라인 코드를 수정하지 않고 `import`하여 ALFA 데이터셋 전처리, 학습 및 실시간 이상 탐지를 수행하는 커스텀 워크스페이스
  - `prepare_alfa.py`: ALFA 원본 시퀀스(10Hz 동기화) $\rightarrow$ GDN 입력 포맷(`train.csv`, `test.csv`, `list.txt`) 변환 스크립트
  - `data/alfa/`: 변환 완료된 24개 고정익 물리/제어 센서 학습 및 테스트 데이터셋

