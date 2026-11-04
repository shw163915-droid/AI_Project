\
import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

SKAB_SENSOR_COLS = [
    'Accelerometer1RMS',
    'Accelerometer2RMS',
    'Current',
    'Pressure',
    'Temperature',
    'Thermocouple',
    'Voltage',
    'Volume Flow RateRMS',
]


class SKABDualLoader(Dataset):
    """
    SKAB 공식 다중 결함 데이터셋 로더
    - 공식 폴더 및 anomaly 컬럼 기반 라벨 매핑:
      * 0 : Normal (모든 파일에서 anomaly == 0.0 인 정상 구간)
      * 1 : WCA_1 (valve1 폴더에서 anomaly == 1.0 인 펌프 입구 밸브 결함 구간)
      * 2 : WCA_2 (valve2 폴더에서 anomaly == 1.0 인 펌프 출구 밸브 결함 구간)
      * -1: Unknown (other 폴더에서 anomaly == 1.0 인 미지의 결함 구간 - 오직 Test 셋에만 포함!)
    """
    def __init__(self, args, root_path, flag='train', stride=4):
        self.args = args
        self.root_path = root_path
        self.flag = flag.lower()
        self.seq_len = args.seq_len
        self.stride = 1 if self.flag == 'test' else stride
        self.sensor_names = SKAB_SENSOR_COLS

        data_dir = os.path.join(root_path, 'data') if os.path.exists(os.path.join(root_path, 'data')) else root_path

        valve1_files = sorted(
            glob.glob(os.path.join(data_dir, 'valve1', '*.csv')),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        valve2_files = sorted(
            glob.glob(os.path.join(data_dir, 'valve2', '*.csv')),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )
        other_files = sorted(
            glob.glob(os.path.join(data_dir, 'other', '*.csv')),
            key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
        )

        if len(valve1_files) == 0:
            raise FileNotFoundError(
                f"SKAB CSV 파일을 찾을 수 없습니다: {data_dir}. "
                f"'git clone https://github.com/waico/SKAB.git ./dataset/SKAB' 명령어를 먼저 실행해 주세요."
            )

        # 파일 단위 Split 분리 (윈도우가 서로 다른 실험 파일 경계를 넘지 않도록 파일별로 관리)
        # 1) Train: valve1(0~10.csv), valve2(0~1.csv) -> Normal(0), WCA_1(1), WCA_2(2) 학습
        # 2) Val:   valve1(11~12.csv), valve2(2.csv)  -> Normal(0), WCA_1(1), WCA_2(2) 검증 및 Threshold 산출
        # 3) Test:  valve1(13~15.csv), valve2(3.csv) + other(1~14.csv) -> 알려진 WCA 분류 + 미지의 이상(Unknown, -1) 탐지 평가
        train_specs = [(f, 1) for f in valve1_files[:11]] + [(f, 2) for f in valve2_files[:2]]
        val_specs = [(f, 1) for f in valve1_files[11:13]] + [(f, 2) for f in valve2_files[2:3]]
        test_specs = (
            [(f, 1) for f in valve1_files[13:]]
            + [(f, 2) for f in valve2_files[3:]]
            + [(f, -1) for f in other_files]
        )

        # 1. 먼저 Train 파일들의 Normal(anomaly == 0) 구간 데이터로 StandardScaler 학습
        train_raw_list = [self._load_single_csv(f, cls_id) for f, cls_id in train_specs]
        normal_chunks = [x[y == 0] for x, y in train_raw_list if (y == 0).sum() > 0]
        self.scaler = StandardScaler()
        self.scaler.fit(np.concatenate(normal_chunks, axis=0))

        # 2. 현재 flag(train / val / test)에 맞는 파일 로드 및 스케일링
        if self.flag == 'train':
            target_specs = train_specs
        elif self.flag in ['val', 'vali']:
            target_specs = val_specs
        else:
            target_specs = test_specs

        self.file_x_list = []
        self.file_y_list = []
        self.window_map = []  # (file_idx, start_row)

        for file_idx, (f_path, fault_cls_id) in enumerate(target_specs):
            x_arr, y_arr = self._load_single_csv(f_path, fault_cls_id)
            x_scaled = self.scaler.transform(x_arr).astype(np.float32)

            self.file_x_list.append(x_scaled)
            self.file_y_list.append(y_arr)

            max_start = len(x_scaled) - self.seq_len - 1
            for s in range(0, max_start, self.stride):
                self.window_map.append((file_idx, s))

        all_y = np.concatenate(self.file_y_list, axis=0)
        print(
            f"[{self.flag.upper()}] 파일 수: {len(target_specs)}개 | 윈도우 수: {len(self.window_map)}개 | "
            f"시점 분포 -> Normal(0): {(all_y == 0).sum()}, WCA_1(1): {(all_y == 1).sum()}, "
            f"WCA_2(2): {(all_y == 2).sum()}, Unknown(-1): {(all_y == -1).sum()}"
        )

    def _load_single_csv(self, csv_path, fault_cls_id):
        df = pd.read_csv(csv_path, sep=';')
        x = df[SKAB_SENSOR_COLS].values.astype(np.float32)
        anom = df['anomaly'].values.astype(np.int64)

        # 각 실험(비행)의 초기 정상 운전 구간(처음 150시점, anomaly==0) 평균을 기준점(Baseline)으로 정렬
        # (운전 세팅 날짜별 기본 유량/온도 차이를 보정하고 순수 결함 변화량과 센서 상관관계를 추출)
        normal_init = x[anom == 0][:150]
        if len(normal_init) > 10:
            x = x - normal_init.mean(axis=0, keepdims=True)

        # anomaly == 0 이면 0(Normal), anomaly == 1 이면 해당 폴더의 고장 번호(1: WCA_1, 2: WCA_2, -1: Unknown)
        y = np.where(anom == 1, fault_cls_id, 0).astype(np.int64)
        return x, y

    def __len__(self):
        return len(self.window_map)

    def __getitem__(self, idx):
        file_idx, start = self.window_map[idx]
        end = start + self.seq_len

        x_seq = self.file_x_list[file_idx]
        y_seq = self.file_y_list[file_idx]

        x_window = x_seq[start:end]        # [seq_len, 8] (t - seq_len + 1 ~ t)
        x_next = x_seq[start + 1:end + 1]  # [seq_len, 8] (한 스텝 뒤 센서 예측 타겟)
        y_label = y_seq[end - 1]           # 현재 시점(t)의 상태 라벨

        return (
            torch.from_numpy(x_window),
            torch.from_numpy(x_next),
            torch.tensor(y_label, dtype=torch.long),
        )

