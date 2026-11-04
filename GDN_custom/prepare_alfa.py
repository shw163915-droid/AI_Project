# -*- coding: utf-8 -*-
"""
ALFA 데이터셋 -> GDN 입력 포맷(train.csv, test.csv, list.txt) 변환 스크립트
- 원본 ALFA 데이터셋(C:/AI Project/dataset/ALFA/processed)은 수정하지 않음
- GDN 공식 전처리(GDN/scripts/process_swat.py) 규격(MinMaxScaler [0, 1], index_col=0, attack 라벨) 준수
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


# 추출할 ALFA 토픽 파일 접미사 및 컬럼 매핑 (총 24개 고정익 비행 역학/제어 센서 노드)
TOPIC_SPECS = [
    (
        "mavros-nav_info-roll.csv",
        {
            "field.commanded": "roll_cmd",
            "field.measured": "roll_meas",
        },
    ),
    (
        "mavros-nav_info-pitch.csv",
        {
            "field.commanded": "pitch_cmd",
            "field.measured": "pitch_meas",
        },
    ),
    (
        "mavros-nav_info-yaw.csv",
        {
            "field.commanded": "yaw_cmd",
            "field.measured": "yaw_meas",
        },
    ),
    (
        "mavros-nav_info-airspeed.csv",
        {
            "field.commanded": "airspeed_cmd",
            "field.measured": "airspeed_meas",
        },
    ),
    (
        "mavros-nav_info-velocity.csv",
        {
            "field.des_x": "vel_des_x",
            "field.des_y": "vel_des_y",
            "field.des_z": "vel_des_z",
            "field.meas_x": "vel_meas_x",
            "field.meas_y": "vel_meas_y",
            "field.meas_z": "vel_meas_z",
        },
    ),
    (
        "mavros-imu-data.csv",
        {
            "field.angular_velocity.x": "gyro_x",
            "field.angular_velocity.y": "gyro_y",
            "field.angular_velocity.z": "gyro_z",
            "field.linear_acceleration.x": "accel_x",
            "field.linear_acceleration.y": "accel_y",
            "field.linear_acceleration.z": "accel_z",
        },
    ),
    (
        "mavros-vfr_hud.csv",
        {
            "field.groundspeed": "groundspeed",
            "field.throttle": "throttle",
            "field.altitude": "altitude",
            "field.climb": "climb_rate",
        },
    ),
]


def get_feature_names():
    names = []
    for _, col_map in TOPIC_SPECS:
        names.extend(col_map.values())
    return names


def find_fault_onset_ns(seq_dir: Path):
    """
    시퀀스 폴더 내 *-failure_status-*.csv 파일에서 결함이 최초 발생한 나노초(%time) 반환.
    정상 비행(no_failure)이면 None 반환.
    """
    fault_files = list(seq_dir.glob("*-failure_status-*.csv"))
    if not fault_files:
        return None

    onset_times = []
    for fpath in fault_files:
        try:
            df = pd.read_csv(fpath)
            if "%time" in df.columns and "field.data" in df.columns:
                active = df[df["field.data"] > 0]
                if not active.empty:
                    onset_times.append(int(active["%time"].min()))
                elif not df.empty:
                    onset_times.append(int(df["%time"].min()))
        except Exception as e:
            print(f"[WARN] Failed reading {fpath.name}: {e}")

    return min(onset_times) if onset_times else None


def load_and_align_sequence(seq_dir: Path, freq_hz: int = 10):
    """
    단일 비행 시퀀스 폴더의 다중 센서 CSV 파일들을 공통 시간 그리드(freq_hz)로 선형 보간 정렬.
    """
    seq_name = seq_dir.name
    topic_dfs = []
    start_times = []
    end_times = []

    for suffix, col_map in TOPIC_SPECS:
        csv_path = seq_dir / f"{seq_name}-{suffix}"
        if not csv_path.exists():
            # 일부 시퀀스에 특정 파일이 누락된 경우 스킵 처리용
            return None

        df = pd.read_csv(csv_path)
        if "%time" not in df.columns or len(df) < 2:
            return None

        sub_cols = ["%time"] + list(col_map.keys())
        df = df[sub_cols].rename(columns=col_map)
        df = df.sort_values("%time").drop_duplicates(subset=["%time"])

        start_times.append(df["%time"].iloc[0])
        end_times.append(df["%time"].iloc[-1])
        topic_dfs.append(df)

    # 모든 센서가 공통으로 존재하는 시간 구간 설정
    t_start = max(start_times)
    t_end = min(end_times)
    if t_end <= t_start:
        return None

    step_ns = int(1e9 / freq_hz)
    common_time_ns = np.arange(t_start, t_end, step_ns, dtype=np.int64)
    if len(common_time_ns) < 20:
        return None

    aligned_data = {"time_ns": common_time_ns}
    for df in topic_dfs:
        t_vals = df["%time"].values.astype(np.float64)
        for col in df.columns:
            if col == "%time":
                continue
            y_vals = df[col].values.astype(np.float64)
            aligned_data[col] = np.interp(common_time_ns.astype(np.float64), t_vals, y_vals)

    aligned_df = pd.DataFrame(aligned_data)

    # 결함 라벨(attack: 0.0 정상, 1.0 결함) 부여
    fault_onset_ns = find_fault_onset_ns(seq_dir)
    if fault_onset_ns is not None:
        aligned_df["attack"] = (aligned_df["time_ns"] >= fault_onset_ns).astype(np.float64)
    else:
        aligned_df["attack"] = 0.0

    aligned_df["sequence"] = seq_name
    return aligned_df


def prepare_alfa_dataset(
    alfa_root: Path,
    output_dir: Path,
    freq_hz: int = 10,
):
    seq_dirs = sorted([d for d in alfa_root.iterdir() if d.is_dir()])
    feature_cols = get_feature_names()

    train_dfs = []
    test_dfs = []
    meta_records = []

    print(f"[INFO] Scanning {len(seq_dirs)} flight sequences in: {alfa_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        if "no_ground_truth" in seq_name:
            print(f"  - Skipping {seq_name} (no ground truth)")
            continue

        aligned_df = load_and_align_sequence(seq_dir, freq_hz=freq_hz)
        if aligned_df is None:
            print(f"  - Skipping {seq_name} (missing required topics or too short)")
            continue

        is_normal_flight = seq_name.endswith("_no_failure")
        n_rows = len(aligned_df)
        n_anom = int(aligned_df["attack"].sum())

        if is_normal_flight:
            train_dfs.append(aligned_df)
            split_type = "train"
        else:
            test_dfs.append(aligned_df)
            split_type = "test"

        meta_records.append(
            {
                "sequence": seq_name,
                "split": split_type,
                "num_samples": n_rows,
                "num_anomaly_samples": n_anom,
                "anomaly_ratio": round(n_anom / max(n_rows, 1), 4),
            }
        )

    train_full = pd.concat(train_dfs, ignore_index=True)
    test_full = pd.concat(test_dfs, ignore_index=True)

    # 결측치 방어 처리
    train_features = train_full[feature_cols].ffill().bfill().fillna(0.0)
    test_features = test_full[feature_cols].ffill().bfill().fillna(0.0)

    # GDN 공식 전처리(process_swat.py)와 동일하게 정상 학습 데이터 기준 MinMaxScaler(0, 1) 정규화
    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(train_features.values)
    x_test_scaled = scaler.transform(test_features.values)

    # 테스트 데이터의 극단적 이상치가 스케일러 범위를 과도하게 벗어나 수치 불안정을 일으키지 않도록 완만한 클리핑 옵션 적용 가능하나
    # GDN 편차 계산을 위해 원본 변환값 유지
    train_out_df = pd.DataFrame(x_train_scaled, columns=feature_cols)
    test_out_df = pd.DataFrame(x_test_scaled, columns=feature_cols)
    test_out_df["attack"] = test_full["attack"].values

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. list.txt 저장
    list_path = output_dir / "list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    # 2. train.csv & test.csv 저장 (GDN의 pd.read_csv(..., index_col=0) 규격 준수)
    train_csv_path = output_dir / "train.csv"
    test_csv_path = output_dir / "test.csv"
    train_out_df.to_csv(train_csv_path, index=True)
    test_out_df.to_csv(test_csv_path, index=True)

    # 3. 시퀀스별 메타정보 및 테스트 시퀀스 매핑 저장 (추후 비행별/결함별 상세 분석용)
    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(output_dir / "sequence_summary.csv", index=False)
    test_full[["time_ns", "sequence", "attack"]].to_csv(
        output_dir / "test_sequence_map.csv", index=True
    )

    print("\n==================== ALFA -> GDN 변환 완료 ====================")
    print(f"저장 경로       : {output_dir}")
    print(f"센서 노드 수    : {len(feature_cols)} 개 (list.txt)")
    print(f"정상 비행(Train): {len(train_dfs)} 개 시퀀스, 총 {len(train_out_df):,} 시점 (train.csv)")
    print(f"결함 비행(Test) : {len(test_dfs)} 개 시퀀스, 총 {len(test_out_df):,} 시점 (test.csv)")
    print(
        f"Test 내 이상비율: {int(test_out_df['attack'].sum()):,} / {len(test_out_df):,} "
        f"({test_out_df['attack'].mean() * 100:.2f}%)"
    )
    print("===============================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ALFA dataset to GDN format")
    parser.add_argument(
        "--alfa_root",
        type=str,
        default=r"C:\AI Project\dataset\ALFA\processed",
        help="Path to extracted ALFA processed folder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=r"C:\AI Project\GDN_custom\data\alfa",
        help="Output directory for GDN formatted files",
    )
    parser.add_argument(
        "--freq_hz",
        type=int,
        default=10,
        help="Resampling frequency in Hz (default: 10Hz)",
    )
    args = parser.parse_args()

    prepare_alfa_dataset(
        alfa_root=Path(args.alfa_root),
        output_dir=Path(args.output_dir),
        freq_hz=args.freq_hz,
    )

