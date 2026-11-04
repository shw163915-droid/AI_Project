# -*- coding: utf-8 -*-
"""
ALFA 데이터셋 -> GDN 입력 포맷(train.csv, test.csv, list.txt) 변환 스크립트 (2번 개선안 적용)
- 원본 ALFA 데이터셋(C:/AI Project/dataset/ALFA/processed)은 수정하지 않음
- [2번 개선안]:
  1) 비행별 초기 고도 오프셋 제거 (해발 고도 -> 비행 시작 기준 상대 고도 변화량 altitude_rel)
  2) 방위각(yaw) +-180도 점프(Wrap-around) 보정 및 최단 각도 오차(yaw_err) 산출
  3) 조종 명령(cmd)과 실제 측정값(meas) 간의 추종 오차(roll_err, pitch_err, yaw_err, airspeed_err, vel_err_z) 반영
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler


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

# 2번 개선이 반영된 최종 22개 센서 노드 목록
# (참고: ALFA 데이터셋에서 2018-09-11 이후 33개 비행은 피토관 센서 비활성화로 airspeed_meas=0.0 고정,
#  airspeed_cmd는 15/16 상수이므로 결측/상수 왜곡을 일으키는 airspeed 3개 채널을 제외하고 groundspeed 사용)
FINAL_FEATURE_COLS = [
    "roll_cmd",
    "roll_meas",
    "roll_err",
    "pitch_cmd",
    "pitch_meas",
    "pitch_err",
    "yaw_sin",
    "yaw_cos",
    "yaw_err",
    "vel_des_z",
    "vel_meas_z",
    "vel_err_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "groundspeed",
    "throttle",
    "altitude_rel",
    "climb_rate",
]


def find_fault_onset_ns(seq_dir: Path):
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


def apply_domain_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    [2번 개선안 핵심 로직]
    1) 비행 시작 구간(초반 2초) 기준 상대 고도(altitude_rel)로 오프셋 제거
    2) 방위각(yaw) +-180도 불연속 점프 제거 (sin/cos 분해 및 최단 각도 오차 yaw_err 계산)
    3) 명령-응답 추종 오차(roll_err, pitch_err, airspeed_err, vel_err_z) 산출
    """
    # 1. 비행별 초기 고도 오프셋 보정 (초반 20틱=2초 중앙값 기준)
    init_alt = df["altitude"].iloc[:20].median()
    df["altitude_rel"] = df["altitude"] - init_alt

    # 2. 방위각(yaw) 연속화 및 최단 회전 오차(-180 ~ +180) 계산
    yaw_rad = np.deg2rad(df["yaw_meas"].values)
    df["yaw_sin"] = np.sin(yaw_rad)
    df["yaw_cos"] = np.cos(yaw_rad)
    df["yaw_err"] = ((df["yaw_cmd"] - df["yaw_meas"] + 180.0) % 360.0) - 180.0

    # 3. 명령값(cmd) - 실제 측정값(meas) 추종 오차 계산
    df["roll_err"] = df["roll_cmd"] - df["roll_meas"]
    df["pitch_err"] = df["pitch_cmd"] - df["pitch_meas"]
    df["airspeed_err"] = df["airspeed_cmd"] - df["airspeed_meas"]
    df["vel_err_z"] = df["vel_des_z"] - df["vel_meas_z"]

    return df


def load_and_align_sequence(seq_dir: Path, freq_hz: int = 10):
    seq_name = seq_dir.name
    topic_dfs = []
    start_times = []
    end_times = []

    for suffix, col_map in TOPIC_SPECS:
        csv_path = seq_dir / f"{seq_name}-{suffix}"
        if not csv_path.exists():
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

    # 2번 개선: 도메인 피처 및 오프셋 보정 적용
    aligned_df = apply_domain_engineering(aligned_df)

    # 비행 시작 후 경과 시간(초) 계산
    aligned_df["flight_time_sec"] = np.round((aligned_df["time_ns"] - t_start) / 1e9, 2)

    fault_onset_ns = find_fault_onset_ns(seq_dir)
    if fault_onset_ns is not None:
        aligned_df["attack"] = (aligned_df["time_ns"] >= fault_onset_ns).astype(np.float64)
        fault_onset_sec = round(max(0.0, (fault_onset_ns - t_start) / 1e9), 2)
    else:
        aligned_df["attack"] = 0.0
        fault_onset_sec = None

    aligned_df["sequence"] = seq_name
    aligned_df.attrs["fault_onset_sec"] = fault_onset_sec
    return aligned_df


def prepare_alfa_dataset(
    alfa_root: Path,
    output_dir: Path,
    freq_hz: int = 10,
):
    seq_dirs = sorted([d for d in alfa_root.iterdir() if d.is_dir()])
    feature_cols = FINAL_FEATURE_COLS

    train_dfs = []
    test_dfs = []
    meta_records = []

    print(f"[INFO] Scanning {len(seq_dirs)} flight sequences in: {alfa_root}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        if "no_ground_truth" in seq_name:
            continue

        aligned_df = load_and_align_sequence(seq_dir, freq_hz=freq_hz)
        if aligned_df is None:
            continue

        is_normal_flight = seq_name.endswith("_no_failure")
        n_rows = len(aligned_df)
        n_anom = int(aligned_df["attack"].sum())
        fault_onset_sec = aligned_df.attrs.get("fault_onset_sec", None)

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
                "duration_sec": round(n_rows / freq_hz, 1),
                "fault_onset_sec": fault_onset_sec,
                "num_anomaly_samples": n_anom,
                "anomaly_ratio": round(n_anom / max(n_rows, 1), 4),
            }
        )

    train_full = pd.concat(train_dfs, ignore_index=True)
    test_full = pd.concat(test_dfs, ignore_index=True)

    train_features = train_full[feature_cols].ffill().bfill().fillna(0.0)
    test_features = test_full[feature_cols].ffill().bfill().fillna(0.0)

    scaler = MinMaxScaler(feature_range=(0, 1))
    x_train_scaled = scaler.fit_transform(train_features.values)
    x_test_scaled = scaler.transform(test_features.values)

    train_out_df = pd.DataFrame(x_train_scaled, columns=feature_cols)
    test_out_df = pd.DataFrame(x_test_scaled, columns=feature_cols)
    test_out_df["attack"] = test_full["attack"].values

    output_dir.mkdir(parents=True, exist_ok=True)

    list_path = output_dir / "list.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for col in feature_cols:
            f.write(f"{col}\n")

    train_out_df.to_csv(output_dir / "train.csv", index=True)
    test_out_df.to_csv(output_dir / "test.csv", index=True)

    meta_df = pd.DataFrame(meta_records)
    meta_df.to_csv(output_dir / "sequence_summary.csv", index=False)
    test_full[["time_ns", "flight_time_sec", "sequence", "attack"]].to_csv(
        output_dir / "test_sequence_map.csv", index=True
    )

    print("\n=============== ALFA -> GDN 변환 완료 (2번 개선안 적용) ===============")
    print(f"저장 경로       : {output_dir}")
    print(f"센서 노드 수    : {len(feature_cols)} 개 (list.txt)")
    print(f"정상 비행(Train): {len(train_dfs)} 개 시퀀스, 총 {len(train_out_df):,} 시점 (train.csv)")
    print(f"결함 비행(Test) : {len(test_dfs)} 개 시퀀스, 총 {len(test_out_df):,} 시점 (test.csv)")
    print("=======================================================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ALFA dataset to GDN format with Step 2 improvements")
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
