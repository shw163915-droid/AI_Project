# -*- coding: utf-8 -*-
"""
CMU ALFA Dataset Preprocessing for CMU Author-Standard 16 Raw Physical Channels
Ref: Keipour et al., "ALFA: A Dataset for UAV Fault and Anomaly Detection" (IJRR / AirLab CMU)

Exact 16 Raw Physical Features (No artificial ERR or sin/cos features):
  1. roll_cmd     (Commanded roll angle)
  2. pitch_cmd    (Commanded pitch angle)
  3. yaw_cmd      (Commanded yaw angle)
  4. throttle     (Engine throttle command, 0~1)
  5. roll_meas    (Measured roll angle)
  6. pitch_meas   (Measured pitch angle)
  7. yaw_meas     (Measured yaw angle)
  8. gyro_x       (IMU roll rate gx)
  9. gyro_y       (IMU pitch rate gy)
 10. gyro_z       (IMU yaw rate gz)
 11. accel_x      (IMU longitudinal accel ax)
 12. accel_y      (IMU lateral accel ay)
 13. accel_z      (IMU vertical accel az)
 14. groundspeed  (Vehicle groundspeed m/s)
 15. altitude     (Altitude m)
 16. climb_rate   (Vertical speed / climb rate m/s)
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALFA_RAW_DIR = PROJECT_ROOT / "dataset" / "ALFA" / "processed"
OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "alfa_cmu16"

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

CMU_16_FEATURE_COLS = [
    "roll_cmd",
    "pitch_cmd",
    "yaw_cmd",
    "throttle",
    "roll_meas",
    "pitch_meas",
    "yaw_meas",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "accel_x",
    "accel_y",
    "accel_z",
    "groundspeed",
    "altitude",
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
                    onset_times.append(int(active["%time"].iloc[0]))
        except Exception:
            continue

    return min(onset_times) if onset_times else None


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


def prepare_cmu16_dataset(
    alfa_root: Path = ALFA_RAW_DIR,
    output_dir: Path = OUTPUT_DIR,
    freq_hz: int = 10,
):
    seq_dirs = sorted([d for d in alfa_root.iterdir() if d.is_dir()])
    feature_cols = CMU_16_FEATURE_COLS

    train_dfs = []
    test_dfs = []
    meta_records = []

    print(f"[INFO] Scanning ALFA processed directory: {alfa_root}")
    print(f"[INFO] Using exact 16 CMU AirLab raw features: {feature_cols}")

    for seq_dir in seq_dirs:
        seq_name = seq_dir.name
        df = load_and_align_sequence(seq_dir, freq_hz=freq_hz)
        if df is None:
            continue

        has_fault = bool(df["attack"].sum() > 0)
        is_normal_seq = ("no_failure" in seq_name) and (not has_fault)

        meta_records.append(
            {
                "sequence": seq_name,
                "split": "train" if is_normal_seq else "test",
                "length_steps": len(df),
                "duration_sec": round(len(df) / freq_hz, 2),
                "has_fault": has_fault,
                "fault_onset_sec": df.attrs.get("fault_onset_sec"),
            }
        )

        if is_normal_seq:
            train_dfs.append(df)
        else:
            test_dfs.append(df)

    print(f"[INFO] Aligned flights: Normal (Train)={len(train_dfs)}, Fault/Evaluation (Test)={len(test_dfs)}")

    meta_df = pd.DataFrame(meta_records)
    all_train_df = pd.concat(train_dfs, ignore_index=True)
    all_test_df = pd.concat(test_dfs, ignore_index=True)

    # Fit MinMaxScaler on normal training data only (strictly avoiding data leakage)
    scaler = MinMaxScaler(feature_range=(0.0, 1.0))
    scaler.fit(all_train_df[feature_cols].values)

    all_train_scaled = all_train_df.copy()
    all_test_scaled = all_test_df.copy()

    all_train_scaled[feature_cols] = np.clip(scaler.transform(all_train_df[feature_cols].values), 0.0, 1.0)
    all_test_scaled[feature_cols] = np.clip(scaler.transform(all_test_df[feature_cols].values), 0.0, 1.0)

    output_dir.mkdir(parents=True, exist_ok=True)

    train_out_csv = output_dir / "train.csv"
    test_out_csv = output_dir / "test.csv"
    list_out_txt = output_dir / "list.txt"
    meta_out_csv = output_dir / "sequence_summary.csv"

    # Save CSVs
    all_train_scaled[feature_cols].to_csv(train_out_csv, index=False)
    test_cols = feature_cols + ["attack", "flight_time_sec", "sequence"]
    all_test_scaled[test_cols].to_csv(test_out_csv, index=False)

    with open(list_out_txt, "w", encoding="utf-8") as f:
        for feat in feature_cols:
            f.write(f"{feat}\n")

    meta_df.to_csv(meta_out_csv, index=False)

    print(f"[SUCCESS] Prepared CMU-16 dataset in: {output_dir}")
    print(f"  - train.csv: {all_train_scaled.shape}")
    print(f"  - test.csv:  {all_test_scaled.shape}")
    print(f"  - features:  {len(feature_cols)} channels")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alfa_root", type=str, default=str(ALFA_RAW_DIR))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--freq_hz", type=int, default=10)
    args = parser.parse_args()

    prepare_cmu16_dataset(
        alfa_root=Path(args.alfa_root),
        output_dir=Path(args.output_dir),
        freq_hz=args.freq_hz,
    )

