# -*- coding: utf-8 -*-
"""
Export Complete 16-Channel Real-time Flight Telemetry & Contribution Data for Professional GCS
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAD_REPO_PATH = PROJECT_ROOT / "USAD"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(USAD_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(USAD_REPO_PATH))

from usad import UsadModel
from run_alfa_usad import AlfaUSADRunner


def export_complete_16ch_telemetry(
    model_path: str,
    output_json_path: Path,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    train_config = {
        "window_size": 12,
        "z_size": 48,
        "batch": 128,
        "epochs": 1,
        "lr": 1e-3,
        "patience": 15,
        "val_ratio": 0.1,
        "alpha": 0.5,
        "beta": 0.5,
    }
    env_config = {
        "dataset": "alfa_cmu16",
        "device": device,
        "load_model_path": model_path,
    }

    runner = AlfaUSADRunner(train_config, env_config)
    runner.model.load_state_dict(torch.load(model_path, map_location=device))
    runner.model.eval()

    K = runner.num_features
    win_size = train_config["window_size"]
    alpha, beta = train_config["alpha"], train_config["beta"]
    feature_names = runner.feature_map

    # Run full test set inference
    test_scores_list = []
    recon_list = []
    with torch.no_grad():
        for [batch] in runner.test_loader:
            batch = batch.to(device)
            w1 = runner.model.decoder1(runner.model.encoder(batch))
            w2 = runner.model.decoder2(runner.model.encoder(w1))

            err1 = torch.mean((batch - w1) ** 2, dim=1)
            err2 = torch.mean((batch - w2) ** 2, dim=1)
            score = alpha * err1 + beta * err2

            test_scores_list.append(score.cpu().numpy())
            recon_list.append(w1.cpu().numpy())

    all_scores = np.concatenate(test_scores_list)
    all_recon = np.vstack(recon_list)

    actual_windows = runner.test_data.reshape(-1, win_size, K)
    recon_windows = all_recon.reshape(-1, win_size, K)

    actual_newest = actual_windows[:, -1, :]  # [N, K]
    recon_newest = recon_windows[:, -1, :]    # [N, K]
    sq_errs = (actual_newest - recon_newest) ** 2  # [N, K]

    target_flights = [
        ("Elevator Failure (승강타 고착)", "elevator_failure"),
        ("Aileron Failure (보조익 고착)", "aileron_failure"),
        ("Engine Failure (엔진 정지)", "engine_failure"),
    ]

    flights_data = {}
    threshold = 0.195306

    for label_title, keyword in target_flights:
        matched_rec = None
        for rec in runner.test_flight_records:
            if keyword in rec["sequence"]:
                matched_rec = rec
                break
        if matched_rec is None:
            continue

        s, e = matched_rec["win_start"], matched_rec["win_end"]
        seq_name = matched_rec["sequence"]
        f_gt = runner.test_labels[s:e].astype(int)
        f_scores = all_scores[s:e]
        f_actual = actual_newest[s:e]
        f_recon = recon_newest[s:e]
        f_sq = sq_errs[s:e]
        T_len = e - s

        time_sec = np.round(np.arange(T_len) * 0.1, 2)
        fault_indices = np.where(f_gt == 1)[0]
        onset_sec = float(time_sec[fault_indices[0]]) if len(fault_indices) > 0 else None

        steps = []
        for t in range(T_len):
            channel_err = f_sq[t]
            tot_err = float(np.sum(channel_err)) + 1e-9
            pcts = np.round((channel_err / tot_err) * 100.0, 1)

            # Store all 16 channels data for full real-time matrix
            all_ch_pcts = [float(p) for p in pcts]
            all_ch_actual = [round(float(v), 3) for v in f_actual[t]]
            all_ch_recon = [round(float(v), 3) for v in f_recon[t]]
            all_ch_err = [round(float(v), 4) for v in channel_err]

            sorted_idx = np.argsort(channel_err)[::-1]
            top_cause_sensor = feature_names[int(sorted_idx[0])]
            top_cause_pct = float(pcts[int(sorted_idx[0])])

            steps.append(
                {
                    "t": float(time_sec[t]),
                    "score": round(float(f_scores[t]), 4),
                    "alarm": bool(f_scores[t] > threshold),
                    "gt": int(f_gt[t]),
                    "top1_sensor": top_cause_sensor,
                    "top1_pct": top_cause_pct,
                    "all_pcts": all_ch_pcts,
                    "all_actual": all_ch_actual,
                    "all_recon": all_ch_recon,
                    "all_err": all_ch_err,
                }
            )

        flights_data[label_title] = {
            "sequence": seq_name,
            "duration_sec": float(time_sec[-1]),
            "onset_sec": onset_sec,
            "threshold": threshold,
            "steps": steps,
        }

    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "channels": feature_names,
                "flights": flights_data,
            },
            f,
        )
    print(f"[SUCCESS] Exported 16-channel Complete Telemetry: {output_json_path}")


if __name__ == "__main__":
    import glob
    model_files = sorted(glob.glob(str(CUSTOM_DIR / "pretrained" / "alfa_cmu16" / "best_*.pt")))
    out_json = CUSTOM_DIR / "results" / "web_demo_telemetry_16ch.json"
    export_complete_16ch_telemetry(model_files[-1], out_json)
