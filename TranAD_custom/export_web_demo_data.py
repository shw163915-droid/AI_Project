# -*- coding: utf-8 -*-
"""
Export 16-Channel Telemetry for TranAD Professional Terminal GCS Web Demo
"""

import sys
import json
from pathlib import Path
from unittest.mock import MagicMock

sys.modules['dgl'] = MagicMock()
sys.modules['dgl.nn'] = MagicMock()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANAD_REPO = PROJECT_ROOT / "TranAD"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(TRANAD_REPO) not in sys.path:
    sys.path.insert(0, str(TRANAD_REPO))

# Runtime PyTorch 2.x adapter
import src.dlutils as dlutils
_orig_tel_forward = dlutils.TransformerEncoderLayer.forward
def _compat_tel_forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):
    return _orig_tel_forward(self, src, src_mask=src_mask, src_key_padding_mask=src_key_padding_mask)
dlutils.TransformerEncoderLayer.forward = _compat_tel_forward

_orig_tdl_forward = dlutils.TransformerDecoderLayer.forward
def _compat_tdl_forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):
    return _orig_tdl_forward(self, tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
dlutils.TransformerDecoderLayer.forward = _compat_tdl_forward

from src.models import TranAD
from run_alfa_tranad import AlfaTranADRunner, FlightWindowDataset


def export_tranad_telemetry(model_path: str, output_json_path: Path, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    train_config = {
        "window_size": 10,
        "batch": 128,
        "epochs": 1,
        "lr": 1e-3,
        "patience": 10,
        "val_ratio": 0.1,
    }
    env_config = {
        "device": device,
        "load_model_path": model_path,
    }

    runner = AlfaTranADRunner(train_config, env_config)
    runner.model.load_state_dict(torch.load(model_path, map_location=device))
    runner.model.eval()

    feature_names = runner.feature_map
    K = runner.num_features

    mse_none = nn.MSELoss(reduction="none")
    recon_list, actual_list, score_list, sq_err_list = [], [], [], []

    with torch.no_grad():
        for src, tgt in runner.test_loader:
            src = src.permute(1, 0, 2).to(device)
            tgt = tgt.permute(1, 0, 2).to(device)
            x1, x2 = runner.model(src, tgt)

            err_matrix = mse_none(x2, tgt)[0]  # [Batch, K]
            score = torch.mean(err_matrix, dim=1)

            recon_list.append(x2[0].cpu().numpy())
            actual_list.append(tgt[0].cpu().numpy())
            score_list.append(score.cpu().numpy())
            sq_err_list.append(err_matrix.cpu().numpy())

    all_scores = np.concatenate(score_list)
    all_recon = np.vstack(recon_list)
    all_actual = np.vstack(actual_list)
    all_sq = np.vstack(sq_err_list)

    target_flights = [
        ("Elevator Failure (승강타 고착)", "elevator_failure"),
        ("Aileron Failure (보조익 고착)", "aileron_failure"),
        ("Engine Failure (엔진 정지)", "engine_failure"),
    ]

    flights_data = {}
    threshold = 0.001972

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
        f_actual = all_actual[s:e]
        f_recon = all_recon[s:e]
        f_sq = all_sq[s:e]
        T_len = e - s

        time_sec = np.round(np.arange(T_len) * 0.1, 2)
        fault_indices = np.where(f_gt == 1)[0]
        onset_sec = float(time_sec[fault_indices[0]]) if len(fault_indices) > 0 else None

        steps = []
        for t in range(T_len):
            channel_err = f_sq[t]
            tot_err = float(np.sum(channel_err)) + 1e-9
            pcts = np.round((channel_err / tot_err) * 100.0, 1)

            sorted_idx = np.argsort(channel_err)[::-1]
            top_cause_sensor = feature_names[int(sorted_idx[0])]
            top_cause_pct = float(pcts[int(sorted_idx[0])])

            steps.append(
                {
                    "t": float(time_sec[t]),
                    "score": round(float(f_scores[t]), 5),
                    "alarm": bool(f_scores[t] > threshold),
                    "gt": int(f_gt[t]),
                    "top1_sensor": top_cause_sensor,
                    "top1_pct": top_cause_pct,
                    "all_pcts": [float(p) for p in pcts],
                    "all_actual": [round(float(v), 3) for v in f_actual[t]],
                    "all_recon": [round(float(v), 3) for v in f_recon[t]],
                    "all_err": [round(float(v), 5) for v in channel_err],
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
                "model_name": "TranAD (Adversarial Transformer)",
                "channels": feature_names,
                "flights": flights_data,
            },
            f,
        )
    print(f"[SUCCESS] Exported TranAD Telemetry JSON: {output_json_path}")


if __name__ == "__main__":
    import glob
    model_files = sorted(glob.glob(str(CUSTOM_DIR / "pretrained" / "alfa_cmu16" / "best_tranad_*.pt")))
    out_json = CUSTOM_DIR / "results" / "tranad_demo_telemetry.json"
    export_tranad_telemetry(model_files[-1], out_json)

