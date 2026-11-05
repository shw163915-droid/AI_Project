# -*- coding: utf-8 -*-
"""
Generate:
 1. Visual Dashboard PNG (Training Loss, Actual vs TranAD Recon, Anomaly Score & Alarm)
 2. Interactive Plotly HTML Graph (22/16-channel Heatmap, Interactive Dropdown Selector, Score)
"""

import sys
import glob
from pathlib import Path
from unittest.mock import MagicMock

sys.modules['dgl'] = MagicMock()
sys.modules['dgl.nn'] = MagicMock()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
from run_alfa_tranad import AlfaTranADRunner


def generate_tranad_plots(model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
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

    print("[INFO] Running evaluation over all 37 test flights...")
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
    labels = runner.test_labels.astype(int)

    threshold = 0.001972

    # Find representative elevator failure flight
    target_rec = None
    for rec in runner.test_flight_records:
        if "elevator_failure" in rec["sequence"]:
            target_rec = rec
            break
    if target_rec is None:
        target_rec = runner.test_flight_records[0]

    s, e = target_rec["win_start"], target_rec["win_end"]
    seq_name = target_rec["sequence"]
    f_scores = all_scores[s:e]
    f_actual = all_actual[s:e]
    f_recon = all_recon[s:e]
    f_sq = all_sq[s:e]
    f_gt = labels[s:e]
    time_sec = np.arange(e - s) * 0.1

    fault_indices = np.where(f_gt == 1)[0]
    onset_sec = time_sec[fault_indices[0]] if len(fault_indices) > 0 else None
    end_sec = time_sec[-1]

    # Find top-1 root cause channel during fault
    if onset_sec is not None:
        fault_means = np.mean(f_sq[fault_indices], axis=0)
        top1_idx = int(np.argmax(fault_means))
    else:
        top1_idx = 0
    top1_sensor = feature_names[top1_idx]

    # -------------------------------------------------------------
    # 1. GENERATE STATIC RESULT DASHBOARD PNG
    # -------------------------------------------------------------
    fig, axes = plt.subplots(3, 1, figsize=(13, 10))

    # [Subplot 1] Multi-Head Reconstruction Error vs Normal
    axes[0].plot(time_sec, f_scores, color="tab:blue", linewidth=1.2, label="TranAD Reconstruction Error (All 16 Ch)")
    axes[0].axhline(threshold, color="tab:red", linestyle="--", linewidth=1.3, label=f"Anomaly Threshold ({threshold:.5f})")
    if onset_sec is not None:
        axes[0].axvline(onset_sec, color="tab:red", linewidth=1.5, label=f"Elevator Fault Onset ({onset_sec:.1f}s)")
        axes[0].axvspan(onset_sec, end_sec, color="tab:orange", alpha=0.18, label="Fault Interval")
    axes[0].set_title(f"1. TranAD Real-time Reconstruction Loss on Flight [{seq_name}]")
    axes[0].set_xlabel("Flight Time (seconds)")
    axes[0].set_ylabel("Phase 2 Adversarial Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper left")

    # [Subplot 2] Actual vs TranAD Reconstructed Signal for Top-1 Root Cause
    pitch_idx = feature_names.index("pitch_meas")
    axes[1].plot(time_sec, f_actual[:, pitch_idx], color="black", linewidth=1.3, label=f"Actual Measured (pitch_meas)")
    axes[1].plot(time_sec, f_recon[:, pitch_idx], color="tab:orange", linestyle="--", linewidth=1.3, label=f"TranAD Reconstructed (pitch_meas)")
    if onset_sec is not None:
        axes[1].axvline(onset_sec, color="tab:red", linewidth=1.5)
        axes[1].axvspan(onset_sec, end_sec, color="tab:orange", alpha=0.18)
    axes[1].set_title(f"2. Sensor Actual vs TranAD Reconstruction (Root Cause: {top1_sensor})")
    axes[1].set_xlabel("Flight Time (seconds)")
    axes[1].set_ylabel("Scaled Value (0~1)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper left")

    # [Subplot 3] Root Cause Attribution: pitch_meas Error Spike
    pitch_err_sq = f_sq[:, pitch_idx]
    axes[2].plot(time_sec, pitch_err_sq, color="tab:purple", linewidth=1.3, label="pitch_meas Squared Reconstruction Error (Δ²)")
    if onset_sec is not None:
        axes[2].axvline(onset_sec, color="tab:red", linewidth=1.5)
        axes[2].axvspan(onset_sec, end_sec, color="tab:orange", alpha=0.18)
    axes[2].set_title("3. Root-Cause Reconstruction Error Deviation (Elevator Actuator Decoupling)")
    axes[2].set_xlabel("Flight Time (seconds)")
    axes[2].set_ylabel("Squared Error (Δ²)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper left")

    plt.tight_layout()
    png_path = CUSTOM_DIR / "results" / "alfa_tranad_result_dashboard.png"
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    print(f"[SUCCESS] Saved TranAD Result Dashboard PNG to: {png_path}")

    # -------------------------------------------------------------
    # 2. GENERATE INTERACTIVE PLOTLY HTML GRAPH
    # -------------------------------------------------------------
    plotly_html_path = CUSTOM_DIR / "results" / "alfa_tranad_interactive_graph.html"
    
    fig_plotly = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            "<b>1. All 16 Raw Physical Channels Reconstruction Error Heatmap</b>",
            "<b>2. Interactive Multi-Sensor Actual vs TranAD Reconstructed Signal (Dropdown Selector)</b>",
            f"<b>3. Integrated TranAD Anomaly Score & Alarm Threshold ({threshold:.5f})</b>",
        ),
        row_heights=[0.38, 0.37, 0.25],
    )

    # Subplot 1: Heatmap
    heatmap = go.Heatmap(
        z=f_sq.T,
        x=time_sec,
        y=feature_names,
        colorscale="Inferno",
        colorbar=dict(title="Error Δ²", x=1.01, y=0.82, len=0.35),
        hovertemplate="Time: %{x:.1f}s<br>Sensor: %{y}<br>Recon Error: %{z:.5f}<extra></extra>",
        name="Error Intensity",
    )
    fig_plotly.add_trace(heatmap, row=1, col=1)

    # Subplot 2: Traces for all 16 sensors
    traces_per_sensor = 2
    start_trace_idx = len(fig_plotly.data)

    for i, sname in enumerate(feature_names):
        is_default = (i == top1_idx)
        fig_plotly.add_trace(
            go.Scatter(
                x=time_sec,
                y=f_actual[:, i],
                name=f"Actual: {sname}",
                mode="lines",
                line=dict(color="black", width=1.5),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>Actual {sname}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        fig_plotly.add_trace(
            go.Scatter(
                x=time_sec,
                y=f_recon[:, i],
                name=f"TranAD Recon: {sname}",
                mode="lines",
                line=dict(color="#1f77b4", width=1.5, dash="dash"),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>TranAD Recon {sname}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Subplot 3: Score
    fig_plotly.add_trace(
        go.Scatter(
            x=time_sec,
            y=f_scores,
            name="TranAD Score",
            mode="lines",
            line=dict(color="#8a2be2", width=1.6),
            hovertemplate="Time: %{x:.1f}s<br>Score: %{y:.5f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig_plotly.add_trace(
        go.Scatter(
            x=[time_sec[0], time_sec[-1]],
            y=[threshold, threshold],
            name=f"Threshold ({threshold:.5f})",
            mode="lines",
            line=dict(color="red", width=1.8, dash="dot"),
            hovertemplate=f"Threshold: {threshold:.5f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Fault shading & vertical lines
    if onset_sec is not None:
        for r in [1, 2, 3]:
            fig_plotly.add_vline(x=onset_sec, line_width=2, line_dash="dash", line_color="red", row=r, col=1)
            fig_plotly.add_vrect(x0=onset_sec, x1=end_sec, fillcolor="rgba(255, 100, 0, 0.12)", line_width=0, row=r, col=1)

    # Dropdown Buttons
    buttons = []
    tot_traces = len(fig_plotly.data)
    for i, sname in enumerate(feature_names):
        visibility = [True] * tot_traces
        for j in range(K):
            t1 = start_trace_idx + j * traces_per_sensor
            t2 = t1 + 1
            visibility[t1] = (j == i)
            visibility[t2] = (j == i)

        prefix = "★ " if i == top1_idx else ""
        button = dict(
            label=f"{prefix}{sname}",
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": (
                        f"<b>TranAD Multi-Sensor Anomaly Inspection Graph (16 Raw Channels)</b><br>"
                        f"<span style='font-size:13px; color:#555;'>"
                        f"Flight: {seq_name} | Fault Onset: {onset_sec:.1f}s | "
                        f"Current Sensor: <b>{sname}</b> (Top-1 Root Cause: <b>{top1_sensor}</b>)"
                        f"</span>"
                    )
                },
            ],
        )
        buttons.append(button)

    fig_plotly.update_layout(
        title=dict(
            text=(
                f"<b>TranAD Multi-Sensor Anomaly Inspection Graph (16 Raw Channels)</b><br>"
                f"<span style='font-size:13px; color:#555;'>"
                f"Flight: {seq_name} | Fault Onset: {onset_sec:.1f}s | "
                f"Current Sensor: <b>{top1_sensor}</b> (Top-1 Root Cause)"
                f"</span>"
            ),
            x=0.03,
            y=0.97,
        ),
        updatemenus=[
            dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                active=top1_idx,
                x=0.98,
                xanchor="right",
                y=0.64,
                yanchor="top",
                bgcolor="#ffffff",
                bordercolor="#cccccc",
                font=dict(size=12),
            )
        ],
        hovermode="x unified",
        template="plotly_white",
        height=1000,
        margin=dict(l=60, r=40, t=90, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="right", x=0.98),
    )

    fig_plotly.update_yaxes(title_text="Sensor Name", row=1, col=1, autorange="reversed")
    fig_plotly.update_yaxes(title_text="Normalized Value", row=2, col=1)
    fig_plotly.update_yaxes(title_text="TranAD Score", row=3, col=1)
    fig_plotly.update_xaxes(title_text="Flight Time (seconds)", row=3, col=1)

    fig_plotly.write_html(str(plotly_html_path), include_plotlyjs="cdn")
    print(f"[SUCCESS] Saved TranAD Interactive Plotly Graph to: {plotly_html_path}")


if __name__ == "__main__":
    model_files = sorted(glob.glob(str(CUSTOM_DIR / "pretrained" / "alfa_cmu16" / "best_tranad_*.pt")))
    generate_tranad_plots(model_files[-1])

