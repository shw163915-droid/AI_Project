# -*- coding: utf-8 -*-
"""
Export Interactive Plotly HTML Dashboard for USAD on CMU 16 Raw Physical Channels
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAD_REPO_PATH = PROJECT_ROOT / "USAD"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(USAD_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(USAD_REPO_PATH))

from usad import UsadModel
from run_alfa_usad import AlfaUSADRunner


def build_usad_interactive_dashboard(
    model_path: str,
    output_html_path: str = None,
    target_flight_pattern: str = "elevator_failure",
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
    alpha = train_config["alpha"]
    beta = train_config["beta"]

    # Compute reconstruction & scores
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

    test_scores = np.concatenate(test_scores_list)
    recon_all = np.vstack(recon_list)

    # Reshape flattened windows into [N, W, K] to get channel-wise reconstruction error
    actual_windows = runner.test_data.reshape(-1, win_size, K)
    recon_windows = recon_all.reshape(-1, win_size, K)

    # Focus on the newest timestep (last index in window: -1)
    actual_newest = actual_windows[:, -1, :]  # [N, K]
    recon_newest = recon_windows[:, -1, :]    # [N, K]
    channel_errors = np.abs(actual_newest - recon_newest)  # [N, K]

    # Find threshold
    labels = runner.test_labels.astype(int)
    th_candidates = np.percentile(test_scores, np.linspace(50, 99.5, 300))
    best_f1, best_th = -1.0, 0.0
    for th in th_candidates:
        pred = (test_scores > th).astype(int)
        tp = np.sum((pred == 1) & (labels == 1))
        fp = np.sum((pred == 1) & (labels == 0))
        fn = np.sum((pred == 0) & (labels == 1))
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_th = th

    # Target flight
    target_rec = None
    for rec in runner.test_flight_records:
        if target_flight_pattern in rec["sequence"]:
            target_rec = rec
            break
    if target_rec is None and len(runner.test_flight_records) > 0:
        target_rec = runner.test_flight_records[0]

    s, e = target_rec["win_start"], target_rec["win_end"]
    seq_name = target_rec["sequence"]
    f_scores = test_scores[s:e]
    f_gt = labels[s:e]
    time_sec = np.arange(e - s) * 0.1

    fault_indices = np.where(f_gt == 1)[0]
    onset_sec = float(time_sec[fault_indices[0]]) if len(fault_indices) > 0 else None
    end_sec = float(time_sec[-1])

    # Sliced matrices
    flight_channel_err = channel_errors[s:e].T  # [K, T]
    flight_actual = actual_newest[s:e]          # [T, K]
    flight_recon = recon_newest[s:e]            # [T, K]

    # Find Top-1 Root Cause channel (highest error jump after fault)
    if onset_sec is not None:
        fault_means = np.mean(flight_channel_err[:, fault_indices], axis=1)
        top1_idx = int(np.argmax(fault_means))
    else:
        top1_idx = 0
    top1_sensor = runner.feature_map[top1_idx]

    print(f"[INFO] Target flight: {seq_name}")
    print(f"[INFO] Fault onset: {onset_sec:.1f}s | Top-1 Root Cause Channel: {top1_sensor}")

    # Build Plotly Subplots
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            f"<b>1. USAD 16 Raw Channels Reconstruction Error Heatmap</b>",
            f"<b>2. Multi-Sensor Actual vs USAD Reconstructed (Interactive Dropdown Selector)</b>",
            f"<b>3. Integrated Real-time USAD Anomaly Score & Alarm Threshold ({best_th:.4f})</b>",
        ),
        row_heights=[0.38, 0.37, 0.25],
    )

    # 1. Heatmap
    heatmap = go.Heatmap(
        z=flight_channel_err,
        x=time_sec,
        y=runner.feature_map,
        colorscale="Inferno",
        colorbar=dict(title="|Error|", x=1.01, y=0.82, len=0.35),
        hovertemplate="Time: %{x:.1f}s<br>Sensor: %{y}<br>Recon Error: %{z:.3f}<extra></extra>",
        name="Error Intensity",
    )
    fig.add_trace(heatmap, row=1, col=1)

    # 2. 16 Sensors Actual vs Recon traces
    traces_per_sensor = 2
    start_trace_idx = len(fig.data)

    for i, sensor_name in enumerate(runner.feature_map):
        is_default = (i == top1_idx)
        # Actual
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=flight_actual[:, i],
                name=f"Actual: {sensor_name}",
                mode="lines",
                line=dict(color="black", width=1.5),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>Actual {sensor_name}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        # USAD Recon
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=flight_recon[:, i],
                name=f"USAD Recon: {sensor_name}",
                mode="lines",
                line=dict(color="#1f77b4", width=1.5, dash="dash"),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>USAD Recon {sensor_name}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # 3. Overall Anomaly Score & Threshold
    fig.add_trace(
        go.Scatter(
            x=time_sec,
            y=f_scores,
            name="USAD Score (AE1+AE2)",
            mode="lines",
            line=dict(color="#8a2be2", width=1.6),
            hovertemplate="Time: %{x:.1f}s<br>Score: %{y:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[time_sec[0], time_sec[-1]],
            y=[best_th, best_th],
            name=f"Threshold ({best_th:.4f})",
            mode="lines",
            line=dict(color="red", width=1.8, dash="dot"),
            hovertemplate=f"Threshold: {best_th:.4f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Alarm scatter
    alarm_mask = f_scores > best_th
    if np.any(alarm_mask):
        fig.add_trace(
            go.Scatter(
                x=time_sec[alarm_mask],
                y=f_scores[alarm_mask],
                name="Alarm Triggered",
                mode="markers",
                marker=dict(color="red", size=4, symbol="circle"),
                hovertemplate="Alarm Time: %{x:.1f}s<br>Score: %{y:.4f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Fault shading & lines
    if onset_sec is not None:
        for r in [1, 2, 3]:
            fig.add_vline(x=onset_sec, line_width=2, line_dash="dash", line_color="red", row=r, col=1)
            fig.add_vrect(x0=onset_sec, x1=end_sec, fillcolor="rgba(255, 100, 0, 0.12)", line_width=0, row=r, col=1)

    # Dropdown menu
    buttons = []
    total_traces = len(fig.data)
    for i, sensor_name in enumerate(runner.feature_map):
        visibility = [True] * total_traces
        for j in range(K):
            t1 = start_trace_idx + j * traces_per_sensor
            t2 = t1 + 1
            visibility[t1] = (j == i)
            visibility[t2] = (j == i)

        prefix = "★ " if i == top1_idx else ""
        button = dict(
            label=f"{prefix}{sensor_name}",
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": (
                        f"<b>USAD Multi-Sensor Anomaly Inspection Dashboard (16 Raw Channels)</b><br>"
                        f"<span style='font-size:13px; color:#555;'>"
                        f"Flight: {seq_name} | Fault Onset: {onset_sec:.1f}s | "
                        f"Current Sensor: <b>{sensor_name}</b> (Top-1 Root Cause: <b>{top1_sensor}</b>)"
                        f"</span>"
                    )
                },
            ],
        )
        buttons.append(button)

    fig.update_layout(
        title=dict(
            text=(
                f"<b>USAD Multi-Sensor Anomaly Inspection Dashboard (16 Raw Channels)</b><br>"
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
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="right",
            x=0.98,
            font=dict(size=11),
        ),
    )

    fig.update_yaxes(title_text="Sensor Name", row=1, col=1, autorange="reversed")
    fig.update_yaxes(title_text="Normalized Value", row=2, col=1)
    fig.update_yaxes(title_text="USAD Score", row=3, col=1)
    fig.update_xaxes(title_text="Flight Time (seconds)", row=3, col=1)

    if output_html_path is None:
        res_dir = CUSTOM_DIR / "results"
        res_dir.mkdir(parents=True, exist_ok=True)
        output_html_path = res_dir / "alfa_usad_cmu16_interactive.html"

    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    print(f"[SUCCESS] USAD Interactive Dashboard saved to: {output_html_path}")
    return str(output_html_path)


if __name__ == "__main__":
    import glob
    model_files = sorted(glob.glob(str(CUSTOM_DIR / "pretrained" / "alfa_cmu16" / "best_*.pt")))
    if not model_files:
        print("[ERROR] No trained USAD model found.")
        sys.exit(1)
    latest_model = model_files[-1]
    build_usad_interactive_dashboard(latest_model)

