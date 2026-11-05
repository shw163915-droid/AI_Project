# -*- coding: utf-8 -*-
"""
Interactive HTML Dashboard Generator for CMU ALFA Multi-Sensor Anomaly Detection
Uses Plotly to visualize:
 1. 22-channel sensor anomaly heatmap (sensor vs time)
 2. Interactive dropdown selector for all 22 sensors (Actual vs GDN Predicted)
 3. Overall real-time anomaly score and threshold trigger
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CUSTOM_DIR = Path(__file__).resolve().parent
GDN_REPO_PATH = PROJECT_ROOT / "GDN"

if str(GDN_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(GDN_REPO_PATH))
if str(CUSTOM_DIR) not in sys.path:
    sys.path.insert(0, str(CUSTOM_DIR))

from run_alfa_gdn import AlfaGDNRunner, test, get_best_performance_data


def build_interactive_dashboard(
    model_path: str,
    output_html_path: str = None,
    target_flight_pattern: str = "elevator_failure",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    print(f"[INFO] Initializing GDN Runner with checkpoint: {model_path}")
    
    train_config = {
        "batch": 32,
        "epoch": 500,
        "slide_win": 15,
        "dim": 128,
        "slide_stride": 1,
        "comment": "eval_interactive",
        "seed": 42,
        "out_layer_num": 2,
        "out_layer_inter_dim": 128,
        "decay": 0,
        "val_ratio": 0.1,
        "topk": 12,
    }

    env_config = {
        "save_path": "alfa",
        "dataset": "alfa",
        "report": "best",
        "device": device,
        "load_model_path": model_path,
        "episode_window": True,
        "ewma_alpha": 0.3,
    }

    runner = AlfaGDNRunner(train_config, env_config)
    runner.model.load_state_dict(torch.load(model_path, map_location=device))
    runner.model.eval()

    print("[INFO] Evaluating test and validation dataloaders...")
    _, test_result = test(runner.model, runner.test_dataloader)
    _, val_result = test(runner.model, runner.val_dataloader)

    np_test_result = np.array(test_result)
    test_labels = np_test_result[2, :, 0].tolist()
    feature_map = runner.feature_map
    num_features = len(feature_map)

    # Compute EWMA anomaly scores
    test_scores, normal_scores = runner.compute_ewma_err_scores(
        test_result, val_result, alpha=env_config["ewma_alpha"]
    )
    top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
    threshold = float(top1_best_info[4])
    max_sensor_scores = np.max(test_scores, axis=0)

    # Find target flight episode
    target_rec = None
    for rec in runner.test_flight_records:
        if target_flight_pattern in rec["sequence"]:
            target_rec = rec
            break
    if target_rec is None and len(runner.test_flight_records) > 0:
        target_rec = runner.test_flight_records[0]

    s, e = target_rec["win_start"], target_rec["win_end"]
    seq_name = target_rec["sequence"]
    failure_type = target_rec.get("failure_type", "elevator_fault" if "elevator" in seq_name else "flight_fault")
    seq_gt = np.array(test_labels[s:e], dtype=int)
    time_sec = np.arange(e - s) * 0.1  # 10Hz to seconds

    fault_indices = np.where(seq_gt == 1)[0]
    onset_sec = float(time_sec[fault_indices[0]]) if len(fault_indices) > 0 else None
    end_sec = float(time_sec[-1])

    # Slice sequence data
    seq_scores_matrix = test_scores[:, s:e]  # [num_features, T]
    seq_max_scores = max_sensor_scores[s:e]
    pred_matrix = np_test_result[0, s:e, :]  # [T, num_features]
    actual_matrix = np_test_result[1, s:e, :]  # [T, num_features]

    # Find Top-1 Root Cause sensor during fault
    if onset_sec is not None:
        fault_means = np.mean(seq_scores_matrix[:, fault_indices], axis=1)
        top1_sensor_idx = int(np.argmax(fault_means))
    else:
        top1_sensor_idx = 0
    top1_sensor_name = feature_map[top1_sensor_idx]

    print(f"[INFO] Target flight: {seq_name}")
    print(f"[INFO] Fault onset: {onset_sec:.1f}s | Top-1 root-cause: {top1_sensor_name}")

    # Build Plotly Subplots
    # Row 1: Heatmap (22 sensors x time)
    # Row 2: Selected Sensor (Actual vs Predicted)
    # Row 3: Overall Anomaly Score & Threshold
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.07,
        subplot_titles=(
            f"<b>1. All 22-Channel Sensor Error Heatmap (Anomaly Intensity Across Flight Time)</b>",
            f"<b>2. Multi-Sensor Actual vs GDN Predicted Signal (Interactive Dropdown Selector)</b>",
            f"<b>3. Integrated Real-time Anomaly Score & Alarm Threshold ({threshold:.2f})</b>",
        ),
        row_heights=[0.38, 0.37, 0.25],
    )

    # ----------------- Subplot 1: Heatmap -----------------
    heatmap_trace = go.Heatmap(
        z=seq_scores_matrix,
        x=time_sec,
        y=feature_map,
        colorscale="Inferno",
        colorbar=dict(title="Score", x=1.01, y=0.82, len=0.35),
        hoverongaps=False,
        hovertemplate="Time: %{x:.1f}s<br>Sensor: %{y}<br>Error Score: %{z:.3f}<extra></extra>",
        name="Error Intensity",
    )
    fig.add_trace(heatmap_trace, row=1, col=1)

    # ----------------- Subplot 2: 22 Sensors Actual vs Pred -----------------
    # For each sensor, we add 2 traces: Actual and Predicted
    # Initially, only top1_sensor_idx traces will be visible=True
    traces_per_sensor = 2
    start_trace_idx = len(fig.data)

    for i, sensor_name in enumerate(feature_map):
        is_default = (i == top1_sensor_idx)
        # Actual trace
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=actual_matrix[:, i],
                name=f"Actual: {sensor_name}",
                mode="lines",
                line=dict(color="black", width=1.5),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>Actual {sensor_name}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        # Pred trace
        fig.add_trace(
            go.Scatter(
                x=time_sec,
                y=pred_matrix[:, i],
                name=f"GDN Pred: {sensor_name}",
                mode="lines",
                line=dict(color="#1f77b4", width=1.5, dash="dash"),
                visible=is_default,
                hovertemplate=f"Time: %{{x:.1f}}s<br>Pred {sensor_name}: %{{y:.3f}}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # ----------------- Subplot 3: Anomaly Score & Alarm -----------------
    fig.add_trace(
        go.Scatter(
            x=time_sec,
            y=seq_max_scores,
            name="Max Anomaly Score",
            mode="lines",
            line=dict(color="#8a2be2", width=1.6),
            hovertemplate="Time: %{x:.1f}s<br>Anomaly Score: %{y:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Alarm Threshold line
    fig.add_trace(
        go.Scatter(
            x=[time_sec[0], time_sec[-1]],
            y=[threshold, threshold],
            name=f"Alarm Threshold ({threshold:.2f})",
            mode="lines",
            line=dict(color="red", width=1.8, dash="dot"),
            hovertemplate=f"Threshold: {threshold:.2f}<extra></extra>",
        ),
        row=3,
        col=1,
    )

    # Alarm Triggers (scatter points above threshold)
    alarm_mask = seq_max_scores > threshold
    if np.any(alarm_mask):
        fig.add_trace(
            go.Scatter(
                x=time_sec[alarm_mask],
                y=seq_max_scores[alarm_mask],
                name="Alarm Triggered",
                mode="markers",
                marker=dict(color="red", size=4, symbol="circle"),
                hovertemplate="Alarm Time: %{x:.1f}s<br>Score: %{y:.2f}<extra></extra>",
            ),
            row=3,
            col=1,
        )

    # Add Fault Shading & Onset Vertical Lines across all subplots
    if onset_sec is not None:
        for r in [1, 2, 3]:
            fig.add_vline(
                x=onset_sec,
                line_width=2,
                line_dash="dash",
                line_color="red",
                row=r,
                col=1,
            )
            fig.add_vrect(
                x0=onset_sec,
                x1=end_sec,
                fillcolor="rgba(255, 100, 0, 0.12)",
                line_width=0,
                row=r,
                col=1,
            )

    # Build Dropdown Menu for all 22 sensors
    buttons = []
    total_traces = len(fig.data)

    for i, sensor_name in enumerate(feature_map):
        visibility = [True] * total_traces  # base
        # Only modify the 22 sensor traces in row 2
        for j in range(num_features):
            t_idx1 = start_trace_idx + j * traces_per_sensor
            t_idx2 = t_idx1 + 1
            visibility[t_idx1] = (j == i)
            visibility[t_idx2] = (j == i)

        prefix = "★ " if i == top1_sensor_idx else ""
        button = dict(
            label=f"{prefix}{sensor_name}",
            method="update",
            args=[
                {"visible": visibility},
                {
                    "title": (
                        f"<b>GDN Multi-Sensor Anomaly Inspection Dashboard</b><br>"
                        f"<span style='font-size:13px; color:#555;'>"
                        f"Flight: {seq_name} | Fault: {failure_type} | Onset: {onset_sec:.1f}s | "
                        f"Current Sensor: <b>{sensor_name}</b> (Top-1 Root Cause: <b>{top1_sensor_name}</b>)"
                        f"</span>"
                    )
                },
            ],
        )
        buttons.append(button)

    # Update Layout
    fig.update_layout(
        title=dict(
            text=(
                f"<b>GDN Multi-Sensor Anomaly Inspection Dashboard</b><br>"
                f"<span style='font-size:13px; color:#555;'>"
                f"Flight: {seq_name} | Fault: {failure_type} | Onset: {onset_sec:.1f}s | "
                f"Current Sensor: <b>{top1_sensor_name}</b> (Top-1 Root Cause)"
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
                active=top1_sensor_idx,
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
    fig.update_yaxes(title_text="Anomaly Score", row=3, col=1)
    fig.update_xaxes(title_text="Flight Time (seconds)", row=3, col=1)

    # Annotation for Dropdown
    fig.add_annotation(
        text="<b>Select Sensor:</b>",
        x=0.88,
        y=0.64,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=12, color="#333"),
        xanchor="right",
        yanchor="middle",
    )

    if output_html_path is None:
        results_dir = CUSTOM_DIR / "results"
        results_dir.mkdir(exist_ok=True, parents=True)
        output_html_path = results_dir / f"alfa_all_sensors_interactive_{Path(model_path).stem}.html"

    fig.write_html(str(output_html_path), include_plotlyjs="cdn")
    print(f"\n[SUCCESS] Interactive Multi-Sensor Dashboard saved to: {output_html_path}")
    return str(output_html_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export Interactive Plotly Dashboard for ALFA GDN")
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(CUSTOM_DIR / "pretrained" / "alfa" / "best_11_05-22_45_41.pt"),
        help="Path to trained GDN checkpoint",
    )
    parser.add_argument(
        "--output_html",
        type=str,
        default=None,
        help="Output HTML file path",
    )
    parser.add_argument(
        "--target_flight",
        type=str,
        default="elevator_failure",
        help="Target flight pattern name",
    )
    args = parser.parse_args()

    build_interactive_dashboard(
        model_path=args.model_path,
        output_html_path=args.output_html,
        target_flight_pattern=args.target_flight,
    )

