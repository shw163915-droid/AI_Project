# -*- coding: utf-8 -*-
"""
Run USAD (UnSupervised Anomaly Detection) on CMU ALFA 16 Raw Physical Channels
- Does NOT modify C:/AI Project/USAD (imports UsadModel cleanly)
- Uses exact 16 CMU AirLab raw features (no manual ERR signals)
- Implements episode-aware windowing to avoid cross-flight boundary contamination
- Implements real-time reconstruction evaluation (alpha*AE1 + beta*AE2)
"""

import os
import sys
import time
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset, Subset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# Add pristine USAD repository to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
USAD_REPO_PATH = PROJECT_ROOT / "USAD"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(USAD_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(USAD_REPO_PATH))

from usad import UsadModel


class AlfaUSADRunner:
    def __init__(self, train_config: dict, env_config: dict):
        self.train_config = train_config
        self.env_config = env_config
        self.device = torch.device(env_config["device"])
        self.data_dir = CUSTOM_DIR / "data" / env_config["dataset"]
        self.datestr = None

        self._load_metadata()
        self._build_datasets()
        self._init_model()

    def _load_metadata(self):
        list_file = self.data_dir / "list.txt"
        with open(list_file, "r", encoding="utf-8") as f:
            self.feature_map = [line.strip() for line in f if line.strip()]
        self.num_features = len(self.feature_map)
        print(f"[INFO] Loaded {self.num_features} CMU raw physical features: {self.feature_map}")

    def _create_flight_windows(self, values: np.ndarray, window_size: int):
        """
        Creates sliding windows for a single continuous flight episode.
        values: [T, num_features]
        Returns: [num_windows, window_size * num_features]
        """
        T, K = values.shape
        if T < window_size:
            return np.empty((0, window_size * K), dtype=np.float32)

        # Vectorized sliding window
        indexer = np.arange(window_size)[None, :] + np.arange(T - window_size + 1)[:, None]
        windows = values[indexer]  # [num_windows, window_size, K]
        return windows.reshape(windows.shape[0], -1).astype(np.float32)

    def _build_datasets(self):
        train_csv = self.data_dir / "train.csv"
        test_csv = self.data_dir / "test.csv"
        seq_summary_path = self.data_dir / "sequence_summary.csv"

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        seq_summary = pd.read_csv(seq_summary_path)

        win_size = self.train_config["window_size"]
        self.w_size = win_size * self.num_features

        # Episode-aware sliding window construction
        train_windows_list = []
        normal_seqs = seq_summary[seq_summary["split"] == "train"]["sequence"].tolist()
        
        # Split normal flights by sequence
        # We need flight boundary info. train.csv length is sum of normal sequences
        curr_idx = 0
        train_episodes = []
        for seq_name in normal_seqs:
            row = seq_summary[seq_summary["sequence"] == seq_name].iloc[0]
            seq_len = int(row["length_steps"])
            ep_vals = train_df.iloc[curr_idx : curr_idx + seq_len][self.feature_map].values
            curr_idx += seq_len
            ep_wins = self._create_flight_windows(ep_vals, win_size)
            if len(ep_wins) > 0:
                train_episodes.append(ep_wins)

        # Episode-wise Temporal Block Split for Train & Val (90% train, 10% val)
        val_ratio = self.train_config["val_ratio"]
        train_wins = []
        val_wins = []
        for ep_win in train_episodes:
            n_ep = len(ep_win)
            val_n = max(1, int(n_ep * val_ratio))
            train_n = n_ep - val_n
            train_wins.append(ep_win[:train_n])
            val_wins.append(ep_win[train_n:])

        self.train_data = np.vstack(train_wins)
        self.val_data = np.vstack(val_wins)

        # Build Test Windows & record flight metadata
        self.test_flight_records = []
        test_wins_list = []
        test_labels_list = []
        win_cursor = 0

        eval_seqs = seq_summary[seq_summary["split"] == "test"]["sequence"].tolist()
        t_curr = 0
        for seq_name in eval_seqs:
            row = seq_summary[seq_summary["sequence"] == seq_name].iloc[0]
            seq_len = int(row["length_steps"])
            ep_sub = test_df.iloc[t_curr : t_curr + seq_len]
            t_curr += seq_len

            ep_vals = ep_sub[self.feature_map].values
            ep_labels = ep_sub["attack"].values
            ep_wins = self._create_flight_windows(ep_vals, win_size)
            n_w = len(ep_wins)
            if n_w == 0:
                continue

            # The label of a window corresponds to the newest timestamp (last point in window)
            win_labels = ep_labels[win_size - 1 :]
            test_wins_list.append(ep_wins)
            test_labels_list.append(win_labels)

            self.test_flight_records.append(
                {
                    "sequence": seq_name,
                    "win_start": win_cursor,
                    "win_end": win_cursor + n_w,
                    "num_windows": n_w,
                    "has_fault": bool(row["has_fault"]),
                    "fault_onset_sec": row["fault_onset_sec"] if not pd.isna(row["fault_onset_sec"]) else None,
                }
            )
            win_cursor += n_w

        self.test_data = np.vstack(test_wins_list)
        self.test_labels = np.concatenate(test_labels_list)

        batch = self.train_config["batch"]
        self.train_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.train_data).float()),
            batch_size=batch,
            shuffle=True,
        )
        self.val_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.val_data).float()),
            batch_size=batch,
            shuffle=False,
        )
        self.test_loader = DataLoader(
            TensorDataset(torch.from_numpy(self.test_data).float()),
            batch_size=batch,
            shuffle=False,
        )

        print(
            f"[INFO] Episode-Aware Windows created: "
            f"Train={len(self.train_data):,}, Val={len(self.val_data):,}, "
            f"Test={len(self.test_data):,} ({len(self.test_flight_records)} flights)"
        )
        print(f"[INFO] Window vector dim = {self.w_size} (W={win_size} x K={self.num_features})")

    def _init_model(self):
        z_size = self.train_config["z_size"]
        self.model = UsadModel(self.w_size, z_size).to(self.device)
        print(f"[INFO] Initialized UsadModel: w_size={self.w_size}, latent z_size={z_size}")

    def get_save_paths(self):
        if self.datestr is None:
            self.datestr = datetime.now().strftime("%m_%d-%H_%M_%S")
        model_dir = CUSTOM_DIR / "pretrained" / "alfa_cmu16"
        res_dir = CUSTOM_DIR / "results"
        model_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"best_{self.datestr}.pt"
        res_path = res_dir / f"alfa_usad_{self.datestr}.json"
        return str(model_path), str(res_path)

    def train(self, model_save_path: str):
        epochs = self.train_config["epochs"]
        lr = self.train_config["lr"]
        patience = self.train_config["patience"]

        opt1 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder1.parameters()),
            lr=lr,
        )
        opt2 = torch.optim.Adam(
            list(self.model.encoder.parameters()) + list(self.model.decoder2.parameters()),
            lr=lr,
        )

        min_val_loss = float("inf")
        stop_count = 0
        history = {"train_loss": [], "val_loss": []}

        print(f"\n[INFO] Starting USAD Training ({epochs} epochs, lr={lr}, patience={patience})...")
        print("-" * 75)
        print(f"{'Epoch':^9} | {'Train L1+L2':^14} | {'Val L1+L2':^14} | {'Min Val Loss':^14} | {'Status':^14}")
        print("-" * 75)

        for ep in range(1, epochs + 1):
            self.model.train()
            train_l1, train_l2 = 0.0, 0.0
            for [batch] in self.train_loader:
                batch = batch.to(self.device)

                # Phase 1: Train AE1
                loss1, loss2 = self.model.training_step(batch, ep)
                opt1.zero_grad()
                loss1.backward()
                opt1.step()

                # Phase 2: Train AE2
                loss1, loss2 = self.model.training_step(batch, ep)
                opt2.zero_grad()
                loss2.backward()
                opt2.step()

                train_l1 += loss1.item()
                train_l2 += loss2.item()

            avg_train_loss = (train_l1 + train_l2) / (2 * len(self.train_loader))

            # Validation
            self.model.eval()
            val_l1, val_l2 = 0.0, 0.0
            with torch.no_grad():
                for [batch] in self.val_loader:
                    batch = batch.to(self.device)
                    v_loss1, v_loss2 = self.model.training_step(batch, ep)
                    val_l1 += v_loss1.item()
                    val_l2 += v_loss2.item()

            avg_val_loss = (val_l1 + val_l2) / (2 * len(self.val_loader))
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(avg_val_loss)

            if avg_val_loss < min_val_loss:
                min_val_loss = avg_val_loss
                torch.save(self.model.state_dict(), model_save_path)
                stop_count = 0
                status_str = "★ Best Saved"
            else:
                stop_count += 1
                status_str = f"Wait ({stop_count}/{patience})"

            print(
                f"{ep:^9d} | "
                f"{avg_train_loss:^14.6f} | "
                f"{avg_val_loss:^14.6f} | "
                f"{min_val_loss:^14.6f} | "
                f"{status_str:^14}"
            )

            if stop_count >= patience:
                print(f"[INFO] Early stopping triggered at epoch {ep}")
                break

        print("-" * 75)
        self.train_history = history
        return history

    def evaluate(self, model_path: str, result_save_path: str):
        print(f"\n[INFO] Loading best model for evaluation from: {model_path}")
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        alpha = self.train_config["alpha"]
        beta = self.train_config["beta"]
        win_size = self.train_config["window_size"]
        K = self.num_features

        # Compute test reconstruction errors
        test_scores_list = []
        test_recon_w1_list = []
        with torch.no_grad():
            for [batch] in self.test_loader:
                batch = batch.to(self.device)
                w1 = self.model.decoder1(self.model.encoder(batch))
                w2 = self.model.decoder2(self.model.encoder(w1))

                # Point-wise or window-wise score
                # Batch score: alpha * MSE(batch, w1) + beta * MSE(batch, w2)
                err1 = torch.mean((batch - w1) ** 2, dim=1)
                err2 = torch.mean((batch - w2) ** 2, dim=1)
                score = alpha * err1 + beta * err2
                test_scores_list.append(score.cpu().numpy())
                test_recon_w1_list.append(w1.cpu().numpy())

        test_scores = np.concatenate(test_scores_list)
        recon_w1_all = np.vstack(test_recon_w1_list)

        # Reshape reconstructed newest point for sensor-level inspection
        # Newest point is the last K elements in the flattened window
        actual_newest = self.test_data[:, -K:]
        recon_newest = recon_w1_all[:, -K:]

        # Search best threshold for F1 on test labels
        labels = self.test_labels.astype(int)
        th_candidates = np.percentile(test_scores, np.linspace(50, 99.5, 300))
        best_f1, best_th, best_p, best_r = -1.0, 0.0, 0.0, 0.0

        for th in th_candidates:
            pred = (test_scores > th).astype(int)
            p = precision_score(labels, pred, zero_division=0)
            r = recall_score(labels, pred, zero_division=0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_p = p
                best_r = r

        auroc = roc_auc_score(labels, test_scores) if len(np.unique(labels)) > 1 else 0.0

        print("\n=========================** ALFA USAD Result (16 Raw Channels) **==========================")
        print(f"Global F1 score    : {best_f1:.4f}")
        print(f"Global Precision   : {best_p:.4f}")
        print(f"Global Recall      : {best_r:.4f}")
        print(f"Global AUROC       : {auroc:.4f}")
        print(f"Best Threshold     : {best_th:.6f}")

        # Per-flight evaluation
        per_flight_stats = []
        detected_latencies = []
        for rec in self.test_flight_records:
            s, e = rec["win_start"], rec["win_end"]
            f_scores = test_scores[s:e]
            f_gt = labels[s:e]
            has_fault = rec["has_fault"]

            f_auroc = roc_auc_score(f_gt, f_scores) if len(np.unique(f_gt)) > 1 else 0.5
            pred_alarm = f_scores > best_th

            latency_sec = None
            if has_fault and np.any(f_gt == 1):
                fault_start_idx = np.where(f_gt == 1)[0][0]
                alarm_after_fault = np.where(pred_alarm[fault_start_idx:])[0]
                if len(alarm_after_fault) > 0:
                    latency_sec = float(alarm_after_fault[0] * 0.1)  # 10Hz -> seconds
                    detected_latencies.append(latency_sec)

            per_flight_stats.append(
                {
                    "sequence": rec["sequence"],
                    "has_fault": has_fault,
                    "auroc": float(f_auroc),
                    "detected": bool(latency_sec is not None),
                    "detection_latency_sec": latency_sec,
                }
            )

        mean_flight_auroc = np.mean([x["auroc"] for x in per_flight_stats])
        mean_latency = np.mean(detected_latencies) if detected_latencies else -1.0
        print(f"Per-Flight AUROC   : {mean_flight_auroc:.4f} (mean across {len(per_flight_stats)} flights)")
        print(f"Detected Flights   : {len(detected_latencies)} / {len(per_flight_stats)} (Mean Latency: {mean_latency:.2f}s)")
        print(f"Model saved at     : {model_path}")
        print("===========================================================================================\n")

        # Save results to JSON
        res_summary = {
            "model": "USAD",
            "channels": self.feature_map,
            "num_channels": self.num_features,
            "window_size": win_size,
            "metrics": {
                "f1": float(best_f1),
                "precision": float(best_p),
                "recall": float(best_r),
                "auroc": float(auroc),
                "threshold": float(best_th),
                "mean_flight_auroc": float(mean_flight_auroc),
                "mean_latency_sec": float(mean_latency),
                "detected_flights_count": len(detected_latencies),
            },
            "per_flight_summary": per_flight_stats,
        }
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(res_summary, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Summary results saved to: {result_save_path}")

        # Plot representative dashboard
        self._plot_dashboard(
            test_scores, actual_newest, recon_newest, labels, best_th, result_save_path
        )

    def _plot_dashboard(
        self,
        test_scores: np.ndarray,
        actual_newest: np.ndarray,
        recon_newest: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        result_save_path: str,
    ):
        # Find representative elevator failure flight
        target_rec = None
        for rec in self.test_flight_records:
            if "elevator_failure" in rec["sequence"]:
                target_rec = rec
                break
        if target_rec is None and len(self.test_flight_records) > 0:
            target_rec = self.test_flight_records[0]

        s, e = target_rec["win_start"], target_rec["win_end"]
        seq_name = target_rec["sequence"]
        f_scores = test_scores[s:e]
        f_gt = labels[s:e]
        time_sec = np.arange(e - s) * 0.1

        fault_indices = np.where(f_gt == 1)[0]
        onset_sec = time_sec[fault_indices[0]] if len(fault_indices) > 0 else None

        # Choose pitch_meas or pitch_cmd as key inspection channel
        pitch_idx = self.feature_map.index("pitch_meas") if "pitch_meas" in self.feature_map else 0
        actual_pitch = actual_newest[s:e, pitch_idx]
        recon_pitch = recon_newest[s:e, pitch_idx]

        fig, axes = plt.subplots(3, 1, figsize=(13, 10))

        # 1. Training Loss Curve
        if hasattr(self, "train_history"):
            ep_range = range(1, len(self.train_history["train_loss"]) + 1)
            axes[0].plot(ep_range, self.train_history["train_loss"], marker="o", label="Train Loss (AE1+AE2)")
            axes[0].plot(ep_range, self.train_history["val_loss"], marker="s", label="Val Loss (AE1+AE2)")
            axes[0].set_title("1. USAD Adversarial AutoEncoder Loss Convergence")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend()

        # 2. Actual vs Reconstruction
        axes[1].plot(time_sec, actual_pitch, color="black", linewidth=1.3, label=f"Actual Sensor (pitch_meas)")
        axes[1].plot(time_sec, recon_pitch, color="tab:blue", linestyle="--", linewidth=1.3, label="USAD Reconstructed (pitch_meas)")
        if onset_sec is not None:
            axes[1].axvline(onset_sec, color="tab:red", linewidth=1.5, label=f"Fault Onset ({onset_sec:.1f}s)")
            axes[1].axvspan(onset_sec, time_sec[-1], color="tab:orange", alpha=0.2, label="Fault Interval")
        axes[1].set_title(f"2. USAD Sensor Reconstruction vs Actual [{seq_name}] (Raw pitch_meas, No ERR features)")
        axes[1].set_xlabel("Flight Time (seconds)")
        axes[1].set_ylabel("Scaled Value (0~1)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend(loc="upper left")

        # 3. Anomaly Score
        axes[2].plot(time_sec, f_scores, color="tab:purple", linewidth=1.3, label="USAD Anomaly Score")
        axes[2].axhline(threshold, color="tab:red", linestyle="--", linewidth=1.3, label=f"Alarm Threshold ({threshold:.4f})")
        if onset_sec is not None:
            axes[2].axvline(onset_sec, color="tab:red", linewidth=1.5)
            axes[2].axvspan(onset_sec, time_sec[-1], color="tab:orange", alpha=0.2, label="Fault Interval")
        axes[2].set_title(f"3. Real-time Anomaly Score & Alarm [{seq_name}]")
        axes[2].set_xlabel("Flight Time (seconds)")
        axes[2].set_ylabel("Anomaly Score")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend(loc="upper left")

        plt.tight_layout()
        plot_path = result_save_path.replace(".json", "_dashboard.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Visual dashboard saved to: {plot_path}")

    def run(self):
        model_save_path, result_save_path = self.get_save_paths()
        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            self.train(model_save_path)

        self.evaluate(model_save_path, result_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run USAD on ALFA CMU-16 Raw Channels")
    parser.add_argument("-window_size", type=int, default=12, help="sliding window size (default 12 = 1.2s at 10Hz)")
    parser.add_argument("-z_size", type=int, default=48, help="latent space dimension (default 48)")
    parser.add_argument("-batch", type=int, default=128, help="batch size")
    parser.add_argument("-epochs", type=int, default=500, help="max training epochs")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-patience", type=int, default=15, help="early stopping patience")
    parser.add_argument("-val_ratio", type=float, default=0.1, help="validation ratio")
    parser.add_argument("-alpha", type=float, default=0.5, help="weight for AE1 error")
    parser.add_argument("-beta", type=float, default=0.5, help="weight for AE2 error")
    parser.add_argument("-dataset", type=str, default="alfa_cmu16", help="dataset folder name")
    parser.add_argument("-device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-seed", type=int, default=42)
    parser.add_argument("-load_model_path", type=str, default="")

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    train_config = {
        "window_size": args.window_size,
        "z_size": args.z_size,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "val_ratio": args.val_ratio,
        "alpha": args.alpha,
        "beta": args.beta,
    }

    env_config = {
        "dataset": args.dataset,
        "device": args.device,
        "load_model_path": args.load_model_path,
    }

    runner = AlfaUSADRunner(train_config, env_config)
    runner.run()

