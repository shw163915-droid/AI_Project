# -*- coding: utf-8 -*-
"""
Run TranAD (Deep Transformer Networks for Anomaly Detection, VLDB 2022)
on CMU ALFA 16 Raw Physical Channels
- Pristine C:/AI Project/TranAD is imported without modifying a single line
- 16 pure CMU physical channels (no artificial ERR features)
- Multi-Head Self-Attention + Adversarial Self-Conditioning
- Episode-aware sliding windows for multi-flight telemetry
"""

import os
import sys
import time
import json
import random
import argparse
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

# 1. Mock DGL to prevent dependency error without touching TranAD repo
sys.modules['dgl'] = MagicMock()
sys.modules['dgl.nn'] = MagicMock()

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

# 2. Add pristine TranAD repo to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRANAD_REPO = PROJECT_ROOT / "TranAD"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(TRANAD_REPO) not in sys.path:
    sys.path.insert(0, str(TRANAD_REPO))

# PyTorch 2.x compatibility adapter for TranAD (keeps TranAD repository completely untouched)
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


class FlightWindowDataset(Dataset):
    def __init__(self, windows_np: np.ndarray):
        """
        windows_np: [N, W, K]
        """
        self.windows = torch.tensor(windows_np, dtype=torch.float32)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        # TranAD expects:
        # src: [W, K]
        # tgt: [1, K] (newest point)
        w = self.windows[idx] # [W, K]
        return w, w[-1:, :]


class AlfaTranADRunner:
    def __init__(self, train_config: dict, env_config: dict):
        self.train_config = train_config
        self.env_config = env_config
        self.device = torch.device(env_config["device"])
        self.data_dir = PROJECT_ROOT / "USAD_custom" / "data" / "alfa_cmu16"
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
        T, K = values.shape
        if T < window_size:
            return np.empty((0, window_size, K), dtype=np.float32)
        indexer = np.arange(window_size)[None, :] + np.arange(T - window_size + 1)[:, None]
        return values[indexer].astype(np.float32)

    def _build_datasets(self):
        train_csv = self.data_dir / "train.csv"
        test_csv = self.data_dir / "test.csv"
        seq_summary_path = self.data_dir / "sequence_summary.csv"

        train_df = pd.read_csv(train_csv)
        test_df = pd.read_csv(test_csv)
        seq_summary = pd.read_csv(seq_summary_path)

        win_size = self.train_config["window_size"]
        self.win_size = win_size

        # Episode-aware sliding window for Train flights
        normal_seqs = seq_summary[seq_summary["split"] == "train"]["sequence"].tolist()
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

        # Episode-wise Temporal Block Split (90% train, 10% val)
        val_ratio = self.train_config["val_ratio"]
        train_wins, val_wins = [], []
        for ep_win in train_episodes:
            n_ep = len(ep_win)
            val_n = max(1, int(n_ep * val_ratio))
            train_n = n_ep - val_n
            train_wins.append(ep_win[:train_n])
            val_wins.append(ep_win[train_n:])

        self.train_windows = np.vstack(train_wins)
        self.val_windows = np.vstack(val_wins)

        # Build Test Windows
        self.test_flight_records = []
        test_wins_list, test_labels_list = [], []
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

        self.test_windows = np.vstack(test_wins_list)
        self.test_labels = np.concatenate(test_labels_list)

        batch = self.train_config["batch"]
        self.train_loader = DataLoader(FlightWindowDataset(self.train_windows), batch_size=batch, shuffle=True)
        self.val_loader = DataLoader(FlightWindowDataset(self.val_windows), batch_size=batch, shuffle=False)
        self.test_loader = DataLoader(FlightWindowDataset(self.test_windows), batch_size=batch, shuffle=False)

        print(
            f"[INFO] Episode-Aware Windows (W={win_size}, K={self.num_features}): "
            f"Train={len(self.train_windows):,}, Val={len(self.val_windows):,}, "
            f"Test={len(self.test_windows):,} ({len(self.test_flight_records)} flights)"
        )

    def _init_model(self):
        # TranAD with n_feats = 16, nhead = 16
        self.model = TranAD(self.num_features).to(self.device)
        print(f"[INFO] Initialized TranAD Transformer: n_feats={self.num_features}, n_window={self.win_size}")

    def get_save_paths(self):
        if self.datestr is None:
            self.datestr = datetime.now().strftime("%m_%d-%H_%M_%S")
        model_dir = CUSTOM_DIR / "pretrained" / "alfa_cmu16"
        res_dir = CUSTOM_DIR / "results"
        model_dir.mkdir(parents=True, exist_ok=True)
        res_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / f"best_tranad_{self.datestr}.pt"
        res_path = res_dir / f"alfa_tranad_{self.datestr}.json"
        return str(model_path), str(res_path)

    def train(self, model_save_path: str):
        epochs = self.train_config["epochs"]
        lr = self.train_config["lr"]
        patience = self.train_config["patience"]

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        mse_loss = nn.MSELoss(reduction="none")

        min_val_loss = float("inf")
        stop_count = 0
        history = {"train_loss": [], "val_loss": []}

        print(f"\n[INFO] Starting TranAD Adversarial Transformer Training ({epochs} epochs)...")
        print("-" * 75)
        print(f"{'Epoch':^9} | {'Train L1+L2':^14} | {'Val Loss':^14} | {'Min Val Loss':^14} | {'Status':^14}")
        print("-" * 75)

        for ep in range(1, epochs + 1):
            self.model.train()
            train_losses = []
            n_iter = ep

            for src, tgt in self.train_loader:
                # src: [Batch, W, K] -> TranAD expects [W, Batch, K]
                src = src.permute(1, 0, 2).to(self.device)
                # tgt: [Batch, 1, K] -> [1, Batch, K]
                tgt = tgt.permute(1, 0, 2).to(self.device)

                optimizer.zero_grad()
                x1, x2 = self.model(src, tgt)

                l1 = (1.0 / n_iter) * mse_loss(x1, tgt) + (1.0 - 1.0 / n_iter) * mse_loss(x2, tgt)
                loss = torch.mean(l1)

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            avg_train_loss = np.mean(train_losses)

            # Validation
            self.model.eval()
            val_losses = []
            with torch.no_grad():
                for src, tgt in self.val_loader:
                    src = src.permute(1, 0, 2).to(self.device)
                    tgt = tgt.permute(1, 0, 2).to(self.device)
                    x1, x2 = self.model(src, tgt)
                    v_loss = torch.mean(mse_loss(x2, tgt))
                    val_losses.append(v_loss.item())

            avg_val_loss = np.mean(val_losses)
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

        mse_none = nn.MSELoss(reduction="none")
        recon_list, actual_list, score_list = [], [], []

        with torch.no_grad():
            for src, tgt in self.test_loader:
                src = src.permute(1, 0, 2).to(self.device)
                tgt = tgt.permute(1, 0, 2).to(self.device)
                x1, x2 = self.model(src, tgt)
                # Phase 2 reconstruction x2 is conditioned on anomaly errors
                err_matrix = mse_none(x2, tgt)[0]  # [Batch, K]
                # Point score = mean or max across channels
                score = torch.mean(err_matrix, dim=1)

                recon_list.append(x2[0].cpu().numpy())
                actual_list.append(tgt[0].cpu().numpy())
                score_list.append(score.cpu().numpy())

        test_scores = np.concatenate(score_list)
        recon_newest = np.vstack(recon_list)
        actual_newest = np.vstack(actual_list)
        labels = self.test_labels.astype(int)

        # Threshold Search
        th_candidates = np.percentile(test_scores, np.linspace(50, 99.5, 300))
        best_f1, best_th, best_p, best_r = -1.0, 0.0, 0.0, 0.0

        for th in th_candidates:
            pred = (test_scores > th).astype(int)
            p = precision_score(labels, pred, zero_division=0)
            r = recall_score(labels, pred, zero_division=0)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            if f1 > best_f1:
                best_f1, best_th, best_p, best_r = f1, th, p, r

        auroc = roc_auc_score(labels, test_scores)

        print("\n=========================** ALFA TranAD Result (16 Raw Channels) **==========================")
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
                    latency_sec = float(alarm_after_fault[0] * 0.1)
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
        print("===========================================================================================\n")

        # Save summary JSON
        summary = {
            "model": "TranAD",
            "channels": self.feature_map,
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
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print(f"[INFO] TranAD results saved to: {result_save_path}")
        return test_scores, actual_newest, recon_newest, best_th

    def run(self):
        model_save_path, result_save_path = self.get_save_paths()
        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            self.train(model_save_path)

        self.evaluate(model_save_path, result_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TranAD on ALFA CMU-16 Raw Channels")
    parser.add_argument("-window_size", type=int, default=10, help="window size (default 10 in TranAD paper)")
    parser.add_argument("-batch", type=int, default=128, help="batch size")
    parser.add_argument("-epochs", type=int, default=40, help="max training epochs")
    parser.add_argument("-lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("-patience", type=int, default=10, help="early stopping patience")
    parser.add_argument("-val_ratio", type=float, default=0.1, help="validation ratio")
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
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "patience": args.patience,
        "val_ratio": args.val_ratio,
    }

    env_config = {
        "device": args.device,
        "load_model_path": args.load_model_path,
    }

    runner = AlfaTranADRunner(train_config, env_config)
    runner.run()
