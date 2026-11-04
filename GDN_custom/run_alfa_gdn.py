# -*- coding: utf-8 -*-
"""
2단계 (1번 개선 + 현업 스타일 실시간 진행률/시각화 반영):
원본 GDN 베이스라인(C:/AI Project/GDN)을 수정 없이 import하여
비행 시퀀스별 독립 슬라이딩 윈도우(ConcatDataset)로 ALFA 데이터셋을 학습 및 평가하는 스크립트
"""

import os
import sys
import time
import json
import random
import warnings
import argparse
import importlib.util
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm

# 1. 원본 GDN 레포지토리 경로를 sys.path에 추가 (원본 파일 수정 없음)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
GDN_REPO_PATH = PROJECT_ROOT / "GDN"
CUSTOM_DIR = Path(__file__).resolve().parent

if str(GDN_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(GDN_REPO_PATH))

# 2. PyTorch Geometric 2.x 호환성 런타임 어댑터 (원본 GDN/models/graph_layer.py 무수정)
from torch_geometric.utils import softmax as pyg_softmax
import models.graph_layer as gl

gl.softmax = lambda src, index, num_nodes=None: pyg_softmax(src, index, num_nodes=num_nodes)
_orig_graph_layer_init = gl.GraphLayer.__init__


def _compat_graph_layer_init(self, *args, **kwargs):
    kwargs.setdefault("node_dim", 0)
    _orig_graph_layer_init(self, *args, **kwargs)


gl.GraphLayer.__init__ = _compat_graph_layer_init

# 3. 원본 GDN 모듈 import
from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc

_td_spec = importlib.util.spec_from_file_location(
    "gdn_time_dataset", GDN_REPO_PATH / "datasets" / "TimeDataset.py"
)
_td_module = importlib.util.module_from_spec(_td_spec)
_td_spec.loader.exec_module(_td_module)
TimeDataset = _td_module.TimeDataset

from models.GDN import GDN
from train import loss_func
from test import test
from evaluate import (
    get_full_err_scores,
    get_best_performance_data,
    get_val_performance_data,
)


def train_with_progress(
    model,
    save_path: str,
    config: dict,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader = None,
):
    """
    원본 GDN/train.py의 학습 알고리즘(Adam lr=0.001, MSE loss, early_stop_win=15)을 100% 동일하게 수행하되,
    현업 표준 스타일(tqdm 실시간 진행률 바, Train/Val Loss 비교, Best 모델 갱신 표시, Early Stopping 카운터)을 출력
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=config["decay"])
    device = get_device()
    epochs = config["epoch"]
    early_stop_win = 15
    stop_improve_count = 0
    min_val_loss = float("inf")

    history = {"train_loss": [], "val_loss": []}

    print("\n" + "=" * 88)
    print(
        f"{'Epoch':^9} | {'Train Loss':^14} | {'Val Loss':^14} | {'Best Val':^14} | {'Time':^8} | {'Status':^15}"
    )
    print("-" * 88)

    for i_epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        acu_loss = 0.0

        pbar = tqdm(
            train_dataloader,
            desc=f"Epoch {i_epoch:02d}/{epochs:02d}",
            leave=False,
            ncols=95,
            bar_format="{l_bar}{bar:25}{r_bar}",
        )

        for x, labels, _, edge_index in pbar:
            x, labels, edge_index = [item.float().to(device) for item in [x, labels, edge_index]]

            optimizer.zero_grad()
            out = model(x, edge_index).float().to(device)
            loss = loss_func(out, labels)

            loss.backward()
            optimizer.step()

            acu_loss += loss.item()
            pbar.set_postfix({"batch_mse": f"{loss.item():.5f}"})

        avg_train_loss = acu_loss / max(len(train_dataloader), 1)
        history["train_loss"].append(avg_train_loss)

        if val_dataloader is not None:
            val_loss, _ = test(model, val_dataloader)
            history["val_loss"].append(val_loss)

            if val_loss < min_val_loss:
                torch.save(model.state_dict(), save_path)
                min_val_loss = val_loss
                stop_improve_count = 0
                status_str = "★ Best Saved"
            else:
                stop_improve_count += 1
                status_str = f"Wait ({stop_improve_count}/{early_stop_win})"
        else:
            val_loss = avg_train_loss
            history["val_loss"].append(val_loss)
            if avg_train_loss < min_val_loss:
                torch.save(model.state_dict(), save_path)
                min_val_loss = avg_train_loss
                status_str = "★ Best Saved"
            else:
                status_str = "-"

        elapsed = time.time() - t0
        print(
            f"{f'{i_epoch:02d}/{epochs:02d}':^9} | "
            f"{avg_train_loss:^14.6f} | "
            f"{val_loss:^14.6f} | "
            f"{min_val_loss:^14.6f} | "
            f"{elapsed:^7.2f}s | "
            f"{status_str:<15}",
            flush=True,
        )

        if val_dataloader is not None and stop_improve_count >= early_stop_win:
            print("-" * 88)
            print(f"[Early Stopping] Validation loss did not improve for {early_stop_win} epochs.")
            break

    print("=" * 88 + "\n")
    return history


class AlfaGDNRunner:
    def __init__(self, train_config: dict, env_config: dict):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        os.chdir(CUSTOM_DIR)

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)
        seq_summary_path = Path(f"./data/{dataset}/sequence_summary.csv")

        train_df, test_df = train_orig.copy(), test_orig.copy()
        if "attack" in train_df.columns:
            train_df = train_df.drop(columns=["attack"])

        feature_map = get_feature_map(dataset)
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config["device"])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc, list(train_df.columns), feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)

        self.feature_map = feature_map

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        # [1번 개선] 비행 시퀀스별 독립 윈도잉 + ConcatDataset 구성
        self.test_flight_records = []
        if seq_summary_path.exists() and self.env_config.get("episode_window", True):
            seq_summary = pd.read_csv(seq_summary_path)
            train_dataset = self._build_episode_concat_dataset(
                train_df, seq_summary, "train", feature_map, fc_edge_index, cfg
            )
            test_dataset = self._build_episode_concat_dataset(
                test_df, seq_summary, "test", feature_map, fc_edge_index, cfg
            )
            print(
                f"[INFO] Episode-aware ConcatDataset enabled: "
                f"Train windows={len(train_dataset):,}, Test windows={len(test_dataset):,}"
            )
        else:
            train_indata = construct_data(train_df, feature_map, labels=0)
            test_indata = construct_data(test_df, feature_map, labels=test_df.attack.tolist())
            train_dataset = TimeDataset(train_indata, fc_edge_index, mode="train", config=cfg)
            test_dataset = TimeDataset(test_indata, fc_edge_index, mode="test", config=cfg)

        train_dataloader, val_dataloader = self.get_loaders(
            train_dataset,
            train_config["seed"],
            train_config["batch"],
            val_ratio=train_config["val_ratio"],
        )

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=train_config["batch"], shuffle=False, num_workers=0
        )

        edge_index_sets = [fc_edge_index]

        self.model = GDN(
            edge_index_sets,
            len(feature_map),
            dim=train_config["dim"],
            input_dim=train_config["slide_win"],
            out_layer_num=train_config["out_layer_num"],
            out_layer_inter_dim=train_config["out_layer_inter_dim"],
            topk=train_config["topk"],
        ).to(self.device)

    def _build_episode_concat_dataset(
        self,
        full_df: pd.DataFrame,
        seq_summary: pd.DataFrame,
        split_name: str,
        feature_map: list,
        fc_edge_index: torch.Tensor,
        cfg: dict,
    ) -> ConcatDataset:
        sub_summary = seq_summary[seq_summary["split"] == split_name]
        episode_datasets = []
        cursor = 0
        win_cursor = 0

        for _, row in sub_summary.iterrows():
            seq_name = row["sequence"]
            n_samples = int(row["num_samples"])
            ep_df = full_df.iloc[cursor : cursor + n_samples].reset_index(drop=True)
            cursor += n_samples

            if len(ep_df) <= cfg["slide_win"]:
                continue

            if split_name == "train":
                indata = construct_data(ep_df, feature_map, labels=0)
            else:
                indata = construct_data(
                    ep_df.drop(columns=["attack"]),
                    feature_map,
                    labels=ep_df["attack"].tolist(),
                )

            ep_ds = TimeDataset(indata, fc_edge_index, mode=split_name, config=cfg)
            episode_datasets.append(ep_ds)

            if split_name == "test":
                n_win = len(ep_ds)
                self.test_flight_records.append(
                    {
                        "sequence": seq_name,
                        "win_start": win_cursor,
                        "win_end": win_cursor + n_win,
                        "num_windows": n_win,
                    }
                )
                win_cursor += n_win

        return ConcatDataset(episode_datasets)

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        """
        다중 비행 에피소드(ConcatDataset)에서 시계열 연속성(Data Leakage 방지)을 지키면서
        특정 1개 비행에만 검증 셋이 편중되지 않도록, 각 정상 비행의 앞쪽 (1 - val_ratio) 구간을 Train으로,
        뒤쪽 val_ratio 구간을 Val로 균등 분할(Episode-wise Temporal Block Split)합니다.
        """
        if isinstance(train_dataset, ConcatDataset) and len(train_dataset.datasets) > 1:
            train_idx_list = []
            val_idx_list = []
            offset = 0
            for ds in train_dataset.datasets:
                n_len = len(ds)
                val_len = max(1, int(n_len * val_ratio))
                train_len = n_len - val_len
                ep_indices = torch.arange(offset, offset + n_len)
                train_idx_list.append(ep_indices[:train_len])
                val_idx_list.append(ep_indices[train_len:])
                offset += n_len

            train_sub_indices = torch.cat(train_idx_list)
            val_sub_indices = torch.cat(val_idx_list)
        else:
            dataset_len = int(len(train_dataset))
            train_use_len = int(dataset_len * (1 - val_ratio))
            val_use_len = int(dataset_len * val_ratio)
            val_start_index = random.randrange(train_use_len)
            indices = torch.arange(dataset_len)

            train_sub_indices = torch.cat(
                [indices[:val_start_index], indices[val_start_index + val_use_len :]]
            )
            val_sub_indices = indices[val_start_index : val_start_index + val_use_len]

        train_subset = Subset(train_dataset, train_sub_indices)
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def get_save_path(self):
        dir_path = self.env_config["save_path"]
        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime("%m_%d-%H_%M_%S")
        datestr = self.datestr

        paths = [
            f"./pretrained/{dir_path}/best_{datestr}.pt",
            f"./results/{dir_path}/{datestr}.json",
        ]

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths

    def save_learned_graph(self):
        if self.model.learned_graph is None:
            return None

        topk_mat = self.model.learned_graph.detach().cpu().numpy()
        graph_dict = {}
        for i, src_sensor in enumerate(self.feature_map):
            neighbors = [self.feature_map[idx] for idx in topk_mat[i]]
            graph_dict[src_sensor] = neighbors

        return graph_dict

    def save_training_plots(
        self,
        history: dict,
        np_test_result: np.ndarray,
        test_scores: np.ndarray,
        max_sensor_scores: np.ndarray,
        test_labels: list,
        threshold: float,
        result_save_path: str,
    ):
        """
        1) *_dashboard.png: 에포크별 Loss 수렴 곡선 + 대표 단일 비행의 [실제 센서값 vs GDN 예측값] 및 [실시간 이상 점수]
        """
        # 대표 비행 선택 (엘리베이터/에일러론/엔진 결함 중 고장 전후 비교가 뚜렷한 비행 선택)
        target_rec = None
        for rec in self.test_flight_records:
            if "elevator_failure" in rec["sequence"]:
                target_rec = rec
                break
        if target_rec is None and len(self.test_flight_records) > 0:
            target_rec = self.test_flight_records[0]

        fig, axes = plt.subplots(3, 1, figsize=(13, 10))

        # [그래프 1] Train & Val Loss Curve
        if history and len(history.get("train_loss", [])) > 0:
            ep_range = range(1, len(history["train_loss"]) + 1)
            axes[0].plot(ep_range, history["train_loss"], marker="o", label="Train MSE Loss")
            if len(history.get("val_loss", [])) > 0:
                axes[0].plot(ep_range, history["val_loss"], marker="s", label="Val MSE Loss")
            axes[0].set_title("1. GDN Training & Validation Loss Curve")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("MSE Loss")
            axes[0].grid(True, alpha=0.3)
            axes[0].legend(loc="upper right")

        if target_rec is not None:
            s, e = target_rec["win_start"], target_rec["win_end"]
            seq_name = target_rec["sequence"]
            seq_gt = np.array(test_labels[s:e], dtype=int)
            seq_scores = max_sensor_scores[s:e]
            time_sec = np.arange(e - s) * 0.1  # 10Hz -> 초(sec) 단위 변환

            # 고장 구간에서 가장 편차가 크게 발생한 Top-1 원인 센서 자동 찾기
            fault_mask = seq_gt == 1
            if np.any(fault_mask):
                sensor_fault_means = np.mean(test_scores[:, s:e][:, fault_mask], axis=1)
                top_sensor_idx = int(np.argmax(sensor_fault_means))
                onset_sec = time_sec[np.where(fault_mask)[0][0]]
            else:
                top_sensor_idx = 0
                onset_sec = None

            top_sensor_name = self.feature_map[top_sensor_idx]
            pred_vals = np_test_result[0, s:e, top_sensor_idx]
            actual_vals = np_test_result[1, s:e, top_sensor_idx]

            # [그래프 2] 단일 비행의 실제 센서 측정값 vs GDN 예측 정상값 비교
            axes[1].plot(
                time_sec, actual_vals, color="black", linewidth=1.3, label=f"Actual Sensor ({top_sensor_name})"
            )
            axes[1].plot(
                time_sec,
                pred_vals,
                color="tab:blue",
                linestyle="--",
                linewidth=1.3,
                label=f"GDN Predicted ({top_sensor_name})",
            )
            if onset_sec is not None:
                axes[1].axvline(onset_sec, color="tab:red", linewidth=1.5, label=f"Fault Onset ({onset_sec:.1f}s)")
                axes[1].axvspan(onset_sec, time_sec[-1], color="tab:orange", alpha=0.2, label="Fault Interval")
            axes[1].set_title(
                f"2. Single Flight Sensor Forecasting vs Actual [{seq_name}] (Root Cause: {top_sensor_name})"
            )
            axes[1].set_xlabel("Flight Time (seconds)")
            axes[1].set_ylabel("Scaled Sensor Value (0~1)")
            axes[1].grid(True, alpha=0.3)
            axes[1].legend(loc="upper left")

            # [그래프 3] 해당 비행의 실시간 이상 점수(Anomaly Score) 및 경보 임계값
            axes[2].plot(time_sec, seq_scores, color="tab:purple", linewidth=1.2, label="Real-time Anomaly Score")
            axes[2].axhline(
                threshold, color="tab:red", linestyle="--", linewidth=1.3, label=f"Alarm Threshold ({threshold:.2f})"
            )
            if onset_sec is not None:
                axes[2].axvline(onset_sec, color="tab:red", linewidth=1.5)
                axes[2].axvspan(onset_sec, time_sec[-1], color="tab:orange", alpha=0.2, label="Fault Interval")
            axes[2].set_title(f"3. Real-time Anomaly Score in Single Flight [{seq_name}]")
            axes[2].set_xlabel("Flight Time (seconds)")
            axes[2].set_ylabel("Anomaly Score")
            axes[2].grid(True, alpha=0.3)
            axes[2].legend(loc="upper left")

        plt.tight_layout()
        plot_path = result_save_path.replace(".json", "_dashboard.png")
        plt.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"[INFO] Visual dashboard plot saved to: {plot_path}")

    def evaluate_per_flight(
        self, max_sensor_scores: np.ndarray, test_labels: list, global_th: float
    ):
        if not self.test_flight_records:
            return []

        per_flight_results = []
        gt_arr = np.array(test_labels, dtype=int)

        for rec in self.test_flight_records:
            s, e = rec["win_start"], rec["win_end"]
            seq_scores = max_sensor_scores[s:e]
            seq_gt = gt_arr[s:e]

            pred_global = (seq_scores > global_th).astype(int)
            f1_global = f1_score(seq_gt, pred_global, zero_division=0)

            try:
                auroc = roc_auc_score(seq_gt, seq_scores) if len(np.unique(seq_gt)) > 1 else 0.0
            except Exception:
                auroc = 0.0

            fault_indices = np.where(seq_gt == 1)[0]
            latency_sec = None
            if len(fault_indices) > 0:
                onset_idx = fault_indices[0]
                post_fault_alarms = np.where(pred_global[onset_idx:] == 1)[0]
                if len(post_fault_alarms) > 0:
                    latency_sec = round(float(post_fault_alarms[0]) * 0.1, 2)

            per_flight_results.append(
                {
                    "sequence": rec["sequence"],
                    "num_windows": rec["num_windows"],
                    "f1_at_global_th": round(float(f1_global), 4),
                    "auroc": round(float(auroc), 4),
                    "detection_latency_sec": latency_sec,
                }
            )

        return per_flight_results

    def compute_ewma_err_scores(
        self,
        test_result: list,
        val_result: list,
        alpha: float = 0.3,
    ):
        """
        원본 GDN(GDN/evaluate.py)의 Median/IQR 정규화 공식은 100% 그대로 유지하되,
        단순 이동 평균(before_num=3)만 지수 가중 이동 평균(EWMA: S_t = alpha * e_t + (1 - alpha) * S_{t-1})으로 교체
        """
        np_test_result = np.array(test_result, dtype=np.float64)
        np_val_result = np.array(val_result, dtype=np.float64)
        feature_num = np_test_result.shape[-1]
        epsilon = 1e-2

        all_scores = []
        all_normals = []

        for i in range(feature_num):
            test_pred, test_gt = np_test_result[0, :, i], np_test_result[1, :, i]
            val_pred, val_gt = np_val_result[0, :, i], np_val_result[1, :, i]

            # 원본 GDN evaluate.py의 get_err_scores와 동일한 Median & IQR 계산
            test_delta = np.abs(test_pred - test_gt)
            n_err_mid = np.median(test_delta)
            n_err_iqr = np.percentile(test_delta, 75) - np.percentile(test_delta, 25)
            err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)

            val_delta = np.abs(val_pred - val_gt)
            v_err_mid = np.median(val_delta)
            v_err_iqr = np.percentile(val_delta, 75) - np.percentile(val_delta, 25)
            val_err_scores = (val_delta - v_err_mid) / (np.abs(v_err_iqr) + epsilon)

            # 비행 에피소드별로 EWMA 적용
            if self.test_flight_records:
                ewma_test = np.zeros_like(err_scores)
                for rec in self.test_flight_records:
                    s, e = rec["win_start"], rec["win_end"]
                    ewma_test[s:e] = (
                        pd.Series(err_scores[s:e]).ewm(alpha=alpha, adjust=False).mean().values
                    )
            else:
                ewma_test = pd.Series(err_scores).ewm(alpha=alpha, adjust=False).mean().values

            ewma_val = pd.Series(val_err_scores).ewm(alpha=alpha, adjust=False).mean().values

            all_scores.append(ewma_test)
            all_normals.append(ewma_val)

        return np.vstack(all_scores), np.vstack(all_normals)

    def get_score(self, test_result, val_result, result_save_path: str, model_save_path: str):
        np_test_result = np.array(test_result)
        test_labels = np_test_result[2, :, 0].tolist()

        ewma_alpha = self.env_config.get("ewma_alpha", 0.3)
        test_scores, normal_scores = self.compute_ewma_err_scores(
            test_result, val_result, alpha=ewma_alpha
        )

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        max_sensor_idx = np.argmax(test_scores, axis=0)
        max_sensor_scores = np.max(test_scores, axis=0)
        per_flight_stats = self.evaluate_per_flight(
            max_sensor_scores, test_labels, float(top1_best_info[4])
        )

        print("=========================** ALFA GDN Result **============================")
        info = top1_best_info if self.env_config["report"] == "best" else top1_val_info

        print(f"Report Mode        : {self.env_config['report']}")
        print(f"EWMA Alpha         : {ewma_alpha}")
        print(f"Global F1 score    : {info[0]:.4f}")
        print(f"Global Precision   : {info[1]:.4f}")
        print(f"Global Recall      : {info[2]:.4f}")
        print(f"Global AUROC       : {info[3]:.4f}")
        print(f"Global Threshold   : {info[4]:.4f}")

        if per_flight_stats:
            mean_auroc = np.mean([x["auroc"] for x in per_flight_stats])
            detected_latencies = [
                x["detection_latency_sec"]
                for x in per_flight_stats
                if x["detection_latency_sec"] is not None
            ]
            mean_latency = np.mean(detected_latencies) if detected_latencies else -1.0
            print(f"Per-Flight AUROC   : {mean_auroc:.4f} (mean across {len(per_flight_stats)} flights)")
            print(
                f"Detected Flights   : {len(detected_latencies)} / {len(per_flight_stats)} "
                f"(Mean Latency: {mean_latency:.2f}s)"
            )

        print(f"Saved Model        : {model_save_path}")
        print("==========================================================================\n")

        learned_graph = self.save_learned_graph()
        summary = {
            "dataset": self.env_config["dataset"],
            "episode_window_enabled": self.env_config.get("episode_window", True),
            "ewma_alpha": ewma_alpha,
            "model_path": model_save_path,
            "train_config": self.train_config,
            "best_metrics": {
                "f1": float(top1_best_info[0]),
                "precision": float(top1_best_info[1]),
                "recall": float(top1_best_info[2]),
                "auroc": float(top1_best_info[3]),
                "threshold": float(top1_best_info[4]),
            },
            "val_metrics": {
                "f1": float(top1_val_info[0]),
                "precision": float(top1_val_info[1]),
                "recall": float(top1_val_info[2]),
                "auroc": float(top1_val_info[3]),
                "threshold": float(top1_val_info[4]),
            },
            "per_flight_summary": per_flight_stats,
            "learned_graph_topk": learned_graph,
        }
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        root_cause_sensors = [self.feature_map[idx] for idx in max_sensor_idx]
        score_df = pd.DataFrame(
            {
                "anomaly_score": max_sensor_scores,
                "ground_truth": [int(x) for x in test_labels],
                "predicted_anomaly_best": (max_sensor_scores > top1_best_info[4]).astype(int),
                "root_cause_sensor": root_cause_sensors,
            }
        )
        score_csv_path = result_save_path.replace(".json", "_scores.csv")
        score_df.to_csv(score_csv_path, index=True)
        print(f"[INFO] Detailed timestep scores & root-cause sensors saved to: {score_csv_path}")

        self.save_training_plots(
            getattr(self, "train_history", {}),
            np_test_result,
            test_scores,
            max_sensor_scores,
            test_labels,
            float(top1_best_info[4]),
            result_save_path,
        )

    def run(self):
        model_save_path, result_save_path = self.get_save_path()
        self.train_history = {}

        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            print(
                f"[INFO] Starting GDN training on '{self.env_config['dataset']}' "
                f"(device={self.device}, nodes={len(self.feature_map)}, "
                f"slide_win={self.train_config['slide_win']}, topk={self.train_config['topk']})..."
            )
            self.train_history = train_with_progress(
                self.model,
                model_save_path,
                config=self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
            )

        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result, result_save_path, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GDN on ALFA dataset without modifying baseline")
    parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=500)
    parser.add_argument("-slide_win", help="slide_win (default 15 = 1.5s at 10Hz)", type=int, default=15)
    parser.add_argument("-dim", help="embedding dimension", type=int, default=64)
    parser.add_argument("-slide_stride", help="slide_stride", type=int, default=1)
    parser.add_argument("-save_path_pattern", help="save path pattern", type=str, default="alfa")
    parser.add_argument("-dataset", help="dataset folder name", type=str, default="alfa")
    parser.add_argument("-device", help="cuda / cpu", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("-random_seed", help="random seed", type=int, default=42)
    parser.add_argument("-comment", help="experiment comment", type=str, default="alfa_gdn")
    parser.add_argument("-out_layer_num", help="outlayer num", type=int, default=1)
    parser.add_argument("-out_layer_inter_dim", help="out_layer_inter_dim", type=int, default=128)
    parser.add_argument("-decay", help="weight decay", type=float, default=0)
    parser.add_argument("-val_ratio", help="val ratio", type=float, default=0.1)
    parser.add_argument("-topk", help="topk graph neighbors (default 8 out of 24 sensors)", type=int, default=8)
    parser.add_argument("-report", help="best / val", type=str, default="best")
    parser.add_argument("-load_model_path", help="trained model path", type=str, default="")
    parser.add_argument("-episode_window", help="1: episode-aware ConcatDataset, 0: flat", type=int, default=1)
    parser.add_argument("-ewma_alpha", help="EWMA smoothing factor alpha (0.3 default)", type=float, default=0.3)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(args.random_seed)

    train_config = {
        "batch": args.batch,
        "epoch": args.epoch,
        "slide_win": args.slide_win,
        "dim": args.dim,
        "slide_stride": args.slide_stride,
        "comment": args.comment,
        "seed": args.random_seed,
        "out_layer_num": args.out_layer_num,
        "out_layer_inter_dim": args.out_layer_inter_dim,
        "decay": args.decay,
        "val_ratio": args.val_ratio,
        "topk": args.topk,
    }

    env_config = {
        "save_path": args.save_path_pattern,
        "dataset": args.dataset,
        "report": args.report,
        "device": args.device,
        "load_model_path": args.load_model_path,
        "episode_window": bool(args.episode_window),
        "ewma_alpha": args.ewma_alpha,
    }

    runner = AlfaGDNRunner(train_config, env_config)
    runner.run()
