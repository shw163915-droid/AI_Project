# -*- coding: utf-8 -*-
"""
2단계: 원본 GDN 베이스라인(C:/AI Project/GDN)을 수정 없이 import하여
ALFA 고정익 비행 데이터셋(GDN_custom/data/alfa)을 학습 및 평가하는 실행 스크립트
"""

import os
import sys
import json
import random
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

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

# 3. 원본 GDN 모듈 import (HuggingFace 'datasets' 패키지와의 이름 충돌 방지를 위해 명시적 로드)
import importlib.util

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
from train import train
from test import test
from evaluate import (
    get_full_err_scores,
    get_best_performance_data,
    get_val_performance_data,
)


class AlfaGDNRunner:
    def __init__(self, train_config: dict, env_config: dict):
        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        # GDN의 get_feature_map('./data/{dataset}/list.txt')가 GDN_custom/data/alfa를 바라보도록 작업 디렉터리 설정
        os.chdir(CUSTOM_DIR)

        dataset = self.env_config["dataset"]
        train_orig = pd.read_csv(f"./data/{dataset}/train.csv", sep=",", index_col=0)
        test_orig = pd.read_csv(f"./data/{dataset}/test.csv", sep=",", index_col=0)

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

        train_dataset_indata = construct_data(train_df, feature_map, labels=0)
        test_dataset_indata = construct_data(test_df, feature_map, labels=test_df.attack.tolist())

        cfg = {
            "slide_win": train_config["slide_win"],
            "slide_stride": train_config["slide_stride"],
        }

        train_dataset = TimeDataset(train_dataset_indata, fc_edge_index, mode="train", config=cfg)
        test_dataset = TimeDataset(test_dataset_indata, fc_edge_index, mode="test", config=cfg)

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

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat(
            [indices[:val_start_index], indices[val_start_index + val_use_len :]]
        )
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index : val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        train_dataloader = DataLoader(train_subset, batch_size=batch, shuffle=True)
        val_dataloader = DataLoader(val_subset, batch_size=batch, shuffle=False)

        return train_dataloader, val_dataloader

    def get_save_path(self):
        dir_path = self.env_config["save_path"]
        if self.datestr is None:
            now = datetime.now()
            # Windows 파일 시스템 호환 포맷 ('|' 및 ':' 제거)
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

    def save_learned_graph(self, result_json_path: str):
        """학습된 센서 간 Top-K 방향성 그래프(Learned Graph) 및 평가 지표 저장"""
        if self.model.learned_graph is None:
            return None

        topk_mat = self.model.learned_graph.detach().cpu().numpy()
        graph_dict = {}
        for i, src_sensor in enumerate(self.feature_map):
            neighbors = [self.feature_map[idx] for idx in topk_mat[i]]
            graph_dict[src_sensor] = neighbors

        return graph_dict

    def get_score(self, test_result, val_result, result_save_path: str, model_save_path: str):
        np_test_result = np.array(test_result)
        test_labels = np_test_result[2, :, 0].tolist()

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print("\n=========================** ALFA GDN Result **============================")
        info = top1_best_info if self.env_config["report"] == "best" else top1_val_info

        print(f"Report Mode : {self.env_config['report']}")
        print(f"F1 score    : {info[0]:.4f}")
        print(f"Precision   : {info[1]:.4f}")
        print(f"Recall      : {info[2]:.4f}")
        print(f"AUROC       : {info[3]:.4f}")
        print(f"Threshold   : {info[4]:.4f}")
        print(f"Saved Model : {model_save_path}")
        print("==========================================================================\n")

        learned_graph = self.save_learned_graph(result_save_path)
        summary = {
            "dataset": self.env_config["dataset"],
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
            "learned_graph_topk": learned_graph,
        }
        with open(result_save_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 시점별 이상 점수 및 가장 편차가 큰 원인 센서(Root Cause) 저장
        max_sensor_idx = np.argmax(test_scores, axis=0)
        max_sensor_scores = np.max(test_scores, axis=0)
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

    def run(self):
        model_save_path, result_save_path = self.get_save_path()

        if len(self.env_config["load_model_path"]) > 0:
            model_save_path = self.env_config["load_model_path"]
        else:
            print(
                f"[INFO] Starting GDN training on '{self.env_config['dataset']}' "
                f"(device={self.device}, nodes={len(self.feature_map)}, "
                f"slide_win={self.train_config['slide_win']}, topk={self.train_config['topk']})..."
            )
            self.train_log = train(
                self.model,
                model_save_path,
                config=self.train_config,
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                feature_map=self.feature_map,
                test_dataloader=self.test_dataloader,
                test_dataset=self.test_dataset,
                train_dataset=self.train_dataset,
                dataset_name=self.env_config["dataset"],
            )

        self.model.load_state_dict(torch.load(model_save_path, map_location=self.device))
        best_model = self.model.to(self.device)

        _, self.test_result = test(best_model, self.test_dataloader)
        _, self.val_result = test(best_model, self.val_dataloader)

        self.get_score(self.test_result, self.val_result, result_save_path, model_save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run GDN on ALFA dataset without modifying baseline")
    parser.add_argument("-batch", help="batch size", type=int, default=128)
    parser.add_argument("-epoch", help="train epoch", type=int, default=30)
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
    }

    runner = AlfaGDNRunner(train_config, env_config)
    runner.run()
