import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from data_provider.data_factory import data_provider
from data_provider.skab_loader import SKAB_SENSOR_COLS
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping


class Exp_SKAB_Dual(Exp_Basic):
    """
    SKAB 다중 결함 데이터셋 기반 Dual-Head TCN 학습 및 평가 클래스
    - 알려진 상태 분류: 0=Normal, 1=WCA_1(Valve1 입구밸브 결함), 2=WCA_2(Valve2 출구밸브 결함)
    - 미지의 이상 탐지: 3=Unknown(Other 폴더의 유체 누출, 로터 불균형, 공동 현상 등 학습 때 본 적 없는 결함)
    - 센서별 기여도 산출: 8개 물리 센서의 예측 오차(|X_next - X_hat|) 백분율 정규화
    """
    def __init__(self, args):
        super().__init__(args)
        self.anomaly_threshold = None

    def _build_model(self):
        model = self.model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        return data_provider(self.args, flag)

    def _select_optimizer(self):
        return optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-4)

    def _compute_dual_loss(self, logits, x_hat, x_next, y_label):
        # 1. Head 1: 알려진 상태(Normal=0, WCA_1=1, WCA_2=2) 분류 Cross-Entropy Loss (클래스 불균형 가중치 보정)
        known_mask = y_label >= 0
        if known_mask.sum() > 0:
            loss_cls = F.cross_entropy(logits[known_mask], y_label[known_mask])
        else:
            loss_cls = torch.tensor(0.0, device=self.device)

        # 2. Head 2: 오직 Normal(y_label == 0) 데이터에 대해서만 정상 물리 법칙(다음 시점 센서값 예측) 학습
        normal_mask = y_label == 0
        if normal_mask.sum() > 0:
            loss_forecast = F.mse_loss(x_hat[normal_mask], x_next[normal_mask])
        else:
            loss_forecast = torch.tensor(0.0, device=self.device)

        total_loss = loss_cls + loss_forecast
        return total_loss, loss_cls.item(), loss_forecast.item()

    @torch.no_grad()
    def vali(self, vali_data, vali_loader):
        self.model.eval()
        total_loss = []
        normal_errors = []

        for x_window, x_next, y_label in vali_loader:
            x_window = x_window.float().to(self.device)
            x_next = x_next.float().to(self.device)
            y_label = y_label.long().to(self.device)

            logits, x_hat, _ = self.model(x_window)
            loss, _, _ = self._compute_dual_loss(logits, x_hat, x_next, y_label)
            total_loss.append(loss.item())

            # 현업 기준 Threshold 산출: 오직 검증셋의 정상(Normal, y_label == 0) 데이터의 복원 오차 분포만 수집
            normal_mask = y_label == 0
            if normal_mask.sum() > 0:
                step_err = torch.mean((x_next[normal_mask, -1, :] - x_hat[normal_mask, -1, :]) ** 2, dim=-1)
                normal_errors.append(step_err.cpu().numpy())

        if len(normal_errors) > 0:
            all_norm_err = np.concatenate(normal_errors, axis=0)
            # 정상 데이터 오차 분포의 95% 백분위수(Percentile)를 Unknown 탐지 임계값으로 설정
            self.anomaly_threshold = float(np.percentile(all_norm_err, 95.0))

        self.model.train()
        return float(np.mean(total_loss))

    def train(self, setting):
        _, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')

        path = os.path.join(self.args.checkpoints, setting)
        os.makedirs(path, exist_ok=True)

        optimizer = self._select_optimizer()
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        for epoch in range(self.args.train_epochs):
            self.model.train()
            train_losses, cls_losses, fc_losses = [], [], []
            epoch_time = time.time()

            for x_window, x_next, y_label in train_loader:
                optimizer.zero_grad()
                x_window = x_window.float().to(self.device)
                x_next = x_next.float().to(self.device)
                y_label = y_label.long().to(self.device)

                logits, x_hat, _ = self.model(x_window)
                loss, l_cls, l_fc = self._compute_dual_loss(logits, x_hat, x_next, y_label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=4.0)
                optimizer.step()

                train_losses.append(loss.item())
                cls_losses.append(l_cls)
                fc_losses.append(l_fc)

            vali_loss = self.vali(vali_data, vali_loader)
            print(
                f"Epoch: {epoch + 1:02d}/{self.args.train_epochs:02d} | Time: {time.time() - epoch_time:.1f}s | "
                f"Train Loss: {np.mean(train_losses):.4f} (Cls: {np.mean(cls_losses):.4f}, Forecast MSE: {np.mean(fc_losses):.4f}) | "
                f"Val Loss: {vali_loss:.4f} | Normal 95% Threshold: {self.anomaly_threshold:.4f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        best_model_path = os.path.join(path, 'checkpoint.pth')
        self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))
        self.vali(vali_data, vali_loader)
        return self.model

    @torch.no_grad()
    def test(self, setting, test=0):
        _, test_loader = self._get_data(flag='test')
        if test:
            best_model_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(best_model_path, map_location=self.device))

        if self.anomaly_threshold is None:
            vali_data, vali_loader = self._get_data(flag='val')
            self.vali(vali_data, vali_loader)

        self.model.eval()
        all_true_labels = []
        all_pred_labels = []
        all_state_probs = []
        all_contributions = []
        all_anomaly_scores = []

        unknown_idx = self.args.num_class  # 3번 인덱스를 UNKNOWN 클래스로 부여

        for x_window, x_next, y_label in test_loader:
            x_window = x_window.float().to(self.device)
            x_next = x_next.float().to(self.device)

            logits, x_hat, _ = self.model(x_window)

            # 1. 현재 시점(-1)의 센서별 물리적 예측 오차 (제곱 오차 및 절대 오차)
            sensor_sq_err = (x_next[:, -1, :] - x_hat[:, -1, :]) ** 2  # [Batch, 8]
            sensor_abs_err = torch.abs(x_next[:, -1, :] - x_hat[:, -1, :])

            # 2. 8개 센서별 이상 기여도(%) 산출 (합계 = 100%)
            sensor_contrib_pct = (
                sensor_abs_err / (sensor_abs_err.sum(dim=-1, keepdim=True) + 1e-8)
            ) * 100.0

            # 3. 시점별 종합 이상 점수 (MSE Anomaly Score)
            anomaly_score = sensor_sq_err.mean(dim=-1)  # [Batch]

            # 4. 상태 분류 확률 [P(Normal), P(WCA_1), P(WCA_2)] 및 P(Unknown) 결합
            known_probs = F.softmax(logits, dim=-1)
            max_known_prob, pred_known_cls = torch.max(known_probs, dim=-1)

            # 예측 오차가 임계값(Normal 95% Threshold)을 넘을수록 Unknown 확률 상승
            unknown_ratio = torch.sigmoid((anomaly_score - self.anomaly_threshold) / (self.anomaly_threshold + 1e-6))
            # 단, 분류기가 알려진 WCA(1 또는 2)라고 높은 확신(>0.75)을 가지면서 오차가 극단적이지 않다면 기존 WCA 유지
            is_known_wca = (pred_known_cls > 0) & (max_known_prob > 0.75) & (anomaly_score < self.anomaly_threshold * 3.0)
            unknown_prob = torch.where(is_known_wca, unknown_ratio * 0.15, unknown_ratio)

            scaled_known_probs = known_probs * (1.0 - unknown_prob.unsqueeze(-1))
            full_probs = torch.cat([scaled_known_probs, unknown_prob.unsqueeze(-1)], dim=-1)  # [Batch, 4]

            final_pred = torch.argmax(full_probs, dim=-1)
            # 분류기가 Normal(0)이라고 했더라도 물리적 센서 오차가 임계값을 초과하면 Unknown(3)으로 판정!
            force_unknown = (final_pred == 0) & (anomaly_score > self.anomaly_threshold)
            final_pred = torch.where(force_unknown, torch.tensor(unknown_idx, device=self.device), final_pred)

            all_true_labels.append(y_label.numpy())
            all_pred_labels.append(final_pred.cpu().numpy())
            all_state_probs.append(full_probs.cpu().numpy())
            all_contributions.append(sensor_contrib_pct.cpu().numpy())
            all_anomaly_scores.append(anomaly_score.cpu().numpy())

        true_y = np.concatenate(all_true_labels, axis=0)
        true_y_mapped = np.where(true_y == -1, unknown_idx, true_y)
        pred_y = np.concatenate(all_pred_labels, axis=0)
        probs_mat = np.concatenate(all_state_probs, axis=0)
        contrib_mat = np.concatenate(all_contributions, axis=0)
        scores_vec = np.concatenate(all_anomaly_scores, axis=0)

        # 평가 지표 계산
        unknown_mask = true_y_mapped == unknown_idx
        unknown_recall = (pred_y[unknown_mask] == unknown_idx).mean() * 100.0 if unknown_mask.sum() > 0 else 0.0
        any_anom_true = true_y_mapped > 0
        any_anom_pred = pred_y > 0
        anom_detect_recall = (any_anom_pred[any_anom_true]).mean() * 100.0 if any_anom_true.sum() > 0 else 0.0

        class_names = {
            0: "NORMAL (정상)",
            1: "WCA_1 (Valve1 입구결함)",
            2: "WCA_2 (Valve2 출구결함)",
            unknown_idx: "UNKNOWN (Other 미지결함)",
        }

        print("\n" + "=" * 85)
        print("[SKAB 공식 데이터셋 기반 Dual-Head TCN 평가 결과]")
        print(f" - 검증셋 정상 데이터 기준 Threshold : {self.anomaly_threshold:.4f}")
        print(f" - 전체 이상(WCA + Unknown) 감지율   : {anom_detect_recall:.2f}%")
        print(f" - 미지의 결함(Other -> Unknown) 탐지율: {unknown_recall:.2f}%")
        print(f" - 출력 1 (상태 확률 행렬) 차원      : {probs_mat.shape} -> [Normal, WCA_1, WCA_2, Unknown]")
        print(f" - 출력 2 (센서 기여도 행렬) 차원    : {contrib_mat.shape} -> 8개 물리 센서별 기여도 (합계 100%)")
        print("=" * 85)

        print("\n[실제 테스트 시점별 출력 예시 (상태 분류 확률 + 8개 센서 기여도 Top 3)]")
        for target_cls in [0, 1, 2, unknown_idx]:
            # 해당 클래스에 속하면서 올바르게 예측된 대표 시점(또는 중간 시점) 추출
            matched_idx = np.where((true_y_mapped == target_cls) & (pred_y == target_cls))[0]
            sample_indices = matched_idx if len(matched_idx) > 0 else np.where(true_y_mapped == target_cls)[0]
            if len(sample_indices) > 0:
                idx = sample_indices[len(sample_indices) // 2]
                p_vec = probs_mat[idx]
                c_vec = contrib_mat[idx]
                top3 = np.argsort(c_vec)[::-1][:3]

                prob_str = " | ".join([f"{class_names[i].split()[0]}: {p_vec[i]*100:.1f}%" for i in range(len(p_vec))])
                top_sensor_str = ", ".join([f"{SKAB_SENSOR_COLS[s]} ({c_vec[s]:.1f}%)" for s in top3])

                print(f" ▶ 테스트 시점 #{idx:05d} | 실제 라벨: {class_names[target_cls]:<24} | 모델 판정: {class_names[pred_y[idx]]:<24}")
                print(f"    - 상태 확률 출력 : [{prob_str}] (물리 오차 점수: {scores_vec[idx]:.4f})")
                print(f"    - 원인 센서 Top 3: {top_sensor_str}")
                all_sensor_str = ", ".join([f"{SKAB_SENSOR_COLS[s]}:{c_vec[s]:.1f}%" for s in range(len(c_vec))])
                print(f"    - 8개 센서 전체  : [{all_sensor_str}]\n")

        folder_path = os.path.join('./results', setting)
        os.makedirs(folder_path, exist_ok=True)
        save_file = os.path.join(folder_path, 'skab_dual_outputs.npz')
        np.savez(
            save_file,
            true_labels=true_y_mapped,
            pred_labels=pred_y,
            state_probabilities=probs_mat,
            sensor_contributions_pct=contrib_mat,
            sensor_names=np.array(SKAB_SENSOR_COLS),
            anomaly_scores=scores_vec,
        )
        print(f"[Info] 전체 시점의 상태 확률 및 8개 센서 기여도 행렬이 저장되었습니다: {save_file}")

