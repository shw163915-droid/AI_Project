import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1dBlock(nn.Module):
    """
    정통 TCN 논문 (Bai et al., 2018 - locuslab/TCN) 기반 인과적 확장 합성곱 블록
    - Dilation(확장 계수) = 1, 2, 4, 8... 로 수용 영역(Receptive Field)을 지수적으로 넓힙니다.
    - Chomp 연산으로 미래 시점(t+1 이후)의 패딩을 잘라내어 실시간 인과성(Causality)을 보장합니다.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation

        self.conv1 = weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        )
        self.act1 = nn.GELU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        )
        self.act2 = nn.GELU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )
        self.out_act = nn.GELU()

    def _chomp(self, x):
        return x[:, :, :-self.padding].contiguous() if self.padding > 0 else x

    def forward(self, x):
        out = self.drop1(self.act1(self._chomp(self.conv1(x))))
        out = self.drop2(self.act2(self._chomp(self.conv2(out))))
        res = x if self.downsample is None else self.downsample(x)
        return self.out_act(out + res)


class Model(nn.Module):
    """
    Dual-Head TCN 모델 (TSLib 호환)
    - Shared Backbone: Multi-layer Dilated Causal TCN
    - Head 1 (Classification Head): 알려진 상태 [Normal(0), WCA_1(1), WCA_2(2)] 분류
    - Head 2 (Forecasting Head): 8개 센서의 다음 시점 값 예측 -> |X_next - X_hat| 오차로 Unknown 탐지 및 센서별 기여도(%) 산출
    """
    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.num_sensors = configs.enc_in
        self.hidden_dim = configs.d_model
        self.num_levels = configs.e_layers
        self.num_classes = getattr(configs, 'num_class', 3)
        kernel_size = getattr(configs, 'd_conv', 3)
        dropout = getattr(configs, 'dropout', 0.2)

        blocks = []
        for i in range(self.num_levels):
            dilation = 2 ** i
            in_ch = self.num_sensors if i == 0 else self.hidden_dim
            blocks.append(
                CausalConv1dBlock(
                    in_channels=in_ch,
                    out_channels=self.hidden_dim,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
        self.tcn_backbone = nn.Sequential(*blocks)

        # Head 1: 현재 시점 상태 분류기 (Normal, WCA_1, WCA_2)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

        # Head 2: 다음 시점 센서 값 예측기 (기여도 및 Unknown 오차 산출)
        self.forecast_head = nn.Conv1d(
            in_channels=self.hidden_dim,
            out_channels=self.num_sensors,
            kernel_size=1,
        )

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # 입력 x_enc: [Batch, Time(T), Sensors(C)] -> Conv1d를 위해 [Batch, C, T]로 변환
        x = x_enc.transpose(1, 2)
        feat_seq = self.tcn_backbone(x)  # [Batch, hidden_dim, T]

        # Head 1: 현재 시점(마지막 타임스탬프 -1) 상태 분류 로짓
        last_step_feat = feat_seq[:, :, -1]
        logits = self.classifier_head(last_step_feat)  # [Batch, num_classes]

        # Head 2: 다음 시점 센서 값 예측 [Batch, T, C]
        x_hat = self.forecast_head(feat_seq).transpose(1, 2)

        if self.task_name == 'skab_dual':
            return logits, x_hat, last_step_feat
        elif self.task_name == 'classification':
            return logits
        else:
            return x_hat

