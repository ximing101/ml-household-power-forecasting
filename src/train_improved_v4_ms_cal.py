import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# =====================
# 1. 基础配置
# =====================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEEDS = [2026, 2027, 2028, 2029, 2030]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_predictions_csv(y_true, y_pred, save_path, max_samples=None):
    """
    保存预测值和真实值，方便后续统一画图和做误差分析。

    y_true: [N, output_len]
    y_pred: [N, output_len]
    """
    rows = []

    n_samples, output_len = y_true.shape

    if max_samples is not None:
        n_samples = min(n_samples, max_samples)

    for sample_idx in range(n_samples):
        for day in range(output_len):
            true_value = y_true[sample_idx, day]
            pred_value = y_pred[sample_idx, day]
            error = pred_value - true_value

            rows.append({
                "sample_idx": sample_idx,
                "day": day + 1,
                "ground_truth": true_value,
                "prediction": pred_value,
                "error": error,
                "abs_error": abs(error),
                "squared_error": error ** 2,
            })

    df = pd.DataFrame(rows)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")

# =====================
# 2. 路径配置
# =====================

def get_project_dir():
    """
    兼容两种情况：
    1. train_improved.py 放在项目根目录
    2. train_improved.py 放在 src/ 目录
    """
    current_dir = Path(__file__).resolve().parent

    if (current_dir / "processed").exists():
        return current_dir

    if (current_dir.parent / "processed").exists():
        return current_dir.parent

    return current_dir


PROJECT_DIR = get_project_dir()
PROCESSED_DIR = PROJECT_DIR / "processed"
RESULTS_DIR = PROJECT_DIR / "results"


# =====================
# 3. Dataset
# =====================

class PowerDataset(Dataset):
    def __init__(self, X, y, future_calendar=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

        if future_calendar is None:
            self.future_calendar = None
        else:
            self.future_calendar = torch.tensor(future_calendar, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.future_calendar is None:
            return self.X[idx], self.y[idx]
        return self.X[idx], self.future_calendar[idx], self.y[idx]


# =====================
# 4. 标准化工具
# =====================

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean(axis=0, keepdims=True)
        self.std = data.std(axis=0, keepdims=True)
        self.std[self.std == 0] = 1.0

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean


def normalize_data(X_train, y_train, X_test, y_test):
    """
    X_train: [N, 90, F]
    y_train: [N, output_len]
    """

    x_scaler = StandardScaler()
    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    x_scaler.fit(X_train_2d)

    X_train_scaled = x_scaler.transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test_scaled = x_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    y_scaler = StandardScaler()
    y_train_2d = y_train.reshape(-1, 1)
    y_scaler.fit(y_train_2d)

    y_train_scaled = y_scaler.transform(
        y_train.reshape(-1, 1)
    ).reshape(y_train.shape)

    y_test_scaled = y_scaler.transform(
        y_test.reshape(-1, 1)
    ).reshape(y_test.shape)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, x_scaler, y_scaler


# =====================
# 5. 日历特征工具
# =====================

def make_calendar_features(start_dates, length):
    """
    根据每个样本的起始日期构造日历周期特征。

    输出 shape: [N, length, 6]
    特征包括：
        dayofyear_sin / dayofyear_cos
        month_sin / month_cos
        dayofweek_sin / dayofweek_cos

    注意：这些信息在预测时是已知的，不包含未来真实功率，因此不属于数据泄露。
    """
    all_features = []

    for start in start_dates:
        start = pd.to_datetime(str(start))
        dates = pd.date_range(start=start, periods=length, freq="D")

        dayofyear = dates.dayofyear.values.astype(np.float32)
        month = dates.month.values.astype(np.float32)
        dayofweek = dates.dayofweek.values.astype(np.float32)

        features = np.stack(
            [
                np.sin(2 * np.pi * dayofyear / 365.25),
                np.cos(2 * np.pi * dayofyear / 365.25),
                np.sin(2 * np.pi * month / 12.0),
                np.cos(2 * np.pi * month / 12.0),
                np.sin(2 * np.pi * dayofweek / 7.0),
                np.cos(2 * np.pi * dayofweek / 7.0),
            ],
            axis=-1,
        )

        all_features.append(features.astype(np.float32))

    return np.stack(all_features, axis=0).astype(np.float32)


def append_input_calendar_features(X, x_start_dates):
    """
    给输入窗口追加日历特征，X: [N, input_len, F] -> [N, input_len, F+6]
    """
    input_len = X.shape[1]
    input_calendar = make_calendar_features(x_start_dates, input_len)
    return np.concatenate([X, input_calendar], axis=-1).astype(np.float32)


def build_future_calendar_features(y_start_dates, output_len):
    """
    为每个未来预测步构造已知的日历特征，供输出端做季节性修正。
    """
    return make_calendar_features(y_start_dates, output_len)


# =====================
# 6. 位置编码
# =====================

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 == 1:
            pe[:, 1::2] = torch.cos(position * div_term[:-1])
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================
# 6. 1D U-Net 模块
# =====================

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class TemporalUNet1D(nn.Module):
    """
    输入:  [B, C, 90]
    输出:  [B, C, 90]
    """

    def __init__(self, hidden_dim=64, dropout=0.1):
        super().__init__()

        self.enc1 = ConvBlock1D(hidden_dim, hidden_dim, dropout)

        self.down1 = nn.Conv1d(
            hidden_dim,
            hidden_dim * 2,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.enc2 = ConvBlock1D(hidden_dim * 2, hidden_dim * 2, dropout)

        self.down2 = nn.Conv1d(
            hidden_dim * 2,
            hidden_dim * 4,
            kernel_size=3,
            stride=2,
            padding=1
        )

        self.bottleneck = ConvBlock1D(hidden_dim * 4, hidden_dim * 4, dropout)

        self.dec2 = ConvBlock1D(
            hidden_dim * 4 + hidden_dim * 2,
            hidden_dim * 2,
            dropout
        )

        self.dec1 = ConvBlock1D(
            hidden_dim * 2 + hidden_dim,
            hidden_dim,
            dropout
        )

        self.out_conv = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, C, L]

        e1 = self.enc1(x)          # [B, H, 90]

        d1 = self.down1(e1)        # [B, 2H, 45]
        e2 = self.enc2(d1)         # [B, 2H, 45]

        d2 = self.down2(e2)        # [B, 4H, 23]
        b = self.bottleneck(d2)    # [B, 4H, 23]

        b_up = F.interpolate(
            b,
            size=e2.size(-1),
            mode="linear",
            align_corners=False
        )

        dec2_input = torch.cat([b_up, e2], dim=1)
        dec2 = self.dec2(dec2_input)

        dec2_up = F.interpolate(
            dec2,
            size=e1.size(-1),
            mode="linear",
            align_corners=False
        )

        dec1_input = torch.cat([dec2_up, e1], dim=1)
        dec1 = self.dec1(dec1_input)

        out = self.out_conv(dec1)

        return out


# =====================
# 7. TD-UNetFormerV4-MS-Cal 模型
# =====================

class TDUNetFormerV4MSCal(nn.Module):
    """
    TD-UNetFormerV4-MS-Cal:
    Multi-scale Trend Decomposition + Calendar-aware Seasonal Conditioning + Temporal U-Net + Transformer

    输入:
        x: [B, 90, 13]

    特征顺序来自 preprocess.py:
        0  global_active_power
        1  global_reactive_power
        2  sub_metering_1
        3  sub_metering_2
        4  sub_metering_3
        5  voltage
        6  global_intensity
        7  sub_metering_remainder
        8  RR
        9  NBJRR1
        10 NBJRR5
        11 NBJRR10
        12 NBJBROU
    """

    def __init__(
        self,
        input_dim,
        hidden_dim,
        nhead,
        num_layers,
        dim_feedforward,
        output_len,
        dropout=0.1,
        trend_kernel_sizes=None,
        baseline_weight=None,
        future_calendar_dim=6,
        calendar_weight=None,
        use_future_calendar=True
    ):
        super().__init__()

        # 多尺度趋势分解：默认同时使用 7/15/21 天移动平均。
        # 7 天捕获短期周内变化，15/21 天捕获更平滑的中期趋势。
        if trend_kernel_sizes is None:
            trend_kernel_sizes = [7, 15, 21]
        self.trend_kernel_sizes = [int(k) for k in trend_kernel_sizes]
        if len(self.trend_kernel_sizes) == 0:
            raise ValueError("trend_kernel_sizes 不能为空，例如 7,15,21")
        for k in self.trend_kernel_sizes:
            if k <= 0 or k % 2 == 0:
                raise ValueError("trend_kernel_sizes 建议使用正奇数，例如 7,15,21")

        self.output_len = output_len

        # horizon-aware baseline：
        # 90 天短期预测中，最近 7 天均值通常是强基线；
        # 365 天长期预测中，最近 7 天均值会过度约束全年曲线，因此默认关闭。
        if baseline_weight is None:
            baseline_weight = 1.0 if output_len <= 90 else 0.0
        self.baseline_weight = float(baseline_weight)

        # Calendar-aware seasonal conditioning：
        # 365 天预测中，未来每一天的月份/年内位置是已知信息，
        # 用它做一个轻量季节性修正，比无相位的 Fourier basis 更稳定。
        self.use_future_calendar = bool(use_future_calendar)
        self.future_calendar_dim = int(future_calendar_dim)
        if calendar_weight is None:
            calendar_weight = 0.30 if output_len <= 90 else 0.50
        self.calendar_weight = float(calendar_weight)

        # 原始特征 + 多尺度 trend + 多尺度 residual
        # 例如 7/15/21 三个尺度时，额外增加 3 个 trend 和 3 个 residual。
        augmented_dim = input_dim + 2 * len(self.trend_kernel_sizes)

        self.input_projection = nn.Linear(augmented_dim, hidden_dim)

        self.temporal_unet = TemporalUNet1D(
            hidden_dim=hidden_dim,
            dropout=dropout
        )

        self.positional_encoding = PositionalEncoding(
            d_model=hidden_dim,
            max_len=500
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        # 趋势预测头：mean token + last token 拼接，因此输入维度为 2H
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )

        # 残差预测头：预测相对于最近 7 天均值 baseline 的偏差
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_len)
        )

        if self.use_future_calendar:
            self.future_calendar_projection = nn.Sequential(
                nn.Linear(self.future_calendar_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            self.calendar_context_projection = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.calendar_output_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def moving_average(self, x_power, kernel_size):
        """
        x_power: [B, L, 1]
        return trend: [B, L, 1]
        """

        # [B, L, 1] -> [B, 1, L]
        x = x_power.transpose(1, 2)

        pad = kernel_size // 2

        # 边界复制填充，避免序列两端趋势异常
        x = F.pad(x, (pad, pad), mode="replicate")

        trend = F.avg_pool1d(
            x,
            kernel_size=kernel_size,
            stride=1
        )

        # [B, 1, L] -> [B, L, 1]
        trend = trend.transpose(1, 2)

        return trend

    def forward(self, x, future_calendar=None):
        """
        x: [B, 90, input_dim]
        future_calendar: [B, output_len, 6]，未来日期的周期编码。
        """

        # =====================
        # 1. 趋势-残差分解
        # =====================

        power = x[:, :, 0:1]              # global_active_power

        trends = []
        residuals = []
        for kernel_size in self.trend_kernel_sizes:
            trend_k = self.moving_average(power, kernel_size)
            residual_k = power - trend_k
            trends.append(trend_k)
            residuals.append(residual_k)

        # 多尺度拼接：[原始特征, trend_7, trend_15, trend_21, residual_7, residual_15, residual_21]
        x_aug = torch.cat([x] + trends + residuals, dim=-1)

        # =====================
        # 2. 输入映射
        # =====================

        h = self.input_projection(x_aug)  # [B, 90, H]

        # =====================
        # 3. Temporal U-Net + residual skip
        # =====================
        # 原 V2/V3 直接用 U-Net 输出替换 h，容易把高频波动平滑掉。
        # 这里改成残差连接：保留原始时序表示，同时加入 U-Net 提取的局部/多尺度模式。

        h_conv = h.transpose(1, 2)         # [B, H, 90]
        h_unet = self.temporal_unet(h_conv).transpose(1, 2)  # [B, 90, H]
        h = h + h_unet

        # =====================
        # 4. Transformer Encoder
        # =====================

        h = self.positional_encoding(h)
        encoded = self.transformer(h)

        # mean token 保留整体趋势，last token 保留最近状态
        mean_token = encoded.mean(dim=1)
        last_token = encoded[:, -1, :]
        pooled = torch.cat([mean_token, last_token], dim=-1)

        # =====================
        # 5. horizon-aware baseline + 趋势/残差修正
        # =====================

        pred_trend = self.trend_head(pooled)
        pred_residual = self.residual_head(pooled)

        pred = pred_trend + pred_residual

        # 未来日历条件修正：
        # 每一个未来 horizon 都有自己的 day-of-year / month / weekday 编码，
        # 同时通过 pooled 历史上下文调制，避免所有样本共享同一个固定季节相位。
        if self.use_future_calendar and future_calendar is not None and self.calendar_weight != 0.0:
            cal_emb = self.future_calendar_projection(future_calendar)       # [B, output_len, H]
            context = self.calendar_context_projection(pooled).unsqueeze(1)  # [B, 1, H]
            cal_pred = self.calendar_output_head(cal_emb * context).squeeze(-1)
            pred = pred + self.calendar_weight * cal_pred

        # 注意：此处 x 是标准化后的输入，所以 baseline 也处于标准化空间。
        # 对 90->90 默认启用；对 90->365 默认关闭，避免最近 7 天均值把全年预测拉成常数附近。
        if self.baseline_weight != 0.0:
            baseline = x[:, -7:, 0].mean(dim=1, keepdim=True)
            baseline = baseline.repeat(1, self.output_len)
            pred = pred + self.baseline_weight * baseline

        return pred


# =====================
# 8. 训练与评估
# =====================

def volatility_aware_loss(
    pred,
    y,
    mae_weight=0.2,
    diff_weight=0.2,
    std_weight=0.05,
    peak_weight=0.03,
):
    """
    让预测不要过度保守的损失函数。

    组成：
    1. MSE：保证整体评价指标；
    2. MAE：提升平均绝对误差稳定性；
    3. diff_loss：约束相邻天变化量，鼓励模型学习上升/下降波动；
    4. std_loss：约束每条预测曲线的标准差，避免预测曲线过平；
    5. peak_loss：对偏离样本均值较远的真实点稍微加权，减少峰值/谷值被拉回均值。
    """
    mse = F.mse_loss(pred, y)
    mae = F.l1_loss(pred, y)

    if pred.size(1) > 1:
        pred_diff = pred[:, 1:] - pred[:, :-1]
        true_diff = y[:, 1:] - y[:, :-1]
        diff_loss = F.mse_loss(pred_diff, true_diff)
    else:
        diff_loss = torch.tensor(0.0, device=pred.device)

    pred_std = pred.std(dim=1, unbiased=False)
    true_std = y.std(dim=1, unbiased=False)
    std_loss = F.mse_loss(pred_std, true_std)

    if peak_weight > 0:
        center = y.mean(dim=1, keepdim=True)
        deviation = (y - center).abs()
        norm_deviation = deviation / (deviation.mean(dim=1, keepdim=True) + 1e-6)
        weights = 1.0 + norm_deviation
        peak_loss = (weights * (pred - y) ** 2).mean()
    else:
        peak_loss = torch.tensor(0.0, device=pred.device)

    return (
        mse
        + mae_weight * mae
        + diff_weight * diff_loss
        + std_weight * std_loss
        + peak_weight * peak_loss
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    mae_weight=0.2,
    diff_weight=0.2,
    std_weight=0.05,
    peak_weight=0.03,
):
    model.train()

    total_loss = 0.0

    for batch in loader:
        if len(batch) == 3:
            X, future_calendar, y = batch
            future_calendar = future_calendar.to(DEVICE)
        else:
            X, y = batch
            future_calendar = None

        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(X, future_calendar)
        loss = volatility_aware_loss(
            pred,
            y,
            mae_weight=mae_weight,
            diff_weight=diff_weight,
            std_weight=std_weight,
            peak_weight=peak_weight,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * len(X)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, y_scaler):
    model.eval()

    preds = []
    trues = []

    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                X, future_calendar, y = batch
                future_calendar = future_calendar.to(DEVICE)
            else:
                X, y = batch
                future_calendar = None

            X = X.to(DEVICE)

            pred = model(X, future_calendar).cpu().numpy()
            y = y.cpu().numpy()

            preds.append(pred)
            trues.append(y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    preds_inv = y_scaler.inverse_transform(
        preds.reshape(-1, 1)
    ).reshape(preds.shape)

    trues_inv = y_scaler.inverse_transform(
        trues.reshape(-1, 1)
    ).reshape(trues.shape)

    mse = np.mean((preds_inv - trues_inv) ** 2)
    mae = np.mean(np.abs(preds_inv - trues_inv))

    return mse, mae, preds_inv, trues_inv


def plot_prediction(y_true, y_pred, save_path, title):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true[0], label="Ground Truth")
    plt.plot(y_pred[0], label="Prediction")
    plt.xlabel("Day")
    plt.ylabel("Global Active Power")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


# =====================
# 9. 单次实验
# =====================

def run_experiment(
    train_path,
    test_path,
    output_len,
    seed,
    batch_size=32,
    epochs=100,
    lr=1e-4,
    hidden_dim=64,
    nhead=4,
    num_layers=2,
    dim_feedforward=128,
    dropout=0.1,
    patience=15,
    baseline_weight=None,
    mae_weight=None,
    diff_weight=None,
    std_weight=None,
    peak_weight=None,
    result_dir="results/improved",
    trend_kernel_sizes=None,
    use_calendar=True,
    calendar_weight=None
):
    set_seed(seed)

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    future_calendar_train = None
    future_calendar_test = None

    if use_calendar:
        required_keys = ["x_start_dates", "y_start_dates"]
        for key in required_keys:
            if key not in train_data or key not in test_data:
                raise KeyError(
                    f"当前 npz 缺少 {key}，无法自动构造日历特征。"
                    "请重新运行 preprocess.py 保存 x_start_dates 和 y_start_dates。"
                )

        X_train = append_input_calendar_features(X_train, train_data["x_start_dates"])
        X_test = append_input_calendar_features(X_test, test_data["x_start_dates"])

        future_calendar_train = build_future_calendar_features(
            train_data["y_start_dates"],
            output_len=output_len,
        )
        future_calendar_test = build_future_calendar_features(
            test_data["y_start_dates"],
            output_len=output_len,
        )

        print(f"Calendar features enabled. X feature dim after append: {X_train.shape[-1]}")

    X_train, y_train, X_test, y_test, x_scaler, y_scaler = normalize_data(
        X_train,
        y_train,
        X_test,
        y_test
    )

    train_dataset = PowerDataset(X_train, y_train, future_calendar=future_calendar_train)
    test_dataset = PowerDataset(X_test, y_test, future_calendar=future_calendar_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    input_dim = X_train.shape[-1]

    if mae_weight is None:
        mae_weight = 0.2 if output_len <= 90 else 0.05
    if diff_weight is None:
        diff_weight = 0.20 if output_len <= 90 else 0.08
    if std_weight is None:
        std_weight = 0.05 if output_len <= 90 else 0.02
    if peak_weight is None:
        peak_weight = 0.03 if output_len <= 90 else 0.01

    if trend_kernel_sizes is None:
        trend_kernel_sizes = [7, 15, 21]

    print(
        f"Loss weights | mae={mae_weight}, diff={diff_weight}, "
        f"std={std_weight}, peak={peak_weight}"
    )
    print(f"Trend kernel sizes: {trend_kernel_sizes}")
    print(f"Use calendar: {use_calendar}, calendar_weight={calendar_weight}")

    model = TDUNetFormerV4MSCal(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_len=output_len,
        dropout=dropout,
        baseline_weight=baseline_weight,
        trend_kernel_sizes=trend_kernel_sizes,
        future_calendar_dim=6,
        calendar_weight=calendar_weight,
        use_future_calendar=use_calendar
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            mae_weight=mae_weight,
            diff_weight=diff_weight,
            std_weight=std_weight,
            peak_weight=peak_weight,
        )

        print(
            f"Seed {seed} | Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f}"
        )

    # 固定 epoch 训练结束后，只评估一次 test
    mse, mae, preds, trues = evaluate(
        model,
        test_loader,
        y_scaler
    )

    mse = mse
    mae = mae
    preds = preds
    trues = trues

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Seed {seed} | Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f}"
        )

    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = result_dir / f"td_unetformer_v4_ms_cal_{output_len}_seed_{seed}_predictions.csv"

    save_predictions_csv(
        trues,
        preds,
        prediction_path,
        max_samples=10
    )
    plot_path = result_dir / f"td_unetformer_v4_ms_cal_{output_len}_seed_{seed}.png"

    plot_prediction(
        trues,
        preds,
        plot_path,
        title=f"TD-UNetFormerV4-MS-Cal {output_len}-day Forecast, Seed {seed}"
    )

    return {
        "seed": seed,
        "output_len": output_len,
        "mse": mse,
        "mae": mae,
    }


# =====================
# 10. 主程序
# =====================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        choices=["90", "365"],
        required=True,
        help="Forecast length: 90 or 365"
    )

    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=128)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument(
        "--baseline_weight",
        type=float,
        default=None,
        help="最近7天均值baseline权重；默认90天任务为1.0，365天任务为0.0"
    )
    parser.add_argument(
        "--mae_weight",
        type=float,
        default=None,
        help="混合损失中MAE权重；默认90天任务为0.2，365天任务为0.05"
    )
    parser.add_argument(
        "--diff_weight",
        type=float,
        default=None,
        help="相邻天变化量损失权重；默认90天任务为0.20，365天任务为0.08"
    )
    parser.add_argument(
        "--std_weight",
        type=float,
        default=None,
        help="曲线标准差匹配损失权重；默认90天任务为0.05，365天任务为0.02"
    )
    parser.add_argument(
        "--peak_weight",
        type=float,
        default=None,
        help="峰值/谷值加权MSE损失权重；默认90天任务为0.03，365天任务为0.01"
    )
    parser.add_argument(
        "--trend_kernel_sizes",
        type=str,
        default="7,15,21",
        help="多尺度趋势分解窗口，用逗号分隔，例如 7,15,21"
    )
    parser.add_argument(
        "--use_calendar",
        action="store_true",
        default=True,
        help="启用输入窗口日历特征和未来horizon日历条件，默认启用"
    )
    parser.add_argument(
        "--no_calendar",
        action="store_false",
        dest="use_calendar",
        help="关闭日历特征，用于消融实验"
    )
    parser.add_argument(
        "--calendar_weight",
        type=float,
        default=None,
        help="未来日历条件修正权重；默认90天任务0.3，365天任务0.5"
    )
    args = parser.parse_args()

    output_len = int(args.task)

    trend_kernel_sizes = [
        int(k.strip())
        for k in args.trend_kernel_sizes.split(",")
        if k.strip()
    ]

    if args.hidden_dim % args.nhead != 0:
        raise ValueError("hidden_dim 必须能被 nhead 整除，例如 hidden_dim=64, nhead=4。")

    if output_len == 90:
        train_path = PROCESSED_DIR / "train_90to90.npz"
        test_path = PROCESSED_DIR / "test_90to90.npz"
    else:
        train_path = PROCESSED_DIR / "train_90to365.npz"
        test_path = PROCESSED_DIR / "test_90to365.npz"

    if not train_path.exists():
        raise FileNotFoundError(f"训练文件不存在：{train_path}")

    if not test_path.exists():
        raise FileNotFoundError(f"测试文件不存在：{test_path}")

    result_dir = RESULTS_DIR / f"td_unetformer_v4_ms_cal_{output_len}"

    all_results = []

    print(f"Using device: {DEVICE}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Task: 90 -> {output_len}")
    print(f"Train path: {train_path}")
    print(f"Test path: {test_path}")

    for seed in SEEDS:
        print("=" * 60)
        print(f"Running seed {seed}")

        result = run_experiment(
            train_path=train_path,
            test_path=test_path,
            output_len=output_len,
            seed=seed,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            hidden_dim=args.hidden_dim,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            patience=args.patience,
            baseline_weight=args.baseline_weight,
            mae_weight=args.mae_weight,
            diff_weight=args.diff_weight,
            std_weight=args.std_weight,
            peak_weight=args.peak_weight,
            result_dir=result_dir,
            trend_kernel_sizes=trend_kernel_sizes,
            use_calendar=args.use_calendar,
            calendar_weight=args.calendar_weight
        )

        all_results.append(result)

    df = pd.DataFrame(all_results)

    mse_mean = df["mse"].mean()
    mse_std = df["mse"].std()
    mae_mean = df["mae"].mean()
    mae_std = df["mae"].std()

    summary = pd.DataFrame([
        {
            "model": "TD-UNetFormer-MS-Cal",
            "task": f"90->{output_len}",
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
        }
    ])

    result_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        result_dir / "td_unetformer_v4_ms_cal_each_seed_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        result_dir / "td_unetformer_v4_ms_cal_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFinal Results:")
    print(df)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()