import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
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


# =====================
# 2. 路径配置
# =====================

def get_project_dir():
    """
    兼容两种情况：
    1. train_transformer.py 放在项目根目录
    2. train_transformer.py 放在 src/ 目录
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
# 3. Dataset
# =====================

class PowerDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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

    # X 标准化：只用训练集统计均值和标准差
    x_scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1, X_train.shape[-1])
    x_scaler.fit(X_train_2d)

    X_train_scaled = x_scaler.transform(
        X_train.reshape(-1, X_train.shape[-1])
    ).reshape(X_train.shape)

    X_test_scaled = x_scaler.transform(
        X_test.reshape(-1, X_test.shape[-1])
    ).reshape(X_test.shape)

    # y 标准化：预测目标单独标准化
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
# 5. 位置编码
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

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]

        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


# =====================
# 6. Transformer 模型
# =====================

class TransformerForecaster(nn.Module):
    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        output_len,
        dropout=0.1
    ):
        super().__init__()

        self.input_projection = nn.Linear(input_dim, d_model)
        self.positional_encoding = PositionalEncoding(d_model=d_model, max_len=500)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )

        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, output_len)
        )

    def forward(self, x):
        """
        x: [batch, 90, input_dim]
        """

        x = self.input_projection(x)       # [batch, 90, d_model]
        x = self.positional_encoding(x)    # [batch, 90, d_model]

        encoded = self.encoder(x)          # [batch, 90, d_model]

        # 用最后一天的表示做预测
        last_token = encoded[:, -1, :]     # [batch, d_model]

        pred = self.regressor(last_token)  # [batch, output_len]

        return pred


# =====================
# 7. 训练与评估
# =====================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0

    for X, y in loader:
        X = X.to(DEVICE)
        y = y.to(DEVICE)

        optimizer.zero_grad()

        pred = model(X)
        loss = criterion(pred, y)

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
        for X, y in loader:
            X = X.to(DEVICE)

            pred = model(X).cpu().numpy()
            y = y.cpu().numpy()

            preds.append(pred)
            trues.append(y)

    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    # 反标准化，回到原始 global_active_power 尺度
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
    """
    默认画测试集第一个样本的预测曲线
    """
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
# 8. 单次实验
# =====================

def run_experiment(
    train_path,
    test_path,
    output_len,
    seed,
    batch_size=32,
    epochs=100,
    lr=1e-4,
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    patience=20,
    result_dir="results/transformer"
):
    set_seed(seed)

    train_data = np.load(train_path)
    test_data = np.load(test_path)

    X_train = train_data["X"]
    y_train = train_data["y"]
    X_test = test_data["X"]
    y_test = test_data["y"]

    X_train, y_train, X_test, y_test, x_scaler, y_scaler = normalize_data(
        X_train,
        y_train,
        X_test,
        y_test
    )

    train_dataset = PowerDataset(X_train, y_train)
    test_dataset = PowerDataset(X_test, y_test)

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

    model = TransformerForecaster(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        output_len=output_len,
        dropout=dropout
    ).to(DEVICE)

    criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=8
    )


    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion
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
    pred = preds
    true = trues

    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Seed {seed} | Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.6f} | "
            f"Test MSE: {mse:.4f} | Test MAE: {mae:.4f}"
        )


    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = result_dir / f"lstm_{output_len}_seed_{seed}_predictions.csv"

    save_predictions_csv(
        true,
        pred,
        prediction_path,
        max_samples=10
    )

    plot_path = result_dir / f"transformer_{output_len}_seed_{seed}.png"

    plot_prediction(
        true,
        pred,
        plot_path,
        title=f"Transformer {output_len}-day Forecast, Seed {seed}"
    )

    return {
        "seed": seed,
        "output_len": output_len,
        "mse": mse,
        "mae": mae,
    }


# =====================
# 9. 主程序：跑 5 次实验
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

    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dim_feedforward", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=100)

    args = parser.parse_args()

    output_len = int(args.task)

    if args.d_model % args.nhead != 0:
        raise ValueError("d_model 必须能被 nhead 整除，例如 d_model=128, nhead=4。")

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

    result_dir = RESULTS_DIR / f"transformer_{output_len}"

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
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout,
            patience=args.patience,
            result_dir=result_dir
        )

        all_results.append(result)

    df = pd.DataFrame(all_results)

    mse_mean = df["mse"].mean()
    mse_std = df["mse"].std()
    mae_mean = df["mae"].mean()
    mae_std = df["mae"].std()

    summary = pd.DataFrame([
        {
            "model": "Transformer",
            "task": f"90->{output_len}",
            "mse_mean": mse_mean,
            "mse_std": mse_std,
            "mae_mean": mae_mean,
            "mae_std": mae_std,
        }
    ])

    result_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        result_dir / "transformer_each_seed_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    summary.to_csv(
        result_dir / "transformer_summary.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFinal Results:")
    print(df)

    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()