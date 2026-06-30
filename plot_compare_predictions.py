import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# =========================
# 你需要修改的部分
# =========================

# 这里填你的文件路径
# 结构：task -> seed -> {model_name: csv_path}
FILES = {
    "90": {
        2026: {
            "LSTM": "src/results/lstm_90/lstm_90_seed_2026_predictions.csv",
            "Transformer": "results/transformer_90/lstm_90_seed_2026_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_90/td_unetformer_v4_ms_cal_90_seed_2026_predictions.csv",
        },
        2027: {
            "LSTM": "src/results/lstm_90/lstm_90_seed_2027_predictions.csv",
            "Transformer": "results/transformer_90/lstm_90_seed_2027_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_90/td_unetformer_v4_ms_cal_90_seed_2027_predictions.csv",
        },
        2028: {
            "LSTM": "src/results/lstm_90/lstm_90_seed_2028_predictions.csv",
            "Transformer": "results/transformer_90/lstm_90_seed_2028_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_90/td_unetformer_v4_ms_cal_90_seed_2028_predictions.csv",
        },
        2029: {
            "LSTM": "src/results/lstm_90/lstm_90_seed_2029_predictions.csv",
            "Transformer": "results/transformer_90/lstm_90_seed_2029_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_90/td_unetformer_v4_ms_cal_90_seed_2029_predictions.csv",
        },
        2030: {
            "LSTM": "src/results/lstm_90/lstm_90_seed_2030_predictions.csv",
            "Transformer": "results/transformer_90/lstm_90_seed_2030_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_90/td_unetformer_v4_ms_cal_90_seed_2030_predictions.csv",
        },
    },
    "365": {
        2026: {
            "LSTM": "src/results/lstm_365/lstm_365_seed_2026_predictions.csv",
            "Transformer": "results/transformer_365/lstm_365_seed_2026_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_365/td_unetformer_v4_ms_cal_365_seed_2026_predictions.csv",
        },
        2027: {
            "LSTM": "src/results/lstm_365/lstm_365_seed_2027_predictions.csv",
            "Transformer": "results/transformer_365/lstm_365_seed_2027_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_365/td_unetformer_v4_ms_cal_365_seed_2027_predictions.csv",
        },
        2028: {
            "LSTM": "src/results/lstm_365/lstm_365_seed_2028_predictions.csv",
            "Transformer": "results/transformer_365/lstm_365_seed_2028_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_365/td_unetformer_v4_ms_cal_365_seed_2028_predictions.csv",
        },
        2029: {
            "LSTM": "src/results/lstm_365/lstm_365_seed_2029_predictions.csv",
            "Transformer": "results/transformer_365/lstm_365_seed_2029_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_365/td_unetformer_v4_ms_cal_365_seed_2029_predictions.csv",
        },
        2030: {
            "LSTM": "src/results/lstm_365/lstm_365_seed_2030_predictions.csv",
            "Transformer": "results/transformer_365/lstm_365_seed_2030_predictions.csv",
            "TD-UNetFormer-MS-Cal": "results/td_unetformer_v4_ms_cal_365/td_unetformer_v4_ms_cal_365_seed_2030_predictions.csv",
        },
    }
}

OUTPUT_DIR = "../curve_pdfs"
SAMPLE_IDX = 0  # 你要画哪个样本


# =========================
# 工具函数
# =========================
def load_sample(csv_path, sample_idx):
    df = pd.read_csv(csv_path)
    sample_df = df[df["sample_idx"] == sample_idx].copy()
    sample_df = sample_df.sort_values("day")
    return sample_df


def plot_compare_one_seed(task, seed, file_dict, sample_idx, save_png=True):
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
    })

    # 90 天可以稍微小一点，365 天建议更宽
    if task == "365":
        plt.figure(figsize=(16, 7))
    else:
        plt.figure(figsize=(12, 6))

    gt_drawn = False

    for model_name, csv_path in file_dict.items():
        df = load_sample(csv_path, sample_idx)

        if not gt_drawn:
            plt.plot(
                df["day"],
                df["ground_truth"],
                label="Ground Truth",
                linewidth=3.2
            )
            gt_drawn = True

        plt.plot(
            df["day"],
            df["prediction"],
            label=model_name,
            linewidth=2.6
        )

    plt.xlabel("Day", fontweight="bold")
    plt.ylabel("Power", fontweight="bold")
    plt.title(f"Task 90->{task} | Seed {seed} | Sample {sample_idx}", fontweight="bold")

    plt.legend(
        loc="best",
        frameon=True,
        prop={"weight": "bold", "size": 13}
    )

    plt.grid(True, linewidth=0.8)
    plt.tight_layout()

    pdf_path = os.path.join(OUTPUT_DIR, f"compare_{task}_seed_{seed}.pdf")
    plt.savefig(pdf_path, bbox_inches="tight")

    if save_png:
        png_path = os.path.join(OUTPUT_DIR, f"compare_{task}_seed_{seed}.png")
        plt.savefig(png_path, dpi=400, bbox_inches="tight")

    plt.close()


def plot_all():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for task, seed_dict in FILES.items():
        for seed, file_dict in seed_dict.items():
            plot_compare_one_seed(task, seed, file_dict, SAMPLE_IDX, save_png=True)

    print("Done! PDF/PNG saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    plot_all()