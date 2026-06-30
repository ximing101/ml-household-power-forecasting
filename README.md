# Household Power Forecasting

This repository contains the code for the 2026 Professional Master Machine Learning course project.

The task is household electric power consumption forecasting. Given the past 90 days of multivariate household power consumption data, the goal is to predict future daily `global_active_power`.

Two forecasting tasks are considered:

- 90 -> 90 short-term forecasting
- 90 -> 365 long-term forecasting

Three models are compared:

- LSTM
- Transformer
- TD-UNetFormer-MS-Cal

## Repository Structure

```text
.
├── README.md
├── requirements.txt
├── preprocess.py
├── plot_compare_predictions.py
├── processed/
│   ├── processed_daily.csv
│   ├── train_90to90.npz
│   ├── test_90to90.npz
│   ├── train_90to365.npz
│   └── test_90to365.npz
├── src/
│   ├── train_lstm.py
│   ├── train_transformer.py
│   ├── train_improved_v4_ms_cal.py
│   └── results/
│       ├── lstm_90/
│       └── lstm_365/
└── results/
    ├── transformer_90/
    ├── transformer_365/
    ├── td_unetformer_v4_ms_cal_90/
    └── td_unetformer_v4_ms_cal_365/
```

## Dataset

The main dataset is the Individual Household Electric Power Consumption dataset from the UCI Machine Learning Repository.

The weather data is monthly climatological data from the French open data platform data.gouv.fr.

Raw data files are not included in this repository because of file size limits. The processed data files used in this project are provided in the `processed/` directory.

## Environment

Install the required Python packages with:

```bash
pip install -r requirements.txt
```

Main dependencies include:

- NumPy
- Pandas
- Matplotlib
- scikit-learn
- PyTorch

## Data Preprocessing

Run:

```bash
python preprocess.py
```

The preprocessing script aggregates the original minute-level household power consumption data into daily data, merges monthly weather features, constructs sliding-window samples, and saves the processed training and testing files.

The generated processed files are:

```text
processed/train_90to90.npz
processed/test_90to90.npz
processed/train_90to365.npz
processed/test_90to365.npz
```

## Training

Train the LSTM model:

```bash
python src/train_lstm.py --task 90 --epochs 15
python src/train_lstm.py --task 365 --epochs 15
```

Train the Transformer model:

```bash
python src/train_transformer.py --task 90 --epochs 15
python src/train_transformer.py --task 365 --epochs 15
```

Train the improved TD-UNetFormer-MS-Cal model:

```bash
python src/train_improved_v4_ms_cal.py --task 90 --epochs 15
python src/train_improved_v4_ms_cal.py --task 365 --epochs 15
```

Each model is trained with five random seeds:

```text
2026, 2027, 2028, 2029, 2030
```

The final results are reported as the mean and standard deviation over five runs.

## Results

The evaluation metrics are Mean Squared Error (MSE) and Mean Absolute Error (MAE).

| Model | Task | MSE Mean | MSE Std | MAE Mean | MAE Std |
|---|---:|---:|---:|---:|---:|
| LSTM | 90 -> 90 | 248526.66 | 50016.72 | 363.89 | 35.36 |
| Transformer | 90 -> 90 | 223709.95 | 11400.58 | 372.19 | 11.56 |
| TD-UNetFormer-MS-Cal | 90 -> 90 | 190390.80 | 5996.43 | 322.56 | 7.60 |
| LSTM | 90 -> 365 | 268127.06 | 21698.24 | 390.41 | 17.07 |
| Transformer | 90 -> 365 | 233966.95 | 10179.45 | 364.18 | 9.27 |
| TD-UNetFormer-MS-Cal | 90 -> 365 | 167349.38 | 1780.98 | 303.78 | 2.57 |

The prediction curves and detailed output files are stored in the `results/` and `src/results/` directories.

## Model Description

### LSTM

The LSTM model uses the past 90-day multivariate time series as input. The final hidden state of the LSTM encoder is passed to a fully connected regression head to predict future daily `global_active_power`.

### Transformer

The Transformer model first maps input features into a hidden space, adds positional encoding, and then uses a Transformer Encoder to model temporal dependencies in the 90-day input window.

### TD-UNetFormer-MS-Cal

TD-UNetFormer-MS-Cal is the proposed improved model. It combines:

- multi-scale trend decomposition
- Temporal U-Net
- Transformer Encoder
- future calendar feature conditioning
- volatility-aware loss

The model is designed to capture local fluctuations, long-term dependencies, and seasonal patterns in household power consumption.

## Clone

```bash
git clone https://github.com/ximing101/ml-household-power-forecasting.git
cd ml-household-power-forecasting
```
