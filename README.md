# Household Power Forecasting

This project is for the 2026 Machine Learning course project.  
The task is to predict future household global active power using past 90-day multivariate time series.

## Models

- LSTM
- Transformer
- TD-UNetFormer-MS-Cal

## Tasks

- 90 -> 90 short-term forecasting
- 90 -> 365 long-term forecasting

## Dataset

The main dataset is Individual Household Electric Power Consumption from UCI Machine Learning Repository.

Weather data is from data.gouv.fr: Données climatologiques de base - mensuelles.

Raw data files are not included in this repository. Please download them and place them in the project directory before running preprocessing.

## Run

```bash
python preprocess.py
python train_lstm.py --task 90 --epochs 15
python train_lstm.py --task 365 --epochs 15
python train_transformer.py --task 90 --epochs 15
python train_transformer.py --task 365 --epochs 15
python train_improved_v4_ms_cal.py --task 90 --epochs 15
python train_improved_v4_ms_cal.py --task 365 --epochs 15
