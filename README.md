# Options Mispricing Detection using Deep Autoencoders

This project builds a machine learning pipeline to detect mispriced options contracts in the Indian stock market (Nifty & BankNifty weekly options) using deep autoencoders.

## Project Goals
- Learn fair pricing structure of options via deep learning
- Detect anomalies in real-time via reconstruction errors
- Backtest arbitrage strategies on detected mispricings

## Structure
- `data/` - Raw and processed OHLCV + options data
- `notebooks/` - Jupyter notebooks for analysis and modeling
- `src/` - Scripts for model training, detection, and backtesting
- `results/` - Saved models and evaluation plots

## Model
An unsupervised deep autoencoder is trained on engineered features (IV, delta, gamma, theta, moneyness) to reconstruct valid option price structures. Significant deviations between actual and reconstructed values indicate potential mispricing.

## Requirements
See `requirements.txt`

## Status
🔧 To run just clone this and run the run.py 

---
