# 📉 Options Mispricing Detection using Deep Autoencoders

This project aims to detect mispriced **Nifty** and **BankNifty** options contracts using a machine learning pipeline built on **deep autoencoders**. It leverages data from the Indian derivatives market to construct a fair-pricing structure and flags anomalies through unsupervised learning.

---

## 🎯 Project Goals

- Learn the latent structure of fairly priced options using deep learning
- Detect anomalies (i.e., mispricings) via reconstruction error
- Backtest arbitrage or hedging strategies based on model-flagged opportunities

---

## 📂 Project Structure

```bash
options-mispricing/
│
├── data/
│   ├── raw/                      # Raw downloaded option chain data
│   └── processed/                # Cleaned options and engineered feature data
│
├── notebooks/                   # Jupyter notebooks for EDA and modeling
│
├── results/                     # Saved model weights, plots, mispricing logs
│
├── src/                         # Main source code
│   ├── fetch_and_clean.py       # Data download and cleaning
│   ├── feature_engineering.py   # Feature computation from raw data
│   ├── build_autoencoder.py     # Autoencoder training + evaluation
│   ├── detect_mispricing.py     # Mispricing inference + logging
│   └── config.py                # Paths, hyperparameters, constants
│
├── run.py                       # One-click runner to execute full pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
````

---

## 🧾 Pipeline Summary

### 1️⃣ `fetch_and_clean.py` – *Fetch Raw Option Chain & Clean*

* Downloads **Call option chains** from Yahoo Finance via `nsepy`
* Filters out:

  * Illiquid contracts (zero bid/ask)
  * Very low volume
  * Deep ITM (optional)
* Combines all expiries into a single file
* ⏎ **Output:** `data/processed/cleaned_options_all_expiries.csv`

### 2️⃣ `feature_engineering.py` – *Add Predictive Features*

* Loads cleaned data and adds features for modeling:

  * Moneyness: how far the strike is from spot
  * Time to expiry
  * Bid/ask mid price (used as label)
  * Ratios like IV/Open Interest
* ⏎ **Output:** `data/processed/features.csv`

### 3️⃣ `build_autoencoder.py` – *Train Pricing Model*

Trains an unsupervised deep autoencoder on engineered features.
The objective is to learn the structure of valid options pricing relationships.

#### 🔧 Model Details:

* Architecture:

  * Encoder → Latent Bottleneck → Decoder
* Input:

  * Features like moneyness, IV, time-to-expiry, etc.
* Loss:

  * MSE between actual features and reconstruction
* Regularization:

  * Dropout or L2 (optional)

🧠 **Interpretation**:
If the model struggles to reconstruct a given option's features (i.e., high reconstruction error), the contract is potentially mispriced relative to the historical pricing structure.

### 4️⃣ `detect_mispricing.py` – *Flag Anomalies*

Loads the trained autoencoder and:

* Normalizes new feature data (scaler saved during training)
* Runs through the autoencoder
* Computes **reconstruction error**
* Flags top N% options with highest errors as potential mispricings
* Saves flagged options to a CSV with timestamps

🛠 You can adjust:

* Thresholds
* Time intervals
* Backtest strategy triggers

---

## 🧠 Engineered Features Explained

| Feature             | Description                                           |
| ------------------- | ----------------------------------------------------- |
| `strike`            | Option strike price                                   |
| `impliedVolatility` | Implied volatility from market (Yahoo Finance)        |
| `moneyness`         | $(K - S) / S$, where $K$ is strike and $S$ is spot    |
| `time_to_expiry`    | In **years**, based on contract expiry                |
| `mid_price`         | $(\text{Bid} + \text{Ask}) / 2$, proxy for fair value |
| `volume`            | Contracts traded                                      |
| `openInterest`      | Total open contracts                                  |
| `iv_oi_ratio`       | IV / Open Interest                                    |
| `volume_oi_ratio`   | Volume / Open Interest                                |

---

## 💡 What is Moneyness?

**Moneyness** indicates how far the strike price is from the current spot price. It helps standardize options across different strikes:

$$
\text{Moneyness} = \frac{\text{Strike} - \text{Spot}}{\text{Spot}}
$$

* **> 0**: OTM (Out-of-the-money)
* **< 0**: ITM (In-the-money)

This helps autoencoders generalize across different market regimes.

---

## 🧪 Model Usage Summary

```bash
# Step 1: Run full pipeline
python run.py

# Step 2: Train autoencoder
python src/build_autoencoder.py

# Step 3: Detect mispricings on new data
python src/detect_mispricing.py
```

---

## 📈 Output Logs

* `results/autoencoder_model.h5`: Trained model weights
* `results/scaler.pkl`: MinMaxScaler used for normalization
* `results/mispricings.csv`: Mispriced options detected (with error scores)

---

## ⚠️ Assumptions & Limitations

* Only **Call options** are used for now (Puts can be added similarly)
* Spot price is taken live at feature generation time
* Implied volatility comes from Yahoo/NSE and may be noisy
* Not a trading strategy — only flags potential pricing inefficiencies

---

## 📚 References

* [NSE Derivatives Market](https://www.nseindia.com/products-services/equity-derivatives)
* [Options Pricing & Greeks](https://en.wikipedia.org/wiki/Options_greeks)
* [Autoencoders for Anomaly Detection](https://arxiv.org/abs/1901.00596)
* [nsepy Python Package](https://pypi.org/project/nsepy/)

---

## 👨‍💻 Author

**Hemanth Madduri**
Aerospace Engineer @ IIT Madras
🛰️ Specializing in Data Assimilation and ML-driven modeling

---

## 🛠 Requirements

```bash
pip install -r requirements.txt
```

* `numpy`, `pandas`, `scikit-learn`
* `tensorflow` or `keras`
* `nsepy`
* `matplotlib`, `seaborn` (for plotting)

```

---

Let me know if you'd like to add a section for **backtesting**, **alerts**, or **deployment (e.g. streamlit)** — I can structure it for real-time use too.
```
