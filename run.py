from src.detect import detect_mispricing

from src.model import build_autoencoder
from src.detect import detect_mispricing
import pandas as pd
import numpy as np

# Load and clean feature data
df = pd.read_csv("data/processed/features.csv")

# Drop any non-numeric columns (e.g., date, symbol, etc.)
df_numeric = df.select_dtypes(include=[np.number])

# Drop rows with NaNs, if any
df_numeric = df_numeric.dropna()

# Convert to numpy float32 array
X_real = df_numeric.astype(np.float32).values

autoencoder = build_autoencoder(input_dim=X_real.shape[1], encoding_dim=4)
autoencoder.fit(X_real, X_real, epochs=50, batch_size=32, verbose=1)

mispricing_flags, price_residual = detect_mispricing(autoencoder, X_real, price_index=-1, threshold=0.02)

df_numeric['price_residual'] = price_residual
df_numeric['is_mispriced'] = mispricing_flags
df_numeric.to_csv("data/processed/mispricing_results.csv", index=False)
print("[INFO] Mispricing detection complete. Results saved to 'mispricing_results.csv'.")