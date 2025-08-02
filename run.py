from src.model import build_autoencoder
from src.detect import detect_mispricing
import pandas as pd
import numpy as np

# Load and clean feature data
from src.data_loader import main
main()  # Ensure cleaned data is available

from src.features import generate_features
# Generate features and save to CSV
X, df_feat = generate_features()    

# df = pd.read_csv("data/processed/features.csv")
df = df_feat.copy()  # Use the generated features DataFrame

# Drop non-numeric and label columns (keep only features)
drop_cols = ['price_residual', 'is_mispriced'] if 'price_residual' in df.columns else []
df_numeric = df.select_dtypes(include=[np.number]).drop(columns=drop_cols, errors='ignore')

# Drop rows with missing values
df_numeric = df_numeric.dropna()

# Convert to numpy float32 array
X_real = df_numeric.astype(np.float32).values

# Build and train the autoencoder
autoencoder = build_autoencoder(input_dim=X_real.shape[1], encoding_dim=4)
autoencoder.fit(X_real, X_real, epochs=50, batch_size=32, verbose=1)

# Detect mispricing (assuming 'mid_price' is the last feature in the dataset)
# You can also explicitly pass its index using: df_numeric.columns.get_loc('mid_price')
price_index = df_numeric.columns.get_loc('mid_price')

mispricing_flags, price_residual = detect_mispricing(
    autoencoder, X_real, price_index=price_index, threshold=0.1
)

# Save results with mispricing info
df['price_residual'] = price_residual
df['is_mispriced'] = mispricing_flags.astype(int)
df.to_csv("data/processed/mispricing_results.csv", index=False)

print("[INFO] Mispricing detection complete. Results saved to 'data/processed/mispricing_results.csv'.")

from plot_results import plot_results
# Plot the results
plot_results()