import numpy as np

def detect_mispricing(autoencoder, X_real, price_index=-1, threshold=0.02):
    X_pred = autoencoder.predict(X_real)
    residuals = np.abs(X_real - X_pred)
    price_residual = residuals[:, price_index]
    mispricing_flags = price_residual / (X_real[:, price_index] + 1e-6) > threshold
    return mispricing_flags, price_residual
