import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

def plot_results():

    # Ensure the plots directory exists
    os.makedirs("data/plots", exist_ok=True)

    # Reload results
    results = pd.read_csv("data/processed/mispricing_results.csv")

    # Basic summary
    num_mispriced = results['is_mispriced'].sum()
    print(f"[INFO] Total mispriced options detected: {num_mispriced}")

    # Histogram of price residuals
    plt.figure(figsize=(10, 5))
    sns.histplot(results['price_residual'], bins=50, kde=True)
    plt.title("Histogram of Price Residuals")
    plt.xlabel("Price Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig("data/plots/residual_histogram.png")
    plt.show()

    # Scatter plot: strike vs. price_residual colored by mispricing
    plt.figure(figsize=(10, 5))
    sns.scatterplot(data=results, x='strike', y='price_residual', hue='is_mispriced', palette='Set1')
    plt.title("Strike vs. Price Residual (Mispricing Highlighted)")
    plt.xlabel("Strike Price")
    plt.ylabel("Price Residual")
    plt.legend(title="Mispriced")
    plt.tight_layout()
    plt.savefig("data/plots/strike_vs_residual.png")
    plt.show()
