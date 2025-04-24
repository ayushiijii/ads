import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.ensemble import IsolationForest

# Set random seed for reproducibility
np.random.seed(42)

# Generate normally distributed data with some extreme outliers
data = np.random.normal(60, 10, 800)  # Mean = 50, Std Dev = 10
data = np.append(data, [80, 150, 200])  # Adding outliers

# Create a DataFrame
df = pd.DataFrame(data, columns=['Value'])

# 1. Outlier Detection using Z-Score Method
z_scores = np.abs(stats.zscore(df))
df_zscore = df[(z_scores < 3).all(axis=1)]  # Keep values within 3 standard deviations
print(f"Original Data Size: {df.shape[0]} | After Z-Score Removal: {df_zscore.shape[0]}")

# 2. Outlier Detection using IQR Method
Q1 = df["Value"].quantile(0.25)
Q3 = df["Value"].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_iqr = df[(df["Value"] >= lower_bound) & (df["Value"] <= upper_bound)]
print(f"After IQR Removal: {df_iqr.shape[0]}")

# Boxplot before removing outliers
plt.figure(figsize=(8, 5))
sns.boxplot(y=df["Value"])
plt.title("Boxplot Before Removing Outliers")
plt.show()

# 3. Outlier Detection using Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
outliers = iso_forest.fit_predict(df)
df_iso = df[outliers == 1]  # Keep only non-outliers
print(f"After Isolation Forest Removal: {df_iso.shape[0]}")

# Boxplot after removing outliers
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_iso["Value"])
plt.title("Boxplot After Removing Outliers")
plt.show()
