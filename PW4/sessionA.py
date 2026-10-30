import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

os.makedirs("results", exist_ok=True)

# ---------- A2: clean and look ----------
df = pd.read_csv("data/concrete.csv")

print("Rows before cleaning:", len(df))
print("Duplicate rows:", df.duplicated().sum())
print("Missing values per column:")
print(df.isna().sum())

df = df.drop_duplicates().reset_index(drop=True)
print("Rows after cleaning:", len(df))

X = df.drop(columns="strength")
y = df["strength"]

print(df.describe())

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(df["cement"], df["strength"], alpha=0.5)
axes[0].set_xlabel("cement (kg/m3)")
axes[0].set_ylabel("strength (MPa)")
axes[0].set_title("strength vs cement")
axes[1].scatter(df["age"], df["strength"], alpha=0.5)
axes[1].set_xlabel("age (days)")
axes[1].set_ylabel("strength (MPa)")
axes[1].set_title("strength vs age")
plt.tight_layout()
plt.savefig("results/A2_scatter.png")

# ---------- A3: split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train size:", len(X_train), ", Test size:", len(X_test))