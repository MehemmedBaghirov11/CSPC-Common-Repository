import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# ---------- Task 1: load the data ----------
df = pd.read_csv("heart.csv")

print("Shape:", df.shape)
print(df.head())

# keep only the four continuous columns
four = df[["age", "chol", "trestbps", "thalach"]]
print(four.describe())

# ---------- Task 2: shape of each variable ----------
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

top_left = axes[0, 0]
top_right = axes[0, 1]
bottom_left = axes[1, 0]
bottom_right = axes[1, 1]

top_left.hist(df["age"], bins=20)
top_left.set_title("age")

top_right.hist(df["chol"], bins=20)
top_right.set_title("chol")

bottom_left.hist(df["trestbps"], bins=20)
bottom_left.set_title("trestbps")

bottom_right.hist(df["thalach"], bins=20)
bottom_right.set_title("thalach")

plt.tight_layout()
plt.savefig("distributions.png")

# ---------- Task 3: are they normal? ----------
names = ["age", "chol", "trestbps", "thalach"]

# numerical check: Shapiro-Wilk test
print()
print("Shapiro-Wilk test (p < 0.05 means NOT normal):")
for name in names:
    stat, p = stats.shapiro(df[name])
    print(name, "p =", p)

# graphical check: Q-Q plots
fig2, axes2 = plt.subplots(2, 2, figsize=(10, 8))

qq_top_left = axes2[0, 0]
qq_top_right = axes2[0, 1]
qq_bottom_left = axes2[1, 0]
qq_bottom_right = axes2[1, 1]

stats.probplot(df["age"], plot=qq_top_left)
qq_top_left.set_title("age")

stats.probplot(df["chol"], plot=qq_top_right)
qq_top_right.set_title("chol")

stats.probplot(df["trestbps"], plot=qq_bottom_left)
qq_bottom_left.set_title("trestbps")

stats.probplot(df["thalach"], plot=qq_bottom_right)
qq_bottom_right.set_title("thalach")

plt.tight_layout()
plt.savefig("qqplots.png")