import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df2 = pd.read_csv("chemicals_cancer.csv")

print("Shape:", df2.shape)
print(df2.head())

# ---------- Task 6: the naive analysis ----------
for chemical in ["benzene", "cadmium"]:
    r_p, p_p = stats.pearsonr(df2[chemical], df2["malignancy"])
    r_s, p_s = stats.spearmanr(df2[chemical], df2["malignancy"])
    print()
    print(chemical, "vs malignancy:")
    print("  Pearson  r =", r_p, ", p =", p_p)
    print("  Spearman r =", r_s, ", p =", p_s)

# two scatter plots side by side
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

left = axes[0]
right = axes[1]

left.scatter(df2["benzene"], df2["malignancy"], alpha=0.5)
left.set_xlabel("benzene")
left.set_ylabel("malignancy")
left.set_title("benzene vs malignancy")

right.scatter(df2["cadmium"], df2["malignancy"], alpha=0.5)
right.set_xlabel("cadmium")
right.set_ylabel("malignancy")
right.set_title("cadmium vs malignancy")

plt.tight_layout()
plt.savefig("naive_chemicals.png")

# ---------- Task 7: look again ----------
print()
print("Correlation of pollution_index with the other columns:")
for col in ["benzene", "cadmium", "malignancy", "age"]:
    r, p = stats.pearsonr(df2["pollution_index"], df2[col])
    print("  pollution_index vs", col, ": r =", r)

# keep only patients with similar pollution (between 40 and 60)
band = df2[(df2["pollution_index"] > 40) & (df2["pollution_index"] < 60)]

print()
print("Patients with pollution_index between 40 and 60:", len(band))
for chemical in ["benzene", "cadmium"]:
    r_p, p_p = stats.pearsonr(band[chemical], band["malignancy"])
    r_s, p_s = stats.spearmanr(band[chemical], band["malignancy"])
    print(chemical, "vs malignancy inside the band:")
    print("  Pearson  r =", r_p, ", p =", p_p)
    print("  Spearman r =", r_s, ", p =", p_s)

# repeat for several pollution bands to check the result is not a fluke
print()
print("Pearson r inside each pollution band:")
edges = [0, 20, 40, 60, 80, 100]
for i in range(len(edges) - 1):
    low = edges[i]
    high = edges[i + 1]
    part = df2[(df2["pollution_index"] >= low) & (df2["pollution_index"] < high)]
    if len(part) > 10:
        r_b, p_b = stats.pearsonr(part["benzene"], part["malignancy"])
        r_c, p_c = stats.pearsonr(part["cadmium"], part["malignancy"])
        print("  band", low, "-", high, ": n =", len(part),
              ", benzene r =", round(r_b, 3), ", cadmium r =", round(r_c, 3))

# plot inside the narrow band
fig2, axes2 = plt.subplots(1, 2, figsize=(11, 5))

band_left = axes2[0]
band_right = axes2[1]

band_left.scatter(band["benzene"], band["malignancy"], alpha=0.6)
band_left.set_xlabel("benzene")
band_left.set_ylabel("malignancy")
band_left.set_title("benzene vs malignancy (pollution 40-60)")

band_right.scatter(band["cadmium"], band["malignancy"], alpha=0.6)
band_right.set_xlabel("cadmium")
band_right.set_ylabel("malignancy")
band_right.set_title("cadmium vs malignancy (pollution 40-60)")

plt.tight_layout()
plt.savefig("within_band.png")