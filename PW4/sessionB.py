import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline

os.makedirs("results", exist_ok=True)

# ---------- B1: load and hide the labels ----------
wine = load_wine(as_frame=True)
X = wine.data               # 178 rows x 13 measurements
cultivar = wine.target      # 0/1/2, not used until B6

print(wine.DESCR)
print(X.describe().T)
spread = pd.DataFrame({"range": X.max() - X.min(), "std": X.std()})
print()
print(spread.sort_values("std", ascending=False))

# ---------- B2: PCA on the raw data ----------
pca_raw = PCA()
pca_raw.fit(X)
print()
print("===== B2: PCA on raw data =====")
print("Explained variance ratio of PC1:", pca_raw.explained_variance_ratio_[0])
load_raw = pd.Series(pca_raw.components_[0], index=X.columns)
print("PC1 loadings (raw), sorted:")
print(load_raw.sort_values())

# ---------- B3: scale, then PCA again ----------
scaler = StandardScaler()
Z = scaler.fit_transform(X)
print()
print("===== B3: scaled data =====")
print("Z mean:", Z.mean(axis=0).round(6))
print("Z std :", Z.std(axis=0).round(6))

pca = PCA()
scores = pca.fit_transform(Z)
ratio = pca.explained_variance_ratio_
cum = np.cumsum(ratio)
table = pd.DataFrame({"component": ["PC" + str(i + 1) for i in range(len(ratio))],
                      "ratio": ratio, "cumulative": cum})
print(table.to_string(index=False))
table.to_csv("results/B3_variance.csv", index=False)
print("Components needed for 90%:", int(np.argmax(cum >= 0.9)) + 1)
print("PC1+PC2 keep:", cum[1])

# scree plot
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(range(1, 14), ratio, label="each component")
ax.plot(range(1, 14), cum, color="red", marker="o", label="cumulative")
ax.axhline(0.9, color="black", linestyle="--", label="0.9")
ax.set_xlabel("component")
ax.set_ylabel("explained variance ratio")
ax.set_title("Scree plot (scaled wine data)")
ax.legend()
plt.tight_layout()
plt.savefig("results/B3_scree.png")
plt.close()

# loadings of PC1 and PC2
loadings = pd.DataFrame({"PC1": pca.components_[0], "PC2": pca.components_[1]},
                        index=X.columns)
print()
print("PC1 loadings, sorted:")
print(loadings["PC1"].sort_values())
print("PC2 loadings, sorted:")
print(loadings["PC2"].sort_values())
loadings.to_csv("results/B3_loadings.csv")

# scatter in one colour
fig, ax = plt.subplots(figsize=(6, 5))
ax.scatter(scores[:, 0], scores[:, 1], alpha=0.7)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Wines on the PC1-PC2 plane (one colour)")
plt.tight_layout()
plt.savefig("results/B3_pc_scatter.png")
plt.close()

# ---------- B4: K-Means ----------
print()
print("===== B4: K-Means =====")
print("(a) k=3, n_init=1, different random_state:")
for seed in range(8):
    km1 = KMeans(n_clusters=3, init="k-means++", n_init=1, max_iter=300, random_state=seed)
    km1.fit(Z)
    print("  random_state", seed, ": inertia =", km1.inertia_)

print("(b) inertia for k = 1..10 (n_init=10, random_state=0):")
ks = list(range(1, 11))
inertias = []
for k in ks:
    kmk = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=0)
    kmk.fit(Z)
    inertias.append(kmk.inertia_)
    print("  k =", k, ": inertia =", kmk.inertia_)
pd.DataFrame({"k": ks, "inertia": inertias}).to_csv("results/B4_inertia.csv", index=False)

fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(ks, inertias, marker="o")
ax.set_xlabel("k")
ax.set_ylabel("inertia")
ax.set_title("Elbow plot")
plt.tight_layout()
plt.savefig("results/B4_elbow.png")
plt.close()

# >>>>> CHOOSE k from the elbow plot, then run again <<<<<
K_CHOSEN = 3

km = KMeans(n_clusters=K_CHOSEN, init="k-means++", n_init=10, max_iter=300, random_state=0)
km.fit(Z)

# ---------- B5: what are the clusters? ----------
print()
print("===== B5: clusters, k =", K_CHOSEN, "=====")
print("Cluster sizes:", np.bincount(km.labels_))
centres = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=X.columns)
print(centres.T.round(2))
centres.round(3).to_csv("results/B5_centres.csv", index_label="cluster")

fig, ax = plt.subplots(figsize=(6, 5))
sc = ax.scatter(scores[:, 0], scores[:, 1], c=km.labels_, alpha=0.8)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("K-Means clusters on the PC1-PC2 plane")
plt.tight_layout()
plt.savefig("results/B5_clusters.png")
plt.close()

# ---------- B6: the reveal ----------
print()
print("===== B6: agreement with the true cultivars (k=3) =====")

def agreement(name, data):
    model = KMeans(n_clusters=3, init="k-means++", n_init=10, max_iter=300, random_state=0)
    model.fit(data)
    tab = pd.crosstab(cultivar, model.labels_)
    score = tab.max(axis=0).sum() / len(cultivar)
    print()
    print(name)
    print(tab)
    print("agreement =", score)
    return score

Z2 = PCA(n_components=2).fit_transform(Z)
agree = pd.DataFrame({
    "version": ["scaled data", "raw data", "first 2 PCs of scaled data"],
    "agreement": [agreement("(i) scaled data", Z),
                  agreement("(ii) raw, unscaled data", X),
                  agreement("(iii) first two PCs of scaled data", Z2)],
})
print()
print(agree)
agree.to_csv("results/B_agreement.csv", index=False)

# ---------- Bonus: what do labels buy you? ----------
print()
print("===== Bonus =====")
X_tr, X_te, y_tr, y_te = train_test_split(X, cultivar, test_size=0.25,
                                          random_state=42, stratify=cultivar)
m1 = make_pipeline(StandardScaler(), KNeighborsClassifier(5))
m2 = make_pipeline(StandardScaler(), PCA(n_components=2), KNeighborsClassifier(5))
m1.fit(X_tr, y_tr)
m2.fit(X_tr, y_tr)
print("KNN, scaled, 13 features   : test accuracy =", m1.score(X_te, y_te))
print("KNN, scaled, 2 PCs         : test accuracy =", m2.score(X_te, y_te))