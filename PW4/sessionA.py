import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.inspection import permutation_importance
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             confusion_matrix, accuracy_score,
                             precision_score, recall_score, f1_score)

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
plt.close()

# ---------- A3: split ----------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print("Train size:", len(X_train), ", Test size:", len(X_test))

# ---------- A4: linear regression ----------
lin = LinearRegression()
lin.fit(X_train, y_train)

print()
print("Intercept (MPa):", lin.intercept_)
for name, coef in zip(X.columns, lin.coef_):
    print("  ", name, ":", coef)

pred_train = lin.predict(X_train)
pred_test = lin.predict(X_test)

n = len(X_train)
p = X_train.shape[1]

train_rmse = np.sqrt(mean_squared_error(y_train, pred_train))
test_rmse = np.sqrt(mean_squared_error(y_test, pred_test))
test_mae = mean_absolute_error(y_test, pred_test)
train_r2 = r2_score(y_train, pred_train)
train_adj_r2 = 1 - (1 - train_r2) * (n - 1) / (n - p - 1)
test_r2 = r2_score(y_test, pred_test)

table = pd.DataFrame({
    "metric": ["train RMSE (MPa)", "test RMSE (MPa)", "test MAE (MPa)",
               "train R2", "train adjusted R2", "test R2"],
    "value": [train_rmse, test_rmse, test_mae,
              train_r2, train_adj_r2, test_r2]
})
print()
print(table)
table.to_csv("results/A4_linear_metrics.csv", index=False)

residuals = y_test - pred_test
fig2, ax = plt.subplots(figsize=(6, 4))
ax.scatter(X_test["age"], residuals, alpha=0.6)
ax.axhline(0, color="red", linestyle="--")
ax.set_xscale("log")
ax.set_xlabel("age (days, log scale)")
ax.set_ylabel("residual = true - predicted (MPa)")
ax.set_title("Linear model: test residuals vs age")
plt.tight_layout()
plt.savefig("results/A4_residuals_age.png")
plt.close()

# =====================================================================
# A5: polynomial regression
# =====================================================================
print()
print("===== A5: polynomial regression =====")
rows = []
for d in [1, 2, 3, 4]:
    poly = make_pipeline(PolynomialFeatures(degree=d, include_bias=False),
                         StandardScaler(),
                         LinearRegression())
    poly.fit(X_train, y_train)
    p_feat = poly[0].n_output_features_
    pr_train = poly.predict(X_train)
    pr_test = poly.predict(X_test)
    r2_tr = r2_score(y_train, pr_train)
    rows.append({
        "degree": d,
        "p (features)": p_feat,
        "train RMSE": np.sqrt(mean_squared_error(y_train, pr_train)),
        "test RMSE": np.sqrt(mean_squared_error(y_test, pr_test)),
        "train R2": r2_tr,
        "train adj R2": 1 - (1 - r2_tr) * (n - 1) / (n - p_feat - 1),
        "test R2": r2_score(y_test, pr_test),
    })
poly_table = pd.DataFrame(rows)
print(poly_table.to_string(index=False))
poly_table.to_csv("results/A5_poly.csv", index=False)

# =====================================================================
# A6: classification, KNN and decision tree
# =====================================================================
print()
print("===== A6: classification =====")
y_class = (df["strength"] >= 40).astype(int)   # 1 = reaches 40 MPa
print(y_class.value_counts(normalize=True))

Xc_train, Xc_test, c_train, c_test = train_test_split(
    X, y_class, test_size=0.2, random_state=42, stratify=y_class)

# ---- KNN sweep ----
Ks = [1, 3, 5, 7, 15, 25, 51, 101]
knn_rows = []
for K in Ks:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K))
    knn.fit(Xc_train, c_train)
    knn_rows.append({"K": K,
                     "train acc": knn.score(Xc_train, c_train),
                     "test acc": knn.score(Xc_test, c_test)})
knn_table = pd.DataFrame(knn_rows)
print(knn_table.to_string(index=False))
knn_table.to_csv("results/A6_knn_sweep.csv", index=False)

fig3, ax3 = plt.subplots(figsize=(6, 4))
ax3.plot(knn_table["K"], knn_table["train acc"], marker="o", label="train")
ax3.plot(knn_table["K"], knn_table["test acc"], marker="o", label="test")
ax3.set_xscale("log")
ax3.set_xlabel("K (log scale)")
ax3.set_ylabel("accuracy")
ax3.set_title("KNN: accuracy vs K")
ax3.legend()
plt.tight_layout()
plt.savefig("results/A6_knn_K.png")
plt.close()

# >>>>> CHOOSE K after reading the table above, then run the script again <<<<<
K_CHOSEN = 7

knn_best = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_CHOSEN))
knn_best.fit(Xc_train, c_train)
knn_noscale = KNeighborsClassifier(n_neighbors=K_CHOSEN)
knn_noscale.fit(Xc_train, c_train)
print()
print("K =", K_CHOSEN)
print("  test accuracy WITH scaler   :", knn_best.score(Xc_test, c_test))
print("  test accuracy WITHOUT scaler:", knn_noscale.score(Xc_test, c_test))

print("Five different splits (random_state 0..4):")
accs = []
for seed in [0, 1, 2, 3, 4]:
    Xa, Xb, ca, cb = train_test_split(X, y_class, test_size=0.2,
                                      random_state=seed, stratify=y_class)
    m = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=K_CHOSEN))
    m.fit(Xa, ca)
    accs.append(m.score(Xb, cb))
    print("  seed", seed, ": test accuracy =", accs[-1])
print("  mean =", np.mean(accs), ", std =", np.std(accs),
      ", min =", np.min(accs), ", max =", np.max(accs))

# ---- decision tree sweep ----
depths = [1, 2, 3, 4, 5, 6, 8, None]
tree_rows = []
for dep in depths:
    t = DecisionTreeClassifier(criterion="entropy", max_depth=dep,
                               min_samples_leaf=1, random_state=0)
    t.fit(Xc_train, c_train)
    tree_rows.append({"max_depth": str(dep),
                      "leaves": t.get_n_leaves(),
                      "train acc": t.score(Xc_train, c_train),
                      "test acc": t.score(Xc_test, c_test)})
tree_table = pd.DataFrame(tree_rows)
print()
print(tree_table.to_string(index=False))
tree_table.to_csv("results/A6_tree_sweep.csv", index=False)

fig4, ax4 = plt.subplots(figsize=(6, 4))
ax4.plot(tree_table["max_depth"], tree_table["train acc"], marker="o", label="train")
ax4.plot(tree_table["max_depth"], tree_table["test acc"], marker="o", label="test")
ax4.set_xlabel("max_depth")
ax4.set_ylabel("accuracy")
ax4.set_title("Decision tree: accuracy vs depth")
ax4.legend()
plt.tight_layout()
plt.savefig("results/A6_tree_depth.png")
plt.close()

# >>>>> CHOOSE the depth after reading the table above, then run again <<<<<
DEPTH_CHOSEN = 5

tree_best = DecisionTreeClassifier(criterion="entropy", max_depth=DEPTH_CHOSEN,
                                   min_samples_leaf=1, random_state=0)
tree_best.fit(Xc_train, c_train)
print()
print("Chosen tree: depth =", tree_best.get_depth(), ", leaves =", tree_best.get_n_leaves())
print(export_text(tree_best, feature_names=list(X.columns), max_depth=2))

# =====================================================================
# A7: the metrics that matter
# =====================================================================
print()
print("===== A7: metrics =====")

def report(name, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {"model": name, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "F1": f1_score(y_true, y_pred)}

metrics = pd.DataFrame([
    report("KNN K=" + str(K_CHOSEN), c_test, knn_best.predict(Xc_test)),
    report("Tree depth=" + str(DEPTH_CHOSEN), c_test, tree_best.predict(Xc_test)),
])
print(metrics.to_string(index=False))
metrics.to_csv("results/A_metrics.csv", index=False)

proba = knn_best.predict_proba(Xc_test)[:, 1]
thr_rows = []
for thr in [0.3, 0.5, 0.7, 0.9]:
    pred_t = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(c_test, pred_t).ravel()
    thr_rows.append({"threshold": thr, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
                     "precision": precision_score(c_test, pred_t, zero_division=0),
                     "recall": recall_score(c_test, pred_t)})
thr_table = pd.DataFrame(thr_rows)
print()
print(thr_table.to_string(index=False))
thr_table.to_csv("results/A7_thresholds.csv", index=False)

# =====================================================================
# A8: feature importance
# =====================================================================
print()
print("===== A8: feature importance =====")
pipe = make_pipeline(StandardScaler(), LinearRegression())
pipe.fit(X_train, y_train)

perm_knn = permutation_importance(knn_best, Xc_test, c_test, scoring="f1",
                                  n_repeats=20, random_state=0)
perm_tree = permutation_importance(tree_best, Xc_test, c_test, scoring="f1",
                                   n_repeats=20, random_state=0)

imp = pd.DataFrame({
    "raw coef": lin.coef_,
    "standardised coef": pipe[-1].coef_,
    "tree importance": tree_best.feature_importances_,
    "perm KNN": perm_knn.importances_mean,
    "perm tree": perm_tree.importances_mean,
}, index=X.columns)
print(imp)
imp.to_csv("results/A8_importance.csv")

fig5, axes5 = plt.subplots(1, 3, figsize=(14, 4))
imp["standardised coef"].plot.barh(ax=axes5[0], title="standardised coef (MPa per SD)")
imp["tree importance"].plot.barh(ax=axes5[1], title="tree importance")
imp["perm KNN"].plot.barh(ax=axes5[2], title="permutation importance, KNN (F1 drop)")
plt.tight_layout()
plt.savefig("results/A8_importance.png")
plt.close()

# =====================================================================
# Bonus: let physics choose the features
# =====================================================================
print()
print("===== Bonus =====")
def fit_linear(Xtr, Xte, label):
    m = LinearRegression()
    m.fit(Xtr, y_train)
    r2 = r2_score(y_test, m.predict(Xte))
    print(label, ": features =", Xtr.shape[1], ", test R2 =", r2)

fit_linear(X_train, X_test, "plain linear           ")

Xtr_log = X_train.copy()
Xte_log = X_test.copy()
Xtr_log["age"] = np.log(Xtr_log["age"])
Xte_log["age"] = np.log(Xte_log["age"])
fit_linear(Xtr_log, Xte_log, "log(age)               ")

Xtr_wb = Xtr_log.copy()
Xte_wb = Xte_log.copy()
Xtr_wb["water"] = X_train["water"] / (X_train["cement"] + X_train["slag"] + X_train["ash"])
Xte_wb["water"] = X_test["water"] / (X_test["cement"] + X_test["slag"] + X_test["ash"])
Xtr_wb = Xtr_wb.rename(columns={"water": "w_over_b"})
Xte_wb = Xte_wb.rename(columns={"water": "w_over_b"})
fit_linear(Xtr_wb, Xte_wb, "log(age) and water/binder")