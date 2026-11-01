import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

df = pd.read_csv("heart.csv")

# ---------- Task 4: is the effect real? ----------
disease = df[df["target"] == 1]["thalach"]
healthy = df[df["target"] == 0]["thalach"]

print("Group sizes: disease =", len(disease), ", healthy =", len(healthy))
print("Mean thalach, disease:", disease.mean())
print("Mean thalach, healthy:", healthy.mean())

# test 1: t-test (assumes roughly normal data)
t_stat, p_t = stats.ttest_ind(disease, healthy)
print("t-test p =", p_t)

# test 2: Mann-Whitney U (no normality needed)
u_stat, p_u = stats.mannwhitneyu(disease, healthy)
print("Mann-Whitney p =", p_u)

# ---------- Task 4 (part 2): mean with uncertainty ----------
mean_disease = disease.mean()
mean_healthy = healthy.mean()

sem_disease = stats.sem(disease)
sem_healthy = stats.sem(healthy)

print("Disease: mean =", mean_disease, ", SEM =", sem_disease)
print("Healthy: mean =", mean_healthy, ", SEM =", sem_healthy)

# bar plot with error bars (mean +/- 1.96 * SEM = 95% confidence interval)
labels = ["Disease", "Healthy"]
means = [mean_disease, mean_healthy]
errors = [1.96 * sem_disease, 1.96 * sem_healthy]

plt.figure(figsize=(5, 5))
plt.bar(labels, means, yerr=errors, capsize=8)
plt.ylabel("mean thalach (max heart rate)")
plt.title("Mean thalach with 95% confidence interval")
plt.savefig("thalach_groups.png")

# ---------- Task 5: do two things move together? ----------
r_pearson, p_pearson = stats.pearsonr(df["age"], df["thalach"])
r_spearman, p_spearman = stats.spearmanr(df["age"], df["thalach"])

print()
print("Pearson  r =", r_pearson, ", p =", p_pearson)
print("Spearman r =", r_spearman, ", p =", p_spearman)

# scatter plot with a straight trend line
plt.figure(figsize=(6, 5))
plt.scatter(df["age"], df["thalach"], alpha=0.6)

slope, intercept = np.polyfit(df["age"], df["thalach"], 1)
x_line = np.array([df["age"].min(), df["age"].max()])
plt.plot(x_line, slope * x_line + intercept, color="red")

plt.xlabel("age")
plt.ylabel("thalach (max heart rate)")
plt.title("Age vs max heart rate")
plt.savefig("age_thalach.png")