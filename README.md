# CSPC — Computer Science for Physics and Chemistry

My coursework repository for the course.
Each practical lives under `PW<n>/Lab <X>/`.

## Setup

Create and activate the environment for a given lab:

```bash
conda env create -f "PW<n>/Lab <X>/environment.yml"
conda activate cspc
```

Run the tests for a lab from inside its folder:

```bash
cd "PW<n>/Lab <X>"
pytest -v
```

---

## PW1 - Lab A: Reproducible Foundations

**What I built:**
- A radioactive decay simulator comparing a pure-Python loop implementation against a vectorised NumPy implementation.

**Speed comparison (loop vs NumPy):**
- loop  : 2.773 s
- numpy : 0.000 s
- speed-up: 9598.8x faster

**Tests:** all passing?  yes

**Conclusion:**
- The NumPy vectorised version is dramatically faster than the pure-Python loop because it processes all atoms at once instead of looping through them individually. All three tests pass, confirming the simulation follows the expected exponential decay law within tolerance.

---

## PW1 - Lab B: Data, Plotting, and Automation

**What the data showed:**
- The observed decay data closely follows an exponential decrease over time, consistent with radioactive decay behavior.

**Comparison with the analytical law:**
- The observed data and the analytical curve (N0 * exp(-lambda * t)) show a strong visual match in shape, though the observed data has some noise/scatter around the smooth theoretical curve, as expected from real measurements.

**Snakemake pipeline:**
- The Snakefile automates figure generation: running `snakemake --cores 1 figure.png` only regenerates figure.png when decay_observed.csv or plot.py has changed, avoiding unnecessary recomputation.

---

## PW2 - Lab A: Motion from Tracking Data

**Mean acceleration measured:**
- -8.58 m/s^2 (standard deviation: 28.72), compared to the expected -9.81 m/s^2.

**Why the acceleration was noisy:**
- Differentiation amplifies measurement noise: small errors in the smooth position data get magnified once when computing velocity, and magnified again when computing acceleration from velocity. That's why the standard deviation (28.72) is much larger than the mean itself.

**What integrating back showed:**
- Integrating the noisy acceleration back up to velocity and then position recovered a curve very close to the original: the maximum difference was about 0.78 meters. This confirms that integration suppresses noise (the opposite effect of differentiation), since random errors partly cancel out when summed.

---

## PW2 - Lab B: Optimization in Chemistry

**Three optimisation methods (Part 2):**
- On the easy convex function f(x) = (x-3)^2 + 1, gradient descent, Newton and SLSQP all reached x = 3 from x0 = 0, so the methods agree.
- On the harder function g(x) = x^4 - 3x^2 + x + 5 (two minima and one maximum), the methods did not agree. From x0 = 0, gradient descent and SLSQP reached the global minimum at x = -1.30, but Newton landed on x = 0.17, which is a maximum (g'' = -5.65 < 0), because Newton only finds a point where g'(x) = 0. From x0 = 2, gradient descent and Newton found the local minimum at x = 1.13 (g'' = 9.35 > 0), while SLSQP found the global minimum at x = -1.30.
- Lesson: on a simple problem the methods agree, but on a complicated landscape the starting point and the algorithm matter.

**Reaction rate constant (Part 3):**
- Fitting C(t) = C0 * exp(-k*t) to the noisy data gave k = 0.262, close to the expected 0.25, and the fitted curve passes through the data points.

**Chemical equilibrium (Part 4):**
- With K = 15.6, Newton (root-finding) gave x = 0.6638 and SLSQP (minimisation) gave x = 0.6638, so both methods agree.
- Equilibrium composition: H2 = 0.336 mol, I2 = 0.336 mol, HI = 1.328 mol. The reactants fall and the product rises as x increases.

**Titration equivalence point (Part 5, bonus):**
- The slope of the pH curve is largest at about 50.0 mL, so the equivalence point is at about 50 mL.

---

## PW3: Data, Distributions, Testing, and Causation

### Dataset 1: Heart Disease (Cleveland)

Source: Kaggle / UCI Machine Learning Repository. 303 patients, 14 columns, no missing values in the four continuous columns I used (age, chol, trestbps, thalach).

#### Session 1 (Tasks 1-3, code: `session1.py`)

**Task 1, loading the data:**
- Loaded `heart.csv` with pandas. Shape is (303, 14). The four continuous columns are age (mean 54.4), chol (mean 246.7), trestbps (mean 131.7) and thalach (mean 149.6).

**Task 2, shape of each variable (see `distributions.png`):**
- `age`: spread mostly between 40 and 70, with a rough, uneven top around 55-60.
- `chol`: one peak near 200-250 and a long tail to the right, with one extreme value near 560.
- `trestbps`: peak around 120-130 and a tail to the right up to 200.
- `thalach`: peak around 150-170 and a tail to the left down to 70.

**Task 3, normality verdicts:**

I used two checks: the Shapiro-Wilk test (numerical) and Q-Q plots (graphical, see `qqplots.png`). With 303 patients Shapiro-Wilk is very sensitive, and it rejected normality for all four variables (age p = 0.006, chol p = 5.9e-9, trestbps p = 1.8e-6, thalach p = 7.0e-5). So I also used the Q-Q plots to judge how far each variable is from normal.
- `age`: approximately normal. The points follow the line closely, with only small deviations at the extreme ends.
- `thalach`: approximately normal. The points follow the line except for a small deviation in the lower tail (a slight skew to the left).
- `trestbps`: not normal. The upper tail rises above the line, so it is skewed to the right.
- `chol`: not normal. It is clearly skewed to the right, with one extreme value (564) far from the rest.

#### Session 2 (Tasks 4-5, code: `session2.py`)

**Task 4, is the effect real? (see `thalach_groups.png`):**
- I split `thalach` by `target`: 139 patients with heart disease and 164 healthy patients. Mean thalach is 139.3 for the disease group and 158.4 for the healthy group.
- In Session 1, `thalach` was approximately normal on the Q-Q plot but Shapiro-Wilk rejected normality (p = 7.0e-5), so it is a borderline case. Because of this I ran both a t-test (assumes normality) and a Mann-Whitney U test (does not). Both agree: t-test p = 3.5e-14 and Mann-Whitney p = 1.9e-13. The groups genuinely differ: patients with heart disease reach a maximum heart rate about 19 beats per minute lower than healthy patients.
- Uncertainty: I used the standard error of the mean (SEM): 1.92 for the disease group and 1.50 for the healthy group. The bar plot shows each mean with a 95% confidence interval (1.96 x SEM). The two error bars do not overlap (about 135.5-143.0 vs 155.4-161.3), so the picture agrees with the tests.

**Task 5, do age and thalach move together? (see `age_thalach.png`):**
- I computed both Pearson and Spearman correlation, since `thalach` was only borderline normal. They agree: Pearson r = -0.39 (p = 1.1e-12) and Spearman r = -0.39 (p = 1.5e-12).
- Interpretation: age and maximum heart rate move in opposite directions. Older patients tend to reach a lower maximum heart rate. The relationship is moderate (r about -0.4), not strong: the scatter plot shows a clear downward trend but with a wide spread around the line. It is very unlikely to be due to chance.

### Dataset 2: A chemical-exposure mystery (synthetic), Session 2 (Tasks 6-7, code: `chemicals.py`)

**Task 6, the naive analysis (see `naive_chemicals.png`):**
- I measured how strongly each chemical is associated with malignancy using both Pearson and Spearman correlation. They agree: benzene r = 0.38 (Spearman 0.38) and cadmium r = 0.88 (Spearman 0.88), both with p far below 0.05.
- Taken at face value, cadmium appears to be the cause: its association is much stronger, and its scatter plot is a tight rising band, while benzene's is a wide cloud. But correlation alone does not prove causation, so I test this in Task 7.

**Task 7, look again (confounding) (see `within_band.png`):**
- Dataset 2 also contains `pollution_index` and `age`. I checked whether pollution could be a third variable behind the cadmium association. Pollution is almost identical to cadmium (r = 0.98) and strongly associated with malignancy (r = 0.90), but unrelated to benzene (r = 0.05). Age is only weakly related (r = -0.10).
- Method: I restricted the data to patients with similar pollution and recomputed the associations inside that group (pollution_index between 40 and 60, n = 199), then repeated this in several pollution bands (0-20, 20-40, 40-60, 60-80) to check the result is stable.
- Result: cadmium's association dropped from r = 0.88 to about 0.3 (0.24-0.31 in every band), while benzene's association rose from r = 0.38 to about 0.7 (0.68-0.77 in every band). Spearman gives the same picture (inside the 40-60 band: benzene 0.71, cadmium 0.30).
- Conclusion: **benzene is the real cause of malignancy; cadmium only looked guilty.** Cadmium tracks pollution, and pollution is associated with malignancy, so cadmium appeared strongly linked. Once pollution is held roughly constant, the cadmium link mostly disappears while benzene's link stays strong and consistent. Benzene looked weak at first because the pollution differences between patients added noise to its relationship with malignancy. Cadmium keeps a small leftover association (about 0.3) because the pollution bands are not perfectly constant and cadmium is almost the same as pollution. Correlation analysis alone cannot fully prove causation, but the pattern is consistent across all bands.

### Bonus: how unpredictable is a category? (code: `bonus.py`)

- I used `value_counts(normalize=True)` and the Shannon entropy (in bits; 1.0 means perfectly balanced and hardest to guess, 0 means completely predictable), plus the accuracy of always guessing the most common category.
- `target`: 54.1% are 0 (healthy) and 45.9% are 1 (disease). Entropy = 0.995 bits, almost the maximum, so the column is very balanced and hard to guess: always answering "0" is right only 54.1% of the time.
- `sex`: 68.0% are coded 1 and 32.0% are coded 0. Entropy = 0.905 bits, so it is more lopsided and easier to guess: always answering "1" is right 68.0% of the time.

---

## PW4: Machine Learning, Supervised and Unsupervised

### Session A: Concrete strength (code: `PW4/fetch_concrete.py`, `PW4/sessionA.py`)

Dataset: UCI Concrete Compressive Strength, I-Cheng Yeh (1998), "Modeling of strength of high-performance concrete using artificial neural networks", Cement and Concrete Research 28(12). DOI of the dataset: 10.24432/C5PK67. It has 1030 laboratory tests, 8 inputs and 1 output, with no missing values.

**A1, variables and units:**

| name in my code | original name | unit |
|---|---|---|
| cement | Cement | kg/m^3 |
| slag | Blast Furnace Slag | kg/m^3 |
| ash | Fly Ash | kg/m^3 |
| water | Water | kg/m^3 |
| superplastic | Superplasticizer | kg/m^3 |
| coarse_agg | Coarse Aggregate | kg/m^3 |
| fine_agg | Fine Aggregate | kg/m^3 |
| age | Age | day |
| strength (target) | Concrete compressive strength | MPa |

I renamed the columns by position and checked that the order matches the `variables` table printed by `fetch_ucirepo`. The download is kept in `fetch_concrete.py`, which saves `data/concrete.csv`, so `sessionA.py` works without internet.

**A2, cleaning:**
- 25 duplicate rows and 0 missing values were found. After `drop_duplicates()` 1005 rows remain.
- Why duplicates are a problem: after a random split one copy of a row can land in the training set and the other in the test set. The model has then already seen that exact row, so the test score is too optimistic and does not measure how well the model handles new mixes.
- Scales (from `describe()`): `coarse_agg` (mean 974) and `fine_agg` (mean 773) are measured in hundreds, `cement` in hundreds (mean 279), while `superplastic` has a mean of only 6.0. `age` has a mean of 46 but ranges from 1 to 365 days. This matters for the polynomial features and for KNN, which compares raw distances.
- Strength vs cement (see `results/A2_scatter.png`): strength tends to rise with cement, but the points form a wide cloud, so cement alone does not determine strength. For cement above 400 kg/m^3 the strength still ranges from about 25 to 80 MPa, because age and the other ingredients also matter.
- Strength vs age (see `results/A2_scatter.png`): age takes only a few values (1, 3, 7, 14, 28, 56, 90, 100, 180, 270, 365 days), so the points form vertical stripes. Strength rises sharply from the first days (mostly below 40 MPa) to 28 days (up to about 80 MPa) and then does not keep rising: at later ages it stays between about 25 and 80 MPa. This fast growth followed by flattening looks like a logarithmic curve and not a straight line, and the large spread at each age shows that the other ingredients matter too. The bonus later supports this: using log(age) raised the test R2 from 0.58 to 0.80.

**A3, split:**
- `train_test_split(test_size=0.2, random_state=42)` gives 804 training rows and 201 test rows. Every model and scaler is fitted on the training part only.

**A4, linear regression (`results/A4_linear_metrics.csv`, `results/A4_residuals_age.png`):**

strength (MPa) = -23.39 + 0.1167 cement + 0.0981 slag + 0.0846 ash - 0.1314 water + 0.3315 superplastic + 0.01552 coarse_agg + 0.02065 fine_agg + 0.1106 age

The intercept is in MPa, the ingredient coefficients are in MPa per kg/m^3, and the age coefficient is in MPa per day.

| metric | value |
|---|---|
| train RMSE | 10.00 MPa |
| test RMSE | 11.19 MPa |
| test MAE | 8.90 MPa |
| train R2 | 0.610 |
| train adjusted R2 | 0.606 |
| test R2 | 0.580 |

- Signs agree with physics: water lowers strength; cement, slag, ash, superplasticizer and age raise it. Train and test R2 are both low and close, so the model underfits (a straight line is too simple).
- Residuals vs age (log axis): the residuals do not scatter randomly around zero, they form an arch. At 1 to 7 days they are negative (down to about -27 MPa), so the model over-predicts young concrete. From 14 to 100 days they are mostly positive (up to about +35 MPa at 28 days), so the model under-predicts. At 180 to 365 days they are negative again (down to about -24 MPa). A systematic pattern in the residuals means the shape of the model is wrong, not just that the data is noisy: a straight line in age cannot follow the curved growth of strength, so a better-shaped model (for example one using log(age)) can improve it.

**A5, polynomial regression (`results/A5_poly.csv`):**

| degree | p | train RMSE | test RMSE | train R2 | train adj R2 | test R2 |
|---|---|---|---|---|---|---|
| 1 | 8 | 10.00 | 11.19 | 0.610 | 0.606 | 0.580 |
| 2 | 44 | 6.99 | 8.31 | 0.809 | 0.798 | 0.769 |
| 3 | 164 | 4.40 | 5.95 | 0.925 | 0.905 | 0.881 |
| 4 | 494 | 2.33 | 39.41 | 0.979 | 0.945 | -4.206 |

- I would choose degree 3: it has the lowest test RMSE and highest test R2, and the train-test gap is moderate. Degree 2 is simpler but clearly worse on test. (Choosing by the test score makes that score slightly optimistic.)
- At degree 4 there are 494 features for only 804 training rows, so the model can almost memorise the training set: train R2 is 0.979, but test R2 is -4.2, worse than always predicting the mean.
- Adjusted R2 does not protect me: it is computed on the training data and is still 0.945 at degree 4. Only the test set revealed the problem.

**A6, KNN (`results/A6_knn_sweep.csv`, `results/A6_knn_K.png`):**
- Class balance: 64.2% of mixes are below 40 MPa and 35.8% reach it.
- K=7 gave the highest test accuracy (0.836) with a train-test gap of 0.057. K=1 memorises the training set (train 0.998, test 0.831). For large K the model becomes too smooth and both accuracies fall (K=101: 0.757 / 0.736).
- With the scaler 0.836, without 0.821.
- Five splits (random_state 0 to 4): 0.801, 0.851, 0.821, 0.836, 0.806; mean 0.823, std 0.019. The scaler difference (0.015) and the differences between K=1 to 25 are inside this spread, so they are not meaningful. Only K=51 and K=101 are clearly worse. Scaling is still correct in principle, because without it the large-scale columns (aggregates, in hundreds) dominate the distance.

**A6, decision tree (`results/A6_tree_sweep.csv`, `results/A6_tree_depth.png`):**

| max_depth | leaves | train acc | test acc |
|---|---|---|---|
| 1 | 2 | 0.755 | 0.701 |
| 2 | 4 | 0.755 | 0.701 |
| 3 | 8 | 0.840 | 0.796 |
| 4 | 16 | 0.869 | 0.826 |
| 5 | 26 | 0.889 | 0.846 |
| 6 | 38 | 0.930 | 0.841 |
| 8 | 61 | 0.973 | 0.856 |
| None | 80 | 0.998 | 0.881 |

- The train-test gap is about 0.04 to 0.05 up to depth 5 and then grows to 0.09 (depth 6) and 0.12 (depth 8 and None).
- I chose depth 5 (26 leaves): the gap is still small and the tree is short enough to read. The unlimited tree has higher test accuracy (0.881) but 80 leaves, and it is too big to read and nearly memorises the training data. I gain readability and a smaller gap; I lose about 3.5 points of test accuracy.
- Root question: `age <= 42` days. This makes physical sense: young concrete has not finished curing, so age is the first thing that separates strong from weak mixes. The next questions are about cement, water and superplasticizer.

**A7, metrics (`results/A_metrics.csv`, `results/A7_thresholds.csv`):** positive class = reaches 40 MPa.

| model | TP | FP | FN | TN | accuracy | precision | recall | F1 |
|---|---|---|---|---|---|---|---|---|
| KNN K=7 | 54 | 15 | 18 | 114 | 0.836 | 0.783 | 0.750 | 0.766 |
| Tree depth 5 | 58 | 17 | 14 | 112 | 0.846 | 0.773 | 0.806 | 0.789 |

- The dangerous mistake is FP: the model says the mix reaches 40 MPa but it does not, so a weak structure could be built. Precision controls it. A FN only wastes a good mix.
- KNN thresholds (with K=7 the probability can only be a multiple of 1/7, so 0.9 means all 7 neighbours must agree):

| threshold | TP | FP | FN | TN | precision | recall |
|---|---|---|---|---|---|---|
| 0.3 | 61 | 23 | 11 | 106 | 0.726 | 0.847 |
| 0.5 | 54 | 15 | 18 | 114 | 0.783 | 0.750 |
| 0.7 | 39 | 11 | 33 | 118 | 0.780 | 0.542 |
| 0.9 | 18 | 1 | 54 | 128 | 0.947 | 0.250 |

- For a safety-critical use I choose 0.9: only 1 of 19 approved mixes is wrong. The price is recall of 0.25: 54 of the 72 mixes that really reach 40 MPa are rejected. Going from 0.5 to 0.7 is not worth it (precision stays about 0.78 while recall drops a lot). With only 19 approved mixes the 0.947 is uncertain.

**A8, feature importance (`results/A8_importance.csv`, `results/A8_importance.png`):**

| input | raw coef | standardised coef | tree importance | perm. KNN | perm. tree |
|---|---|---|---|---|---|
| cement | 0.117 | 12.12 | 0.253 | 0.165 | 0.170 |
| slag | 0.098 | 8.44 | 0.036 | 0.055 | 0.009 |
| ash | 0.085 | 5.44 | 0.037 | 0.016 | 0.000 |
| water | -0.131 | -2.83 | 0.095 | 0.068 | 0.012 |
| superplastic | 0.331 | 1.97 | 0.164 | 0.066 | 0.188 |
| coarse_agg | 0.016 | 1.20 | 0.011 | 0.038 | 0.002 |
| fine_agg | 0.021 | 1.67 | 0.091 | 0.081 | 0.072 |
| age | 0.111 | 6.89 | 0.313 | 0.176 | 0.165 |

- Raw ranking puts superplastic first and cement third; the standardised ranking puts cement first and superplastic sixth. The raw coefficients have different units, and superplastic varies very little (std about 6 kg/m^3), so a large coefficient per kg/m^3 gives a small effect over its real range. The standardised coefficient multiplies by the standard deviation, so it measures the effect over the typical range.
- The methods mostly agree that cement and age are among the top inputs (tree: age, cement; permutation KNN: age, cement; permutation tree: superplastic, cement, age). Ash is used by the tree (0.037) but its permutation importance is 0.000, so shuffling it does not hurt the test score.
- Engineers' claim: partly consistent. Age and cement rank high. Water ranks only in the middle, but the water-to-cement ratio is not an input, and its effect is shared between water and cement. In the bonus, adding water/binder gave almost no extra gain.
- A high importance does not mean that changing the ingredient would change the strength. Importance only says how much the model relies on the input. The data are observational and the inputs are correlated (mix designers change several together, for example superplasticizer is used to reduce water), exactly the correlation-versus-causation problem from PW3.

**Bonus, physics-chosen features:**

| model | features | test R2 |
|---|---|---|
| plain linear | 8 | 0.580 |
| log(age) | 8 | 0.804 |
| log(age) and water/binder | 8 | 0.806 |
| polynomial degree 2 | 44 | 0.769 |
| polynomial degree 3 | 164 | 0.881 |

- Replacing age by log(age) raised test R2 from 0.58 to 0.80 with no extra features, and it beats the degree-2 polynomial that uses 44 features. The water/binder ratio added only 0.003, which is within noise for a single split. Degree 3 is higher but needs 164 features. A little domain knowledge (strength grows with log of age) gave most of the gain much more cheaply than model flexibility.

### Session B: Wine chemistry (code: `PW4/sessionB.py`)

Dataset: UCI Wine recognition data (Forina et al., PARVUS, Institute of Pharmaceutical and Food Analysis and Technologies, Genoa), loaded with `sklearn.datasets.load_wine`. It has 178 wines from the same region of Italy, from three cultivars (59, 71 and 48 wines), with 13 chemical measurements each and no missing values. The cultivar labels were set aside until B6.

**B1, the measurements:** alcohol, malic acid, ash, alcalinity of ash, magnesium, total phenols, flavanoids, nonflavanoid phenols, proanthocyanins, colour intensity, hue, OD280/OD315 of diluted wines and proline. The largest scales are proline (std 315, range 1402) and magnesium (std 14.3, range 92); the smallest are nonflavanoid phenols (std 0.12), hue (std 0.23) and ash (std 0.27).

**B2, PCA on raw data:** PC1 carries 99.8% of the variance and its loading on proline is 0.9998 (magnesium 0.018, all others below 0.005). PCA does not rescale, and proline has by far the largest variance (std 315 against 14 for magnesium and below 4 for the rest), so the component that maximises variance simply follows proline. I would not accept this PC1 as "the main chemical difference between the wines": it only reflects the units of the measurements.

**B3, scaled PCA (`results/B3_scree.png`, `results/B3_loadings.csv`, `results/B3_variance.csv`):**
- After `StandardScaler` every column has mean 0 and std 1. Explained variance: PC1 0.362, PC2 0.192, PC3 0.111, PC4 0.071, PC5 0.066, PC6 0.049, PC7 0.042, PC8 0.027, PC9 0.022, PC10 0.019, PC11 0.017, PC12 0.013, PC13 0.008. Cumulative: 0.362, 0.554, 0.665, 0.736, 0.802, 0.851, 0.893, 0.920, ...
- 8 components are needed to keep 90% (7 give 89.3%). The PC1-PC2 plane keeps 55.4%.
- PC1 positive side: flavanoids (0.42), total phenols (0.39), OD280/OD315 (0.38), proanthocyanins (0.31), hue (0.30), proline (0.29). Negative side: nonflavanoid phenols (-0.30), malic acid (-0.25), alcalinity of ash (-0.24). So PC1 separates phenol-rich wines (many flavanoids and phenols, high OD280, low malic acid) from phenol-poor wines.
- PC2 positive side: colour intensity (0.53), alcohol (0.48), proline (0.36), ash (0.32), magnesium (0.30). Negative side: hue (-0.28), OD280/OD315 (-0.16). So PC2 separates strong, dark, alcoholic wines from lighter ones.
- In the single-colour PC1-PC2 scatter (`results/B3_pc_scatter.png`) I would guess three groups by eye, and the K-Means result in B6 agrees with this.

**B4, K-Means (`results/B4_elbow.png`, `results/B4_inertia.csv`):**
- (a) With k=3 and `n_init=1`, the inertia depends on the random starting centres: 1277.93 (seeds 0, 2, 3), 1278.76 (seeds 1, 5, 7), 1279.97 (seed 4) and 1282.46 (seed 6). K-Means only finds a local minimum, and a bad start can end in a worse one. `n_init=10` repeats the whole run from 10 different starts and keeps the lowest inertia, so the result no longer depends on one lucky or unlucky start.
- (b) Inertia for k = 1 to 10: 2314.0, 1659.0, 1277.9, 1180.7, 1110.4, 1044.5, 996.0, 944.6, 913.2, 864.6. The drops are large up to k=3 (655, then 381) and small afterwards (97, 70, 66, 49, ...), so the elbow is at **k = 3**. I did not use the cultivars to choose k.
- I cannot choose k by minimising the inertia, because it always decreases when k grows (with k = 178 every wine is its own cluster and the inertia is 0), so the minimum would always select the largest k and say nothing about the real structure.

**B5, the clusters (`results/B5_centres.csv`, `results/B5_clusters.png`):** sizes are 65, 51 and 62. Centres in real units, main differences:

| measurement | cluster 0 | cluster 1 | cluster 2 |
|---|---|---|---|
| alcohol | 12.25 | 13.13 | 13.68 |
| malic acid | 1.90 | 3.31 | 2.00 |
| flavanoids | 2.05 | 0.82 | 3.00 |
| colour intensity | 2.97 | 7.23 | 5.45 |
| hue | 1.06 | 0.69 | 1.07 |
| OD280/OD315 | 2.80 | 1.70 | 3.16 |
| proline | 510 | 619 | 1100 |

- Cluster 0 (65 wines): light wines with the lowest alcohol (12.3), pale colour (intensity 3.0) and low proline (510).
- Cluster 1 (51 wines): dark, phenol-poor wines: highest colour intensity (7.2), most malic acid (3.3), very few flavanoids (0.82), lowest hue (0.69) and OD280 (1.7).
- Cluster 2 (62 wines): rich wines with the highest proline (1100), highest alcohol (13.7) and the most flavanoids (3.0).

**B6, the reveal (`results/B_agreement.csv`):**

| version (k=3) | agreement |
|---|---|
| (i) scaled data | 0.966 |
| (ii) raw, unscaled data | 0.702 |
| (iii) first two PCs of scaled data | 0.966 |

- (i) 172 of 178 wines are in the cluster of their cultivar. Cultivar 0 is cluster 2 (59 of 59), cultivar 1 is cluster 0 (65 of 71) and cultivar 2 is cluster 1 (48 of 48).
- (ii) Raw data gives only 0.702. As in B2, the distances are almost entirely determined by proline (and a little magnesium), because they have the largest numbers, so K-Means effectively clusters on one measurement, and the cultivars overlap in proline. Scaling gives every measurement a comparable weight.
- (iii) The first two PCs give the same score (0.966) as all 13 scaled measurements, even though they keep only 55.4% of the variance (6 wines are misplaced in both cases). The three groups are visible in the leading components; the remaining variance is not needed to separate them.
- K-Means never saw the cultivars, yet it recovers 96.6% of them, so the three grape varieties have clearly different chemical profiles (phenols and flavanoids, colour, proline and alcohol).

**Bonus, what do labels buy you?** Both KNN models (scaled with 13 features, and scaled with 2 PCs) have a test accuracy of 0.933 (42 of 45 test wines). This is lower than the unsupervised agreement (0.966), but the comparison is not completely fair: the agreement is measured on all 178 wines, including those K-Means was fitted on, and each cluster is matched to its best cultivar after seeing the labels, which is optimistic. The supervised accuracy comes from 45 unseen wines in a single split, where one wine is 2.2 percentage points, and the models had to learn from the labels.

### B7: Automating PW4 with Snakemake (`PW4/Snakefile`)

The Snakefile has four rules: `all` (input: `results/A_metrics.csv` and `results/B_agreement.csv`), `fetch` (output: `data/concrete.csv`), `session_a` (inputs: `data/concrete.csv` and `sessionA.py`, output: `results/A_metrics.csv`) and `session_b` (input: `sessionB.py`, output: `results/B_agreement.csv`).

The one command that reproduces all of PW4:

    cd PW4 && snakemake --cores 1

1. **`snakemake -n` (dry run):** it printed the plan without running anything. It planned `session_a` ("Updated input files: sessionA.py", because I had just edited the script) and `all`, but not `fetch` and `session_b`, whose outputs already existed and were newer than their inputs.
2. **After deleting `data/` and `results/` and running `snakemake --cores 1`:** all 4 jobs ran (4 of 4 steps). Snakemake created the folders itself and worked backwards from `all`: `fetch` ran before `session_a`, because `session_a` needs `data/concrete.csv`. `session_b` has no dependency on the other rules, so it could run at any point. The numbers were identical to my manual runs (for example test R2 0.580 and agreement 0.966).
3. **Running the same command again:** Snakemake printed "Nothing to be done (all requested files are present and up to date)". Every output file exists and no input file is newer than its output, so no rule needs to run.
4. **After changing `K_CHOSEN` from 7 to 5 in `sessionA.py`:** only `session_a` (reason: "Updated input files: sessionA.py") and `all` (reason: "Input files updated by another job: results/A_metrics.csv") reran. `fetch` did not rerun because it has no input files and `data/concrete.csv` still exists, so there is nothing that could be newer than its output. `session_b` did not rerun because `sessionB.py` was not changed. I then set `K_CHOSEN` back to 7 and ran Snakemake again so that all results in this README match the final code.