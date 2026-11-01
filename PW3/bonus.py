import numpy as np
import pandas as pd

df = pd.read_csv("heart.csv")

# ---------- Bonus: how unpredictable is a category? ----------
for column in ["target", "sex"]:
    proportions = df[column].value_counts(normalize=True)

    print()
    print("Column:", column)
    print(proportions)

    # entropy in bits: 1.0 = perfectly balanced, 0 = completely predictable
    entropy = 0
    for p in proportions:
        entropy = entropy - p * np.log2(p)

    # the best simple guess is the most common category
    best_guess_accuracy = proportions.max()

    print("Entropy (bits):", entropy)
    print("Accuracy of always guessing the most common category:", best_guess_accuracy)