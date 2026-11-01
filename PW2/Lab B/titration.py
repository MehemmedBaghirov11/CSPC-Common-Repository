"""
PW2 Lab B Part 5 (bonus) -- find a titration's equivalence point.

titration.csv holds a titration curve: pH versus the volume of base added.
The equivalence point is the volume where the pH changes fastest (the steep
jump). Numerically, that is where the SLOPE of the pH curve is largest.
Run:  python titration.py
"""
import numpy as np
import matplotlib.pyplot as plt

# TODO 1: read titration.csv (columns volume_base, pH) into arrays V, pH.
data = np.loadtxt("titration.csv", delimiter=",", skiprows=1)
V = data[:, 0]
pH = data[:, 1]

# TODO 2: compute the slope of the pH curve with np.gradient(pH, V), and find
#         the volume where that slope is largest (np.argmax). That is the
#         equivalence point. Print it.
slope = np.gradient(pH, V)
index = np.argmax(slope)
equivalence_volume = V[index]
print("Equivalence point at about", equivalence_volume, "mL")

# TODO 3: make two plots side by side: (left) pH vs volume with a line at the
#         equivalence point; (right) the slope vs volume, showing it peaks
#         at the equivalence point. Save as titration.png.
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(V, pH)
axes[0].axvline(equivalence_volume, color="red", linestyle="--")
axes[0].set_title("pH curve")
axes[0].set_xlabel("Volume of base (mL)")
axes[0].set_ylabel("pH")

axes[1].plot(V, slope)
axes[1].axvline(equivalence_volume, color="red", linestyle="--")
axes[1].set_title("Slope of pH curve")
axes[1].set_xlabel("Volume of base (mL)")
axes[1].set_ylabel("dpH/dV")

plt.tight_layout()
plt.savefig("titration.png")