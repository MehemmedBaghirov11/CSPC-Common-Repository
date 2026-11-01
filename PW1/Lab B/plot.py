"""
PW1 Lab B -- read observed decay data and compare it to the analytical law.
Produce a 1x2 figure:  left = observed data,  right = analytical N0*exp(-lam*t).
with SHARED axes so the two shapes are directly comparable.

Complete the TODOs below. Run with:  python plot.py
"""
import numpy as np
import matplotlib.pyplot as plt

LAMBDA = 0.3    # decay constant, given

# TODO 1: read decay_observed.csv (columns: time, count; skip the header row)
#         and split it into two arrays: t and observed.
data = np.loadtxt("decay_observed.csv", delimiter=",", skiprows=1)
t = data[:, 0]
observed = data[:, 1]

# TODO 2: set N0 to the FIRST observed value, then build the analytical curve
#         analytical = N0 * exp(-LAMBDA * t)
N0 = observed[0]
analytical = N0 * np.exp(-LAMBDA * t)

# TODO 3: make a 1x2 subplot with SHARED x and y axes.
#         left panel : scatter of the observed data, titled "Observed data"
#         right panel: line plot of the analytical curve, titled "Analytical"
#         label the axes.
figure, axes = plt.subplots(1, 2, sharex=True, sharey=True)

left_panel = axes[0]
right_panel = axes[1]

left_panel.scatter(t, observed)
left_panel.set_title("Observed data")
left_panel.set_xlabel("time")
left_panel.set_ylabel("count")

right_panel.plot(t, analytical)
right_panel.set_title("Analytical")
right_panel.set_xlabel("time")
right_panel.set_ylabel("count")

# TODO 4: save the figure as figure.png
plt.savefig("figure.png")