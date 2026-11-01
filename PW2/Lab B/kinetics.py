"""
PW2 Lab B Part 3 -- fit a reaction's rate constant to measured data.

A first-order reaction decays as  C(t) = C0 * exp(-k*t).  You have noisy
concentration-vs-time measurements; find the k that best matches them.
Complete the TODOs. Run:  python kinetics.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# TODO 1: read kinetics.csv (columns time, concentration) into arrays t, C.
#         Set C0 = the first concentration.
data = np.loadtxt("kinetics.csv", delimiter=",", skiprows=1)
t = data[:, 0]
C = data[:, 1]
C0 = C[0]

# TODO 2: write total_error(k) = sum of (measured - C0*exp(-k*t))^2.
#         This is the "how bad" number: small when the model matches the data.
def total_error(k):
    model = C0 * np.exp(-k[0] * t)
    difference = C - model
    return np.sum(difference ** 2)

# TODO 3: minimise total_error with scipy.optimize.minimize (method "SLSQP",
#         bounds [(0, 5)], start x0=0.5). Print the fitted k.
result = minimize(total_error, x0=[0.5], method="SLSQP", bounds=[(0, 5)])
k_fit = result.x[0]
print("Fitted k:", k_fit)

# TODO 4: plot the measured data (points) and your fitted curve (line) together.
#         Save as kinetics.png.
fitted_curve = C0 * np.exp(-k_fit * t)

plt.scatter(t, C, label="Measured data")
plt.plot(t, fitted_curve, color="red", label="Fitted curve")
plt.xlabel("time")
plt.ylabel("concentration")
plt.title("First-order reaction fit")
plt.legend()
plt.savefig("kinetics.png")