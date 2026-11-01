"""
PW2 Lab A -- Motion from tracking data.

Read noisy free-fall position measurements, then:
  - differentiate once  -> velocity
  - differentiate twice -> acceleration (should be ~ constant -g, but noisy!)
  - integrate the acceleration back up -> recover velocity and position

Complete the TODOs. Run:  python analysis.py
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid

# TODO 1: read freefall.csv into arrays t and y
#         (hint: np.loadtxt with a comma delimiter, skipping the header)
data = np.loadtxt("freefall.csv", delimiter=",", skiprows=1)
t = data[:, 0]
y = data[:, 1]

# TODO 2: compute velocity v = derivative of y w.r.t. t    (np.gradient)
#         and acceleration a = derivative of v w.r.t. t    (np.gradient again)
#         Print the mean acceleration. Is it close to -9.81? Is it noisy?
v = np.gradient(y, t)
a = np.gradient(v, t)

mean_a = a.mean()
print("Mean acceleration:", mean_a)

std_a = a.std()
print("Standard deviation of acceleration:", std_a)

# TODO 3: integrate a back up to recover velocity and position
#         (hint: cumulative_trapezoid(a, t, initial=0) + v[0], then again)
v_back = cumulative_trapezoid(a, t, initial=0)
v_back = v_back + v[0]

y_back = cumulative_trapezoid(v_back, t, initial=0)
y_back = y_back + y[0]

difference = y_back - y
max_diff = np.max(np.abs(difference))
print("Max difference:", max_diff)

# TODO 4: make a figure with 3 stacked panels: position, velocity, acceleration
#         vs time. Mark the true -9.81 line on the acceleration panel.
#         Save it as motion.png
fig, axes = plt.subplots(3, 1, sharex=True)

position_panel = axes[0]
velocity_panel = axes[1]
acceleration_panel = axes[2]

position_panel.plot(t, y)
position_panel.set_title("Position")
position_panel.set_ylabel("Position (m)")

velocity_panel.plot(t, v)
velocity_panel.set_title("Velocity")
velocity_panel.set_ylabel("Velocity (m/s)")

acceleration_panel.plot(t, a)
acceleration_panel.axhline(-9.81, color="red", linestyle="--")
acceleration_panel.set_title("Acceleration")
acceleration_panel.set_ylabel("Acceleration (m/s^2)")
acceleration_panel.set_xlabel("Time (s)")

plt.tight_layout()
plt.savefig("motion.png")