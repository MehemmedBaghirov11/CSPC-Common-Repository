import numpy as np
import matplotlib.pyplot as plt

# Read the trajectory data
data = np.loadtxt("trajectory.csv", delimiter=",", skiprows=1)
t = data[:, 0]
x = data[:, 1]
y = data[:, 2]

# Compute velocity in x and y directions separately
vx = np.gradient(x, t)
vy = np.gradient(y, t)

# Compute speed using the formula sqrt(vx^2 + vy^2)
speed = np.sqrt(vx**2 + vy**2)

# Make a figure with two panels: the path, and the speed over time
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

path_panel = axes[0]
speed_panel = axes[1]

path_panel.plot(x, y)
path_panel.set_title("Path (x vs y)")
path_panel.set_xlabel("x")
path_panel.set_ylabel("y")

speed_panel.plot(t, speed)
speed_panel.set_title("Speed over time")
speed_panel.set_xlabel("time (s)")
speed_panel.set_ylabel("speed")

plt.tight_layout()
plt.savefig("trajectory.png")