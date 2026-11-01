"""
PW2 Lab B Part 2 -- three routes to a minimum.

Compare gradient descent, Newton, and SLSQP on two functions:
  2A: f(x) = (x-3)**2 + 1          (easy, one minimum at x=3)
  2B: g(x) = x**4 - 3*x**2 + x + 5 (harder, several stationary points)
Run:  python warmup.py
"""
import numpy as np
from scipy.optimize import newton, minimize


# gradient descent by hand: used for both 2A and 2B
def gradient_descent(slope_function, x_start):
    learning_rate = 0.01
    x = x_start
    for i in range(100000):
        step = learning_rate * slope_function(x)
        x = x - step
        if abs(step) < 1e-10:
            break
    return x


# ---------- 2A: easy convex function ----------
def f(x):   return (x-3)**2 + 1
def df(x):  return 2*(x-3)
def d2f(x): return 2.0

# TODO 2A: minimise f three ways from x0=0 and print each result:
#   (1) gradient descent by hand (loop x = x - lr*df(x) until the step is tiny)
#   (2) scipy.optimize.newton(df, x0, fprime=d2f)
#   (3) scipy.optimize.minimize(f, x0, method="SLSQP")
print("=== 2A: f(x) = (x-3)^2 + 1, start x0 = 0 ===")
x_gd = gradient_descent(df, 0)
x_newton = newton(df, 0, fprime=d2f)
x_slsqp = minimize(lambda x: f(x[0]), x0=[0], method="SLSQP").x[0]
print("Gradient descent:", x_gd)
print("Newton:          ", x_newton)
print("SLSQP:           ", x_slsqp)

# ---------- 2B: harder landscape ----------
def g(x):   return x**4 - 3*x**2 + x + 5
def dg(x):  return 4*x**3 - 6*x + 1
def d2g(x): return 12*x**2 - 6

# TODO 2B: run the same three methods on g, from x0=0 AND from x0=2.
#   For Newton (which solves dg(x)=0), also check the sign of d2g at the answer:
#   d2g > 0 means a minimum, d2g < 0 means a maximum.
#   In your README note: do the methods agree? did Newton find a minimum or
#   another stationary point? how did the starting point change the result?
for start in [0, 2]:
    print()
    print("=== 2B: g(x) = x^4 - 3x^2 + x + 5, start x0 =", start, "===")
    x_gd = gradient_descent(dg, start)
    x_newton = newton(dg, start, fprime=d2g)
    x_slsqp = minimize(lambda x: g(x[0]), x0=[start], method="SLSQP").x[0]
    print("Gradient descent:", x_gd)
    print("Newton:          ", x_newton)
    print("SLSQP:           ", x_slsqp)

    curvature = d2g(x_newton)
    print("d2g at Newton's answer:", curvature)
    if curvature > 0:
        print("Newton landed on a MINIMUM")
    else:
        print("Newton landed on a MAXIMUM")