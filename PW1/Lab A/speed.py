import time
from decay import simulate_loop, simulate

N0 = 200_000
lam = 0.4

start = time.perf_counter()
simulate_loop(N0, lam)
loop_time = time.perf_counter() - start

start = time.perf_counter()
simulate(N0, lam)
numpy_time = time.perf_counter() - start

print(f"loop  : {loop_time:.3f} s")
print(f"numpy : {numpy_time:.3f} s")
print(f"speed-up: {loop_time / numpy_time:.1f}x faster")