import numpy as np
from scipy.optimize import brentq

# 1. Define the mathematical function
def f(x):
    return x**3 - 2*x - 5

# 2. Pick a bracket [a, b] where f(a) and f(b) have opposite signs
# f(1) = 1 - 2 - 5 = -6  (negative)
# f(3) = 27 - 6 - 5 = 16 (positive)
a, b = 1.0, 3.0

# 3. Find the root
root, result = brentq(f, a, b, full_output=True)

# 4. Display results
print(f"Root found at x = {root:.6f}")
print(f"f(root)         = {f(root):.2e}")
print(f"Converged in {result.iterations} iterations.")