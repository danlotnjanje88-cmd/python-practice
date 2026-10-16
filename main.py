import numpy as np
import matplotlib.pyplot as plt
from math import exp

def panjer_poisson(lam, f, n_max):
    g = np.zeros(n_max + 1)
    g[0] = exp(-lam * (1.0 - f[0]))
    for k in range(1, n_max + 1):
        s = 0.0
        for j in range(1, k + 1):
            if j < len(f):
                s += j * f[j] * g[k - j]
        g[k] = (lam / k) * s
    return g

n_max = 25
lam = 3.0

f_det = np.zeros(n_max + 1)
f_det[1] = 1.0

g_poisson = panjer_poisson(lam, f_det, n_max)

print(f"{'k':>3} | {'Panjer':>16}")
print("-" * 22)
for k in range(0, 16):
    print(f"{k:>3} | {g_poisson[k]:>16.10f}")

print("\nSum of Panjer pmf:", g_poisson.sum())

ks = np.arange(16)
plt.figure(figsize=(10, 6))
plt.plot(ks, g_poisson[:16], marker="o", linestyle="-", color="steelblue",
         linewidth=2, markersize=7, markerfacecolor="orange",
         markeredgecolor="black", label="Panjer recursion")
plt.title("Panjer Recursion - Poisson(lambda = 3)")
plt.xlabel("k")
plt.ylabel("P(S = k)")
plt.xticks(np.arange(0, 16, 1))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("panjer_poisson_line.png", dpi=150)
plt.show()