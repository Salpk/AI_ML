"""
================================================================================
 FIREFLY ALGORITHM - Pedagogical Implementation (Python)
================================================================================
 Companion to: "Introducing the Firefly Algorithm (FA) to K-12 and
                Postgraduate STEM Education Using Block-Based and Textual
                Programming Environments"

 Provides TWO aligned implementations:

   1) FireflyAlgorithmCanonical  - Yang (2009), O(n^2) pairwise
   2) FireflyAlgorithmSimplified - single-attractor, mirrors the Scratch
                                   project block-for-block; O(n)

 ================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt


def sphere(x):
    return np.sum(x ** 2)

def rastrigin(x):
    A = 10
    n = len(x)
    return A * n + np.sum(x ** 2 - A * np.cos(2 * np.pi * x))

def ackley(x):
    n = len(x)
    s1 = np.sum(x ** 2); s2 = np.sum(np.cos(2 * np.pi * x))
    return (-20 * np.exp(-0.2 * np.sqrt(s1 / n))
            - np.exp(s2 / n) + 20 + np.e)


class FireflyAlgorithmCanonical:
    """Pairwise FA (Yang, 2009). O(n^2) per iteration."""

    def __init__(self, objective_func, n_fireflies=25, dim=2,
                 bounds=(-5.0, 5.0), max_iter=100,
                 alpha=0.5, beta0=1.0, gamma=1.0, alpha_damp=0.97, seed=None):
        self.f = objective_func; self.n = n_fireflies; self.dim = dim
        self.lb, self.ub = bounds; self.max_iter = max_iter
        self.alpha = alpha; self.beta0 = beta0; self.gamma = gamma
        self.alpha_damp = alpha_damp
        self.rng = np.random.default_rng(seed)
        self.positions = self.rng.uniform(self.lb, self.ub, (self.n, self.dim))
        self.brightness = np.array([self.f(p) for p in self.positions])
        bi = np.argmin(self.brightness)
        self.best_position = self.positions[bi].copy()
        self.best_value = self.brightness[bi]
        self.history_best = []

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.n):
                for j in range(self.n):
                    if self.brightness[j] < self.brightness[i]:
                        r = np.linalg.norm(self.positions[i] - self.positions[j])
                        beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                        eps = self.alpha * (self.rng.random(self.dim) - 0.5) \
                              * (self.ub - self.lb)
                        self.positions[i] += (
                            beta * (self.positions[j] - self.positions[i]) + eps)
                        self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                        self.brightness[i] = self.f(self.positions[i])
            bi = np.argmin(self.brightness)
            if self.brightness[bi] < self.best_value:
                self.best_value = self.brightness[bi]
                self.best_position = self.positions[bi].copy()
            self.history_best.append(self.best_value)
            self.alpha *= self.alpha_damp
        return self.best_position, self.best_value


class FireflyAlgorithmSimplified:
    """Single-attractor FA - mirrors the Scratch project."""

    def __init__(self, objective_func, n_fireflies=20, dim=2,
                 bounds=(-5.0, 5.0), max_iter=100,
                 alpha=0.5, beta0=0.3, gamma=0.05, alpha_damp=0.97,
                 vision=None, seed=None):
        self.f = objective_func; self.n = n_fireflies; self.dim = dim
        self.lb, self.ub = bounds; self.max_iter = max_iter
        self.alpha = alpha; self.beta0 = beta0; self.gamma = gamma
        self.alpha_damp = alpha_damp
        self.vision = vision if vision is not None else 0.8 * (self.ub - self.lb)
        self.rng = np.random.default_rng(seed)
        self.positions = self.rng.uniform(self.lb, self.ub, (self.n, self.dim))
        self.brightness = np.array([self.f(p) for p in self.positions])
        bi = np.argmin(self.brightness)
        self.light = self.positions[bi].copy()
        self.best_value = self.brightness[bi]
        self.best_position = self.light.copy()
        self.history_best = []

    def optimize(self):
        for t in range(self.max_iter):
            for i in range(self.n):
                disp = self.light - self.positions[i]
                r = np.linalg.norm(disp)
                eps = self.alpha * (self.rng.random(self.dim) - 0.5) \
                      * (self.ub - self.lb)
                if r < self.vision:
                    beta = self.beta0 * np.exp(-self.gamma * r ** 2)
                    self.positions[i] += beta * disp + eps
                else:
                    self.positions[i] += eps
                self.positions[i] = np.clip(self.positions[i], self.lb, self.ub)
                self.brightness[i] = self.f(self.positions[i])
            bi = np.argmin(self.brightness)
            if self.brightness[bi] < self.best_value:
                self.best_value = self.brightness[bi]
                self.light = self.positions[bi].copy()
                self.best_position = self.light.copy()
            self.history_best.append(self.best_value)
            self.alpha *= self.alpha_damp
        return self.best_position, self.best_value


if __name__ == "__main__":
    obj = rastrigin
    fa_c = FireflyAlgorithmCanonical(obj, n_fireflies=30, dim=2,
        bounds=(-5.12, 5.12), max_iter=80, seed=42)
    fa_s = FireflyAlgorithmSimplified(obj, n_fireflies=30, dim=2,
        bounds=(-5.12, 5.12), max_iter=80, gamma=0.05, vision=8.0, seed=42)
    pos_c, val_c = fa_c.optimize()
    pos_s, val_s = fa_s.optimize()
    print(f"Canonical  : f* = {val_c:.4e}  at {pos_c}")
    print(f"Simplified : f* = {val_s:.4e}  at {pos_s}")
    print(f"True opt   : f = 0  at [0, 0]")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(fa_c.history_best, label="Canonical", linewidth=2, color="#1f77b4")
    axes[0].plot(fa_s.history_best, label="Simplified", linewidth=2,
                 color="#d62728", linestyle="--")
    axes[0].set_yscale("symlog", linthresh=1e-6); axes[0].grid(alpha=0.3)
    axes[0].set_xlabel("Iteration"); axes[0].set_ylabel("Best f")
    axes[0].set_title("Convergence comparison"); axes[0].legend()

    g = np.linspace(-5.12, 5.12, 150); X, Y = np.meshgrid(g, g)
    Z = np.array([[obj(np.array([x, y])) for x, y in zip(rx, ry)]
                  for rx, ry in zip(X, Y)])
    axes[1].contourf(X, Y, Z, levels=25, cmap="viridis", alpha=0.7)
    axes[1].scatter(fa_c.positions[:, 0], fa_c.positions[:, 1], s=30,
                    c="cyan", edgecolor="black", label="Canonical")
    axes[1].scatter(fa_s.positions[:, 0], fa_s.positions[:, 1], s=30,
                    c="orange", edgecolor="black", marker="^", label="Simplified")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y"); axes[1].legend()
    axes[1].set_title("Final swarm positions")
    plt.tight_layout()
    plt.savefig("fa_comparison.png", dpi=300, bbox_inches="tight")
    print("Saved fa_comparison.png")
