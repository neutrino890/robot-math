from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="病态多项式最小二乘对比实验")
    parser.add_argument("--show", action="store_true", help="同时弹出图片窗口")
    return parser.parse_args()

def build_vandermonde(x: np.ndarray, degree: int) -> np.ndarray:
    columns = []

    for power in range(degree + 1):
        columns.append(x**power)

    return np.column_stack(columns)

def condition_number(A: np.ndarray) -> float:
    singular_values = np.linalg.svd(A, compute_uv=False)

    return float(singular_values[0] / singular_values[-1])

def solve_normal(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    G = A.T @ A
    rhs = A.T @ b
    return np.linalg.solve(G, rhs)

def solve_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    Q, R = np.linalg.qr(A, mode="reduced")
    rhs = Q.T @ b
    return np.linalg.solve(R, rhs)

def solve_svd(
    A: np.ndarray,
    b: np.ndarray,
    rcond: float = 1e-12,
) -> np.ndarray:
    U, singular_values, Vt = np.linalg.svd(A, full_matrices=False)

    cutoff = rcond * singular_values[0]
    singular_values_inverse = np.zeros_like(singular_values)
    keep = singular_values > cutoff
    singular_values_inverse[keep] = 1.0 / singular_values[keep]

    return Vt.T @ (singular_values_inverse * (U.T @ b))

def solve_ridge(A: np.ndarray, b: np.ndarray, alpha: float) -> np.ndarray:
    singular_values = np.linalg.svd(A, compute_uv=False)
    sigma_max = singular_values[0]

    lam = alpha * sigma_max**2

    G = A.T @ A + lam * np.eye(A.shape[1])
    rhs = A.T @ b
    return np.linalg.solve(G, rhs) 

def plot_summary(rows: list[dict], show: bool) -> None:
    """按 baseline_fn_steady.png 的风格画 2×2 四联图，存 PNG 到 experiments/figures/。

    风格要点（对齐 dex-hand 基线图）：宽幅、右侧外部图例（无边框）、tab20 配色、
    仅水平网格线（低透明度）、suptitle + 副标题、dpi=180 白底、英文标签。
    """
    solvers = ["normal", "qr", "svd", "ridge"]
    labels = {
        "normal": "Normal equations",
        "qr": "QR",
        "svd": "SVD",
        "ridge": "Ridge (α=1e-4)",
    }
    palette = plt.get_cmap("tab20")
    colors = {s: palette(i) for i, s in enumerate(solvers)}

    degrees = np.array(sorted({r["degree"] for r in rows}), dtype=int)

    def series(solver: str, key: str) -> np.ndarray:
        """按 degree 对齐，抽出某个 solver 的某个指标成一维数组。"""
        vals = {r["degree"]: r[key] for r in rows if r["solver"] == solver}
        return np.array([vals[d] for d in degrees], dtype=float)

    # cond(V) 与 solver 无关，每个 degree 取一次即可
    cond_map = {r["degree"]: r["cond_v"] for r in rows}
    conds = np.array([cond_map[d] for d in degrees], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.6))
    fig.subplots_adjust(left=0.085, right=0.74, bottom=0.09, top=0.86,
                        hspace=0.34, wspace=0.30)
    fig.suptitle("Ill-conditioned polynomial least squares: four solvers vs degree",
                 fontsize=14, y=0.97)
    fig.text(0.085, 0.92,
             "100 noise trials per degree; a_true fixed; errors normalized (log scale)",
             fontsize=10)

    
    ax = axes[0, 0]
    ax.semilogy(degrees, conds, "-o", color="#333333", lw=1.8, ms=4)
    ax.set_title("Condition number cond(V)", loc="left", fontsize=11)
    ax.set_ylabel("cond(V)")
    ax.grid(axis="y", alpha=0.25)

    
    ax = axes[0, 1]
    for s in solvers:
        ax.semilogy(degrees, series(s, "param_err_mean"), "-o", color=colors[s], lw=1.4, ms=4)
    ax.set_title("Parameter error ||a_hat − a_true|| / ||a_true|| (mean)",
                 loc="left", fontsize=11)
    ax.set_ylabel("Parameter error")
    ax.grid(axis="y", alpha=0.25)

    
    ax = axes[1, 0]
    for s in solvers:
        ax.semilogy(degrees, series(s, "param_err_std"), "-o", color=colors[s], lw=1.4, ms=4)
    ax.set_title("Noise sensitivity: std of parameter error (100 trials)",
                 loc="left", fontsize=11)
    ax.set_ylabel("Std of parameter error")
    ax.grid(axis="y", alpha=0.25)

    
    ax = axes[1, 1]
    for s in solvers:
        ax.semilogy(degrees, series(s, "test_err_mean"), "-o", color=colors[s], lw=1.4, ms=4)
    ax.set_title("Noise-free test error (mean)", loc="left", fontsize=11)
    ax.set_ylabel("Test error")
    ax.grid(axis="y", alpha=0.25)

    for ax in axes.flat:
        ax.set_xlabel("Polynomial degree")

    
    handles = [Line2D([0], [0], color=colors[s], lw=1.4, marker="o", label=labels[s])
               for s in solvers]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.76, 0.93),
               frameon=False, fontsize=9)

    path = Path("experiments/figures/ill_conditioning.png")
    fig.savefig(path, dpi=180, facecolor="white")
    print(f"已保存: {path.resolve()}")

    if show:
        plt.show()


def plot_time(rows: list[dict], show: bool) -> None:
    
    solvers = ["normal", "qr", "svd", "ridge"]
    labels = {
        "normal": "Normal equations",
        "qr": "QR",
        "svd": "SVD",
        "ridge": "Ridge (α=1e-4)",
    }
    palette = plt.get_cmap("tab20")
    colors = {s: palette(i) for i, s in enumerate(solvers)}

    degrees = np.array(sorted({r["degree"] for r in rows}), dtype=int)

    def series(solver: str, key: str) -> np.ndarray:
        vals = {r["degree"]: r[key] for r in rows if r["solver"] == solver}
        return np.array([vals[d] for d in degrees], dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    fig.subplots_adjust(left=0.10, right=0.72, bottom=0.15, top=0.86)
    fig.suptitle("Per-solve time vs degree", fontsize=14, y=0.97)
    fig.text(0.10, 0.92, "Mean over 100 trials; solver call only (perf_counter_ns)", fontsize=10)

    for s in solvers:
        ax.semilogy(degrees, series(s, "time_us"), "-o", color=colors[s], lw=1.4, ms=4)
    ax.set_title("Per-solve time (µs)", loc="left", fontsize=11)
    ax.set_xlabel("Polynomial degree")
    ax.set_ylabel("time [µs]")
    ax.grid(axis="y", alpha=0.25)

    handles = [Line2D([0], [0], color=colors[s], lw=1.4, marker="o", label=labels[s])
               for s in solvers]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.74, 0.9),
               frameon=False, fontsize=9)

    path = Path("experiments/figures/ill_conditioning_time.png")
    fig.savefig(path, dpi=180, facecolor="white")
    print(f"已保存: {path.resolve()}")

    if show:
        plt.show()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(42)
    N_TRIALS = 100
    rows = []
    solvers = {
        "normal": lambda A, y: solve_normal(A, y),
        "qr":     lambda A, y: solve_qr(A, y),
        "svd":    lambda A, y: solve_svd(A, y),
        "ridge":  lambda A, y: solve_ridge(A, y, alpha=1e-4),
    }

    for degree in range(1, 15):
        x = np.linspace(0.0, 1.0, 60)
        V = build_vandermonde(x, degree)
        cond_v = condition_number(V)
        print(f"\ndegree={degree}, cond(V)={cond_v:.3e}")

        a_true = rng.standard_normal(degree + 1)
        y_true = V @ a_true

        sigma = 0.01 * float(np.std(y_true))
        
        x_test = np.linspace(0.0, 1.0, 200)
        V_test = build_vandermonde(x_test, degree)
        y_test_true = V_test @ a_true
        test_denom = np.linalg.norm(y_test_true)

        results = {name: [] for name in solvers}
        times = {name: 0 for name in solvers}
        
        for _ in range(N_TRIALS):
            b = y_true + rng.standard_normal(V.shape[0]) * sigma
            for name, solver in solvers.items():
                t0 = time.perf_counter_ns()
                try:
                    a_hat = solver(V, b)
                except np.linalg.LinAlgError:
                    results[name].append((np.nan, np.nan, np.nan))
                    continue
                times[name] += time.perf_counter_ns() - t0
                param_err = np.linalg.norm(a_hat - a_true) / np.linalg.norm(a_true)
                fit_err   = np.linalg.norm(V @ a_hat - b) / np.linalg.norm(b)
                test_err  = np.linalg.norm(V_test @ a_hat - y_test_true) / test_denom
                results[name].append((param_err, fit_err, test_err))

        

        for name in solvers:
            arr = np.array(results[name])
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=1)
            time_us = times[name] / N_TRIALS / 1e3  # ns → µs，平均每次求解耗时
            print(
                f"{name:>6}: param_err={mean[0]:.3e} ± {std[0]:.3e}  "
                f"fit_err={mean[1]:.3e} ± {std[1]:.3e}  "
                f"test_err={mean[2]:.3e} ± {std[2]:.3e}  "
                f"time={time_us:.2f} µs"
            )
            rows.append({
                "degree": degree, "cond_v": cond_v, "solver": name,
                "param_err_mean": mean[0], "param_err_std": std[0],
                "fit_err_mean": mean[1], "fit_err_std": std[1],
                "test_err_mean": mean[2], "test_err_std": std[2],
                "time_us": time_us,
            })

    out = Path("experiments/results/ill_conditioning_summary.csv")
    fieldnames = [
        "degree", "cond_v", "solver",
        "param_err_mean", "param_err_std",
        "fit_err_mean", "fit_err_std",
        "test_err_mean", "test_err_std",
        "time_us",
    ]
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    plot_summary(rows, args.show)
    plot_time(rows, args.show)





if __name__ == "__main__":
    main()