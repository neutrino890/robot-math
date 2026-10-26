"""
第 II 部分：4 类矩阵的“谱性质 + 二次型 + 状态轨迹”可视化
================================================================
沿用 matrix_review.py 里构造好的 4 类矩阵：
  ① 对称正定 SPD       A = QΛQᵀ, λ = {1,2,3}
  ② 半正定 PSD（秩亏）  A = BᵀB,   λ = {0, 2.35, 7.65}
  ③ 不稳定（鞍点）      A = diag(1, -2)
  ④ 秩亏（列相关）      A 有两列线性相关, rank = 2

每个矩阵我们要算出并画出来：
  秩 rank / 特征值 λ / 谱半径 ρ / 二次型 xᵀAx / 轨迹 x(t) = e^{At} x0

运行：
  python src/matrix_trajectories.py              # 存图到 experiments/figures/
  python src/matrix_trajectories.py --show       # 存图并弹窗显示
  python src/matrix_trajectories.py --with-ref   # 多画一行稳定对照组 diag(-2,-1)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from scipy.linalg import expm

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

TOL = 1e-9


# ============================================================
# 第 1 步：构造 4 类矩阵（和 matrix_review.py 完全一致）
# ============================================================

def build_matrices() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)  # 与复习脚本同一种子 → 数字对得上

    B = rng.standard_normal((3, 3))
    A1 = B.T @ B                                   # 方法 1 的 SPD（本例接近奇异）

    Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    A2 = Q @ np.diag([3.0, 2.0, 1.0]) @ Q.T        # 方法 2 的 SPD（干净、好画）

    B_psd = np.array([[1.0, 0.0, 1.0],
                      [0.0, 1.0, 1.0],
                      [2.0, -1.0, 1.0]])           # 第三列 = 第一列 + 第二列
    A3 = B_psd.T @ B_psd                           # 半正定，rank = 2

    A_unstable = np.diag([1.0, -2.0])              # 特征值 1 与 -2 → 鞍点

    A_rankdef = np.array([[2.0, 1.0, 0.0],
                          [4.0, 2.0, 1.0],
                          [6.0, 3.0, 0.0]])        # 列 1 与列 2 相关的非对称 3x3

    return {"A1": A1, "A2": A2, "A3": A3,
            "A_unstable": A_unstable, "A_rankdef": A_rankdef}


# ============================================================
# 第 2 步：一次性算出 5 个量 —— 秩 / 特征值 / 谱半径 / 二次型类型 / 稳定性
# ============================================================

@dataclass
class Case:
    """一个待分析的矩阵 + 轨迹仿真设置。"""
    key: str                 # 短名，写进图里
    title: str               # 行标题
    A: np.ndarray
    note: str = ""           # 一句话说明它“特别”在哪
    x0: np.ndarray | None = None
    T: float | None = None   # 仿真时长；None = 自动按增长率选
    info: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        d = self.A.shape[0]
        if self.x0 is None:                     # 默认初值：每个方向都有分量
            self.x0 = np.ones(d) / np.sqrt(d)
        self.x0 = np.asarray(self.x0, dtype=float)


def definiteness(A: np.ndarray) -> str:
    """用特征值符号给（对称）矩阵定性。"""
    if not np.allclose(A, A.T, atol=1e-12):
        return "非对称（看对称部分）"
    w = np.linalg.eigvalsh(A)
    tol = TOL * max(1.0, float(np.abs(w).max()))
    if w.min() > tol:
        return "正定 PD"
    if w.min() >= -tol:
        return "半正定 PSD（含零特征值）"
    if w.max() < -tol:
        return "负定 ND"
    if w.max() <= tol:
        return "半负定 NSD"
    return "不定（鞍形）"


def analyze(case: Case) -> dict:
    """把 5 个量一次算齐，并给出连续/离散两种稳定判据。"""
    A = case.A
    d = A.shape[0]

    ev = np.linalg.eigvals(A)                       # 特征值（可能复数）
    rho = float(np.abs(ev).max())                   # 谱半径 ρ = max|λ|
    rank = int(np.linalg.matrix_rank(A))            # 秩（SVD 数值判据）
    sv = np.linalg.svd(A, compute_uv=False)         # 奇异值，用来解释 rank
    max_re = float(ev.real.max())                   # 连续时间稳定判据
    cond = float(sv[0] / sv[-1]) if sv[-1] > 0 else np.inf

    # 连续时间 ẋ = Ax：Re λ > 0 发散，= 0 临界，< 0 收敛
    tol_c = 1e-9 * max(1.0, rho)
    if max_re > tol_c:
        verdict_c = "发散（不稳定）"
    elif max_re >= -tol_c:
        verdict_c = "有界（临界）"
    else:
        verdict_c = "收敛（稳定）"

    # 离散时间 x_{k+1} = Ax_k：看 ρ 与 1 的关系
    if rho > 1 + tol_c:
        verdict_d = "发散"
    elif rho >= 1 - tol_c:
        verdict_d = "有界（临界）"
    else:
        verdict_d = "收敛"

    # 自动选时长：让增长因子大约落在 e^4 ≈ 55 倍
    T = case.T if case.T is not None else float(np.clip(4.0 / max(max_re, 1e-6), 0.5, 6.0))

    info = {
        "dim": d, "ev": ev, "rho": rho, "rank": rank, "sv": sv,
        "max_re": max_re, "cond": cond, "deftype": definiteness(A),
        "verdict_c": verdict_c, "verdict_d": verdict_d, "T": T,
    }
    case.info = info
    return info


# ============================================================
# 第 3 步：状态轨迹 x(t) = e^{At} x0，以及沿轨迹的二次型 V(t)
# ============================================================

def simulate(A: np.ndarray, x0: np.ndarray, T: float, n: int = 400) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0.0, T, n)
    X = np.array([expm(A * t) @ x0 for t in ts])     # 解析解，无积分误差
    return ts, X


def quad_form(X: np.ndarray, A: np.ndarray) -> np.ndarray:
    """V(x) = xᵀAx，对一组列向量（或一行一条）同时求值。

    注意 xᵀAx 只依赖 A 的对称部分，所以非对称矩阵这里等于用 (A+Aᵀ)/2。
    """
    return np.einsum("ni,ij,nj->n", X, A, X)


def principal_plane(A: np.ndarray, k: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """取对称部分绝对值最大的 k 个特征方向，作为画等值线的 2D 平面。"""
    w, V = np.linalg.eigh(0.5 * (A + A.T))
    order = np.argsort(-np.abs(w))[:k]
    return V[:, order], w[order]


# ============================================================
# 第 4 步：四联图 —— 谱 / 二次型等值线 / 状态轨迹 / V(t)
# ============================================================

def plot_spectrum(ax, case: Case) -> None:
    info = case.info
    ev, rho = info["ev"], info["rho"]
    lim = max(1.25 * max(float(np.abs(ev).max()), rho), 1.2)

    ax.axhspan(0, lim, color="#f7d9d9", zorder=0)       # Re λ > 0 → 连续时间发散区
    ax.axhspan(-lim, 0, color="#dce9f7", zorder=0)      # Re λ < 0 → 收敛区
    ax.axhline(0, color="0.45", lw=0.8)
    ax.axvline(0, color="0.45", lw=0.8)

    th = np.linspace(0, 2 * np.pi, 361)
    ax.plot(rho * np.cos(th), rho * np.sin(th), ":", color="0.2", lw=1.2)  # 谱半径圆

    ax.plot(ev.real, ev.imag, "o", ms=9, mfc="#2166ac", mec="k", zorder=5)
    # 特征值全挤在实轴上，标签按顺序上下错开来避免重叠
    offsets = [(10, 10), (10, -18), (10, 28), (10, -34)]
    for k, lam in enumerate(np.unique(np.round(ev, 10))):
        label = (f"{lam.real:.2f}{lam.imag:+.2f}i" if abs(lam.imag) > 1e-12
                 else f"{lam.real:.2f}")
        ax.annotate(label, (lam.real, lam.imag), textcoords="offset points",
                    xytext=offsets[k % len(offsets)], fontsize=9, color="#2166ac")

    sv_txt = ", ".join(f"{s:.3g}" for s in info["sv"])
    txt = (f"rank = {info['rank']}\nρ(A) = {rho:.2f}\nmax Re λ = {info['max_re']:.2f}\n"
           f"κ = {info['cond']:.3g}\nσ = [{sv_txt}]")
    ax.text(0.02, 0.02, txt, transform=ax.transAxes, va="bottom", ha="left",
            fontsize=8, family="DejaVu Sans Mono",
            bbox=dict(fc="white", ec="0.7", alpha=0.85, boxstyle="round,pad=0.35"))

    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_box_aspect(1)
    ax.set_xlabel("Re λ")
    ax.set_ylabel("Im λ")
    handles = [
        Patch(fc="#f7d9d9", ec="none", label="Re λ > 0：连续时间发散"),
        Patch(fc="#dce9f7", ec="none", label="Re λ < 0：连续时间衰减"),
        Line2D([], [], ls=":", color="0.2", lw=1.2, label=f"ρ = {rho:.2f}（谱半径）"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=7.5, framealpha=0.9)


def plot_quadratic(ax, case: Case, X: np.ndarray) -> None:
    A = case.A
    P, pw = principal_plane(A, k=min(2, case.info["dim"]))
    Abar = P.T @ A @ P
    Y = X @ P                                        # 轨迹投影到 (e1,e2) 平面

    span = max(float(np.abs(Y).max()), 1.0)
    g = np.linspace(-1.7 * span, 1.7 * span, 241)
    Y1, Y2 = np.meshgrid(g, g)
    Q = (Abar[0, 0] * Y1**2 + (Abar[0, 1] + Abar[1, 0]) * Y1 * Y2 + Abar[1, 1] * Y2**2)

    qmin, qmax = float(Q.min()), float(Q.max())
    qscale = max(abs(qmin), abs(qmax), 1e-12)
    qtol = 1e-9 * qscale                             # 相对容差，避免数值"零"误判
    if qmin >= -qtol:                                # PD / PSD：同心椭圆
        levels, cmap = np.geomspace(qmax * 1e-4, qmax, 12), "viridis"
    elif qmax <= qtol:                               # ND / NSD：也是椭圆，只是取负
        levels, cmap = -np.geomspace(abs(qmin), qscale * 1e-4, 12), "viridis"
    else:                                            # 不定：双曲线族 + 零等值线
        levels, cmap = np.linspace(-0.9 * qscale, 0.9 * qscale, 17), "RdBu_r"

    ax.contourf(Y1, Y2, Q, levels=levels, cmap=cmap, alpha=0.85, extend="both")
    ax.contour(Y1, Y2, Q, levels=levels, colors="k", linewidths=0.3, alpha=0.35)
    if qmin < -qtol < qmax:
        ax.contour(Y1, Y2, Q, levels=[0.0], colors="k", linewidths=1.4, linestyles="--")
        ax.text(0.02, 0.96, r"虚线 = $x^\top A x = 0$", transform=ax.transAxes,
                va="top", fontsize=8.5,
                bbox=dict(fc="white", ec="none", alpha=0.8, boxstyle="round,pad=0.25"))

    ax.plot(Y[:, 0], Y[:, 1], color="#d62728", lw=1.9, label="轨迹投影")
    ax.plot(Y[0, 0], Y[0, 1], "o", color="#2ca02c", ms=8, mec="k", label="x(0)")
    ax.plot(Y[-1, 0], Y[-1, 1], "X", color="#d62728", ms=10, mec="k", label=f"x(T)")

    ax.set_xlim(g[0], g[-1])
    ax.set_ylim(g[0], g[-1])
    ax.set_xlabel(f"沿主轴 e1 (λ_sym={pw[0]:.2f})")
    ax.set_ylabel(f"沿主轴 e2 (λ_sym={pw[1]:.2f})")
    ax.legend(loc="lower right", fontsize=8.5, framealpha=0.9)


def plot_states(ax, case: Case, ts: np.ndarray, X: np.ndarray) -> None:
    for i in range(X.shape[1]):
        ax.plot(ts, X[:, i], lw=1.8, label=f"x{i + 1}")
    if np.abs(X).max() > 1e3:
        ax.set_yscale("symlog", linthresh=1.0)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xlabel("t")
    ax.set_ylabel("状态 x(t)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, ncols=2)


def plot_energy(ax, case: Case, ts: np.ndarray, X: np.ndarray) -> None:
    V = quad_form(X, case.A)
    # 对对称 A：V̇ = xᵀ(AᵀA + A²)x = 2‖Ax‖² ≥ 0 → SPD/PSD 的 V(t) 只增不减
    v0 = float(quad_form(case.x0[None, :], case.A)[0])
    vmax = max(float(np.abs(V).max()), abs(v0), 1e-9)

    ax.plot(ts, V, lw=2.0, color="#6a3d9a", label=r"$V(t) = x^\top A x$")
    ax.plot(ts, v0 * np.exp(-2 * ts), "--", color="0.5", lw=1.3,
            label=r"参考 $e^{-2t}$（稳定）")
    ax.axhline(0, color="0.45", lw=0.8)
    # 不定矩阵的 V(t) 会穿过 0，用 symlog 才能同时看到 ±和数量级
    ax.set_yscale("symlog", linthresh=max(vmax * 1e-3, 1e-12))
    ax.set_ylim(-3 * vmax, 3 * vmax)
    ax.set_xlabel("t")
    ax.set_ylabel(r"二次型 $V(t)$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="best")
    ax.text(0.02, 0.03, f"V(0)={v0:.3g}   V(T)={V[-1]:.3g}",
            transform=ax.transAxes, fontsize=8,
            bbox=dict(fc="white", ec="0.8", alpha=0.85, boxstyle="round,pad=0.3"))


def make_cases() -> list[Case]:
    M = build_matrices()
    return [
        Case("SPD", "① 对称正定 SPD", M["A2"],
             note="A = QΛQᵀ，λ = {1,2,3} 全正 → 所有方向都发散"),
        Case("PSD", "② 半正定 PSD（秩亏）", M["A3"],
             note="A = BᵀB，λ = {0, 2.35, 7.65}：零方向静止，其余发散；V≥0 且 V̇=2‖Ax‖²≥0"),
        Case("UNSTABLE", "③ 不稳定（鞍点）", M["A_unstable"],
             note="diag(1,-2)：一个方向发散、一个方向收敛；V 不定、会变号"),
        Case("RANKDEF", "④ 秩亏（列相关）", M["A_rankdef"],
             note="rank 2 < 3：λ={4.65, 0, -0.65} → 收敛/静止/发散 三种方向同时存在"),
    ]


def draw_row(axes, case: Case) -> None:
    info = analyze(case)
    ts, X = simulate(case.A, case.x0, info["T"])

    plot_spectrum(axes[0], case)
    plot_quadratic(axes[1], case, X)
    plot_states(axes[2], case, ts, X)
    plot_energy(axes[3], case, ts, X)

    axes[0].set_title(f"{case.title}\n特征值谱 / 谱半径（复平面 Re–Im）",
                      fontsize=11.5, fontweight="bold", loc="left")
    axes[1].set_title(r"二次型 $x^\top A x$ 等值线（对称部分主方向投影）", fontsize=10.5)
    axes[2].set_title(f"状态轨迹 $x(t)$，T = {info['T']:.2f}", fontsize=10.5)
    axes[3].set_title(r"二次型沿轨迹 $V(t)$", fontsize=10.5)

    axes[0].text(0.02, 0.99, f"{info['deftype']}\n连续: {info['verdict_c']}\n离散: {info['verdict_d']}",
                 transform=axes[0].transAxes, va="top", ha="left", fontsize=8.5,
                 bbox=dict(fc="#fffbe6", ec="#c9a227", alpha=0.95, boxstyle="round,pad=0.35"))


def print_summary(cases: list[Case]) -> None:
    head = f"{'类别':<6} {'rank':>4} {'ρ(A)':>7} {'max Re λ':>9} {'二次型类型':<22} {'连续':<10} {'离散':<10}"
    print("\n" + "=" * 100)
    print(head)
    print("-" * 100)
    for c in cases:
        i = c.info
        ev_txt = ", ".join(
            (f"{l.real:.3g}{l.imag:+.3g}j" if abs(l.imag) > 1e-9 else f"{l.real:.3g}")
            for l in i["ev"])
        print(f"{c.key:<6} {i['rank']:>4} {i['rho']:>7.3f} {i['max_re']:>9.3f} "
              f"{i['deftype']:<22} {i['verdict_c']:<10} {i['verdict_d']:<10}")
        print(f"       λ = [{ev_txt}]   σ = {np.array2string(i['sv'], precision=3)}")
        print(f"       {c.note}")
    print("=" * 100)


def main() -> None:
    ap = argparse.ArgumentParser(description="4 类矩阵的谱 / 二次型 / 轨迹可视化")
    ap.add_argument("--out", default="experiments/figures/matrix_trajectories.png",
                    help="输出图片路径（notes/week1.md 里引用的就是这个文件）")
    ap.add_argument("--dpi", type=int, default=130)
    ap.add_argument("--show", action="store_true", help="保存后弹窗显示")
    ap.add_argument("--with-ref", action="store_true",
                    help="额外画一行稳定对照组 diag(-2,-1)")
    args = ap.parse_args()

    ref = Case("REF", "⑤ 参考：对称负定（稳定）", np.diag([-2.0, -1.0]),
               note="diag(-2,-1)：全负实部 → 收敛；离散判据 ρ=2>1 却发散")
    analyze(ref)

    cases = make_cases()
    plotted = cases + [ref] if args.with_ref else cases

    fig, axs = plt.subplots(len(plotted), 4, figsize=(22, 3.95 * len(plotted)),
                            layout="constrained")
    for row, case in zip(axs, plotted):
        draw_row(row, case)

    fig.suptitle(r"$\dot{x} = A x$ 的谱性质与状态轨迹：SPD / PSD / 不稳定 / 秩亏", fontsize=16)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")

    print_summary(cases + ([] if args.with_ref else [ref]))
    print(f"\n图已保存: {out.resolve()}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()
