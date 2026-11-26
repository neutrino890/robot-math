from __future__ import annotations
import torch
import numpy as np
import csv
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

def f_torch(x):
    return torch.sin(x[0]) + x[0] * x[1] ** 2 + torch.exp(x[2])

x = torch.tensor([0.5, -1.2, 0.3], dtype=torch.float64, requires_grad=True)
y = f_torch(x)
y.backward()      
grad = x.grad

def f_n(x):
    return np.sin(x[0]) + x[0] * x[1] ** 2 + np.exp(x[2])

def grad_analytic(x):
    return np.array([np.cos(x[0]) + x[1] ** 2, 
                     2 * x[0] * x[1], 
                     np.exp(x[2])])

def grad_forward(x, h):
    g = np.zeros_like(x)
    fx = f_n(x)
    for i in range(len(x)):
        xp = x.copy()
        xp[i] += h
        g[i] = (f_n(xp) - fx) / h

    return g

def grad_central(x, h):
    g = np.zeros_like(x)
    for i in range(len(x)):
        xp = x.copy()
        xm = x.copy()
        xp[i] += h
        xm[i] -= h
        g[i] = (f_n(xp) - f_n(xm)) / (2 * h)

    return g

x0 = np.array([0.5, -1.2, 0.3])
h = 1e-6

g_true  = grad_analytic(x0)


g_fwd   = grad_forward(x0, h)
g_cen   = grad_central(x0, h)
g_torch = grad.detach().numpy()   

def rel_err(g_hat, g):
    return np.linalg.norm(g_hat - g) / np.linalg.norm(g)



print("analytic :", g_true)
print("forward  :", g_fwd)
print("central  :", g_cen)
print("autograd :", g_torch)
print()
print("forward  相对误差:", rel_err(g_fwd, g_true))   
print("central  相对误差:", rel_err(g_cen, g_true))   
print("autograd 相对误差:", rel_err(g_torch, g_true)) 

hs = 10.0 ** np.arange(-1, -16, -1)

fwd_errs = [rel_err(grad_forward(x0, h), g_true) for h in hs]
cen_errs = [rel_err(grad_central(x0, h), g_true) for h in hs]

for h, fe, ce in zip(hs, fwd_errs, cen_errs):
    print(f"h={h:.0e} forward={fe:.3e} central={ce:.3e}")

matplotlib.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False

out = Path("experiments/results/gradient_error_vs_h.csv")
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["h", "forward_err", "central_err"])
    w.writerows(zip(hs, fwd_errs, cen_errs))


fig, ax = plt.subplots(figsize=(8, 5.5))
ax.loglog(hs, fwd_errs, "-o", color="#1f77b4", label="forward O(h)")
ax.loglog(hs, cen_errs, "-s", color="#ff7f0e", label="central O(h²)")
ax.axhline(np.finfo(float).eps, color="0.4", ls="--", lw=1, label="machine ε")
ax.set_xlabel("step size h")
ax.set_ylabel("relative error")
ax.set_title("Finite-difference error vs step size", loc="left", fontsize=11)
ax.grid(True, which="both", alpha=0.3)
ax.legend(frameon=False, fontsize=9)

fig.savefig("experiments/figures/gradient_error_vs_h.png", dpi=180, facecolor="white")
print("已保存:", out.resolve())