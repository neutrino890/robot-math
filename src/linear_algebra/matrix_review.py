"""
矩阵性质复习：对称正定、半正定、不稳定状态矩阵、秩亏矩阵
========================================================
 一步步填，每填一步就跑一遍验证。
"""

import numpy as np

# 让随机结果可复现（可选，方便我们对照）
rng = np.random.default_rng(42)

# ============================================================
# 第一步：对称正定矩阵 (SPD)
# ============================================================

# ---------- 方法 1：A = B^T B ----------
#  1: 生成一个 3x3 矩阵 B，要求 B 的列线性无关（列满秩）。
#         可以用 rng.standard_normal((3, 3))，也可以手写一个简单的矩阵。

B = rng.standard_normal((3, 3))

#  2: 用 B 构造 A = B^T @ B

A1 = B.T @ B

#  3: 验证对称性，运行后应该打印 True
print("A1 是否对称", np.allclose(A1, A1.T))

#  4: 验证正定性 —— 判据 a：所有特征值 > 0
#         提示：np.linalg.eigvalsh(A1) 只适用于对称矩阵，返回升序特征值
eig1 = np.linalg.eigvalsh(A1)
print("A1 特征值:", eig1)
if eig1[0] > 0:
    print("A1正定")
else:
    print("A1不正定")

#  5: 验证正定性 —— 判据 b：Cholesky 分解能成功
#         提示：np.linalg.cholesky(A1)，若矩阵不正定会抛异常
L = np.linalg.cholesky(A1)
print("A1 的 Cholesky 下三角因子 L 的 (0,0) 元素:", L[0, 0])


# ---------- 方法 2：A = Q Λ Qᵀ ----------
#  6: 生成正交矩阵 Q。提示：Q, _ = np.linalg.qr(rng.standard_normal((3, 3)))
Q, _  = np.linalg.qr(rng.standard_normal((3, 3))) 

#  7: 构造对角矩阵 Lambda = diag(3, 2, 1)。提示：np.diag([...])
Lam = np.diag([3, 2, 1])

#  8: A2 = Q Λ Qᵀ
A2 = Q @ Lam @ Q.T

#  9: 验证——对称、特征值恰好 [1,2,3]、Cholesky 成功
print("A2 是否对称:", np.allclose(A2, A2.T))      # ← np.allclose(A2, A2.T)
print("A2 特征值:", np.linalg.eigvalsh(A2))        # ← np.linalg.eigvalsh(A2)
print("A2 Cholesky 成功?", np.linalg.cholesky(A2)) # ← np.linalg.cholesky(A2) 不报错即为成功


# ============================================================
# 第二步：半正定矩阵 (PSD)
# ============================================================

#  10: 构造"半正定但非正定"矩阵（存在零特征值）。
#          提示：让 B_psd 的第三列 = 前两列的线性组合（秩亏），再算 B_psd.T @ B_psd
B_psd = np.array([[1., 0., 1.],
          [0., 1., 1.],
          [2., -1., 1.]])  # ← 秩亏的 3x3 矩阵
A3 = B_psd.T @ B_psd     # ← B_psd.T @ B_psd

#  11: 验证最小特征值 = 0
print("A3 特征值:", np.linalg.eigvalsh(A3))  # ← np.linalg.eigvalsh(A3)，看是否有 ≈0

#  11b: 观察 Cholesky 失败（用 try/except 包住，避免脚本中断）
try:
    np.linalg.cholesky(A3)
    print("A3 Cholesky 成功（意外！）")
except np.linalg.LinAlgError:
    print("A3 Cholesky 失败 → 不是正定，只是半正定")

#  12: 验证 rank = 非零特征值的个数
print("A3 的秩:", np.linalg.matrix_rank(A3))  # ← np.linalg.matrix_rank(A3)

#  13: 找一个非零向量 x，使 xᵀ A3 x = 0
#          提示：np.linalg.eigh(A3) 返回 (特征值, 特征向量)，取最小特征值对应的那列
evals3, evecs3 = np.linalg.eigh(A3)  # ← 填代码
x_null = evecs3[:, 0]                # ← evecs3[:, 0]（eigh 升序，第一列对应最小特征值）
print("xᵀ A3 x =", x_null @ A3 @ x_null)     # ← x_null @ A3 @ x_null，应 ≈ 0


# ============================================================
# 第三步：不稳定状态矩阵
# ============================================================

#  14: 构造一个特征值实部 > 0 的矩阵 A（系统 ẋ = Ax 不稳定）
#          提示：先从对称对角矩阵开始，比如 diag(1, -2)
A_unstable = np.diag([1, -2])  # ← 填代码

#  15: 验证存在特征值实部 > 0
#          提示：np.linalg.eigvals(A_unstable).real，检查是否有 > 0 的项
print("A_unstable 特征值实部:", np.linalg.eigvals(A_unstable).real)

#  16: 关键对照——对称负定 → 稳定；对称正定 → 不稳定
#          各构造一个 2x2 对称矩阵验证其特征值符号
A_neg = np.diag([-2.0, -1.0])  # ← 对称负定：所有特征值 < 0
A_pos = np.diag([1.0, 2.0])  # ← 对称正定：所有特征值 > 0
print("A_neg 特征值:", np.linalg.eigvalsh(A_neg))  # 全负 → 系统稳定
print("A_pos 特征值:", np.linalg.eigvalsh(A_pos))  # 全正 → 系统不稳定

#  17: 模拟 ẋ = Ax 的轨迹，观察发散 vs 收敛
#          提示：x(t) = expm(A t) x0，用 scipy.linalg.expm
#          或离散递推 x_{k+1} = (I + dt A) x_k
from scipy.linalg import expm

x0 = np.array([1., 0.5])      # ← 非零初值，例如 np.array([1., 0.5])
A_sym = A_pos   # ← 用一个对称矩阵做实验（先正定后负定对比）
t = 1.0        # 时刻 t
x_t = expm(A_sym * t) @ x0     # ← expm(A_sym * t) @ x0
print("x(t=1):", x_t)


# ============================================================
# 第四步：秩亏矩阵
# ============================================================

#  18: 用两种方式构造秩亏矩阵
#          方式 a：两列线性相关（例如第一列 = 2 * 第二列）
A_rankdef_a = np.array([[2., 1., 0.],
                       [4., 2., 1.],
                       [6., 3., 0.]])
#          方式 b：外积 uuᵀ（秩 1）
u = np.array([1., 2., 3.])
A_rankdef_b = np.outer(u, u)

#  19: 验证秩、以及用 SVD 看奇异值
print("A_rankdef_a 的秩:", np.linalg.matrix_rank(A_rankdef_a))
print("A_rankdef_b 的秩:", np.linalg.matrix_rank(A_rankdef_b))
print("A_rankdef_b 奇异值:", np.linalg.svd(A_rankdef_b, compute_uv=False))



#  20: 总结（写在你自己的注释里，用你自己的话）
# ① 对称矩阵：正定/半正定/负定 由特征值符号决定
# ② 秩 = 非零特征值个数（对称矩阵）
# ③ 稳定性看特征值实部（不要求对称）；对称正定 ⇒ 发散，对称负定 ⇒ 收敛
