import numpy as np


def get_W():
    rank = 2
    W = np.random.randn(1000, rank) @ np.random.randn(rank, 1000)
    return W

W = get_W()
b = np.random.randn(1000)
x = np.random.randn(1000)
y = W @ x + b

# SVD - singular value decomposition
U, S, V = np.linalg.svd(W, full_matrices=True)
V = V.T

estimated_rank = 2
U_r = U[:, 0:estimated_rank]
S_r = np.diag(S[0:estimated_rank])
V_r = V[:, 0:estimated_rank]

A = V_r.T
B = U_r @ S_r

W_hat = B @ A
y_hat = W_hat @ x + b

print("W shape:", W.shape)
print("A shape:", A.shape)
print("B shape:", B.shape)
print("params in W:", W.size)
print("params in A + B:", A.size + B.size)
print("top singular values:", S[0:5])
print("reconstruction error ||W - W_hat||:", np.linalg.norm(W - W_hat))
print("output error ||y - y_hat||:", np.linalg.norm(y - y_hat))
