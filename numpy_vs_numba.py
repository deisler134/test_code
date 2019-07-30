import time
import numpy as np

x = np.random.random((3,5))

def pairwise_numpy(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))

def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

start = time.time()
for _ in range(10):
    pairwise_numpy(x)
print('pairwise_numpy:',time.time()-start)

start = time.time()
for _ in range(10):
    pairwise_python(x)
print('pairwise_python:',time.time()-start)

from numba import double
from numba.decorators import jit, autojit

#pairwise_py_numba = autojit(pairwise_python)
#pairwise_np_numba = autojit(pairwise_numpy)

@jit
def pairwise_np_numba(X):
    return np.sqrt(((X[:, None, :] - X) ** 2).sum(-1))

@jit
def pairwise_py_numba(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D

start = time.time()
for _ in range(10):
    pairwise_np_numba(x)
print('pairwise_np_numba:',time.time()-start)

start = time.time()
for _ in range(10):
    pairwise_py_numba(x)
print('pairwise_py_numba:',time.time()-start)


