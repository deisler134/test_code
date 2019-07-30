import time
import numpy as np
import torch

x = np.random.random((3,5))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
x_cpu = torch.from_numpy(x).float()
x_gpu = torch.from_numpy(x).float().to(device)

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
    ret = pairwise_numpy(x)
print('pairwise_numpy:',time.time()-start)
print(ret)

start = time.time()
for _ in range(10):
    ret = pairwise_python(x)
print('pairwise_python:',time.time()-start)
print(ret)


@torch.jit.script
def pairwise_torch(X):
    M = X.shape[0]
    N = X.shape[1]
    D = torch.empty(M, M)
    for i in range(M):
        for j in range(M):
            d = torch.zeros(1)
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = torch.squeeze(torch.sqrt(d))
    return D


start = time.time()
for _ in range(10):
    ret = pairwise_torch(x_cpu)
    #torch.jit.trace(pairwise_torch,(x_cpu))
print('pairwise_torch:',time.time()-start)
print(ret)

start = time.time()
for _ in range(10):
    ret = pairwise_torch(x_gpu)
    #torch.jit.trace(pairwise_torch,(x_gpu))
print('pairwise_torch_gpu:',time.time()-start)
print(ret)


