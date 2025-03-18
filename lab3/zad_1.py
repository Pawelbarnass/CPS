import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
N = 20
A = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A[k, n] = s_k * np.cos((np.pi * k / N) * (n + 0.5))
for k in range(N):
    for n in range(N):
        iloczyn = dot(A[k, :], A[n, :])
        if k != n and not np.isclose(iloczyn, 0, atol=1e-10):
            print(f"Macierz A nie jest ortogonalna, bo iloczyn skalarny wierszy {k} i {n} wynosi 0")
            break
    else:
        print(f"Macierz A jest ortogonalna")