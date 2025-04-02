import numpy as np
import matplotlib.pyplot as plt
from numpy import dot
N = 20
A = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A[k, n] = s_k * np.cos((np.pi * k / N) * (n + 0.5))
S = A.T
I = A @ S
print("Maksymalne odchylenie od macierzy jednostkowej:",
      np.max(np.abs(I - np.eye(N))))
print("Elementy diagonalne:", np.diag(I))
x = np.random.randn(N)
X = A @ x.T
x_s = S @ X
print("Maksymalne odchylenie od wektora x:", np.max(np.abs(x - x_s)))
plt.figure()
t = np.linspace(0,1,N,endpoint=False)
f =7
x_harmonic = np.sin(2*np.pi*f*t)
np.random.seed(69)
x_noise = np.random.randn(N)
A_broken = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A_broken[k, n] = s_k * np.cos((np.pi * (k+0.25) / N) * (n + 0.5))
S_broken = A_broken.T
I_broken = A_broken @ S_broken
print("max odchylenie zepsutej macierzy" , np.max(np.abs(I_broken - np.eye(N))))
X_harmonic = A_broken @ x_harmonic.T
x_harmonic_s = S_broken @ X_harmonic
print("Maksymalna różnica po rekonstrukcji:", 
      np.max(np.abs(x_harmonic_s - x_harmonic)))
print("Czy rekonstrukcja jest perfekcyjna?", 
      np.allclose(x_harmonic_s,x_harmonic, rtol=1e-10, atol=1e-10))
X_noise = A_broken @ x_noise.T
x_noise_s = S_broken @ X_noise
print("Maksymalna różnica po rekonstrukcji:", 
      np.max(np.abs(x_noise_s - x_noise)))
print("Czy rekonstrukcja jest perfekcyjna?", 
      np.allclose(x_noise_s,x_noise, rtol=1e-10, atol=1e-10))