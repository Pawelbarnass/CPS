import numpy as np
import matplotlib.pyplot as plt

fs = 1000
N = 100
M = 100
f1 = 125
f2 = 200
A1, A2 = 100, 200
phi1, phi2 = np.pi / 7, np.pi / 11

t = np.linspace(0, (N-1)/fs, N)
x = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)

k = np.arange(N).reshape((N, 1))
n = np.arange(N)
A = np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

X1 = A @ x
fx1 = fs * np.arange(N) / N

xz = np.concatenate((x, np.zeros(M)))
X2 = np.fft.fft(xz) / (N+M)
fx2 = fs * np.arange(N + M) / (N + M)

f3 = np.arange(0, 1000, 0.25)
X3 = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * f / fs * np.arange(N))) / N for f in f3])

plt.figure(figsize=(10, 5))
plt.plot(fx1, np.abs(X1), 'o', label='X1 (DFT)')
plt.plot(fx2, np.abs(X2), 'bx', label='X2 (DFT z zerami)')
plt.plot(f3, np.abs(X3), 'k-', label='X3 (DtFT)')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Amplituda')
plt.legend()
plt.title('Widma X1, X2, X3 dla f=0:fs/2')
plt.grid()
plt.show()

f3_full = np.arange(-2000, 2000, 0.25)
X3_full = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * f / fs * np.arange(N))) / N for f in f3_full])

plt.figure(figsize=(10, 5))
plt.plot(fx1, np.abs(X1), 'o', label='X1 (DFT)')
plt.plot(fx2, np.abs(X2), 'bx', label='X2 (DFT z zerami)')
plt.plot(f3_full, np.abs(X3_full), 'k-', label='X3 (DtFT)')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Amplituda')
plt.legend()
plt.title('Widma X1, X2, X3 dla f=-2fs:2fs')
plt.grid()
plt.show()