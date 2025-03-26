import numpy as np
import matplotlib.pyplot as plt

N = 100
fs = 1000
f1, f2 = 100, 200
A1, A2 = 100, 200
phi1, phi2 = np.pi / 7, np.pi / 11

t = np.linspace(0, (N-1)/fs, N)
x = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)

k = np.arange(N).reshape((N, 1))
n = np.arange(N)
A = np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

X = A @ x
freqs = np.arange(N) * fs / N

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.stem(freqs, X.real)
plt.title("Część rzeczywista")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 2)
plt.stem(freqs, X.imag)
plt.title("Część urojona")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 3)
plt.stem(freqs, np.abs(X))
plt.title("Moduł")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 4)
plt.stem(freqs, np.angle(X))
plt.title("Faza")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Kąt [rad]")

plt.tight_layout()
plt.show()

B = np.conj(A.T)
x_r = B @ X

print("Czy x == xr?", np.allclose(x, x_r))

X_fft = np.fft.fft(x) / np.sqrt(N)
x_r_fft = np.fft.ifft(X_fft) * np.sqrt(N)

print("X - FFT == DFT", np.allclose(X, X_fft))
print("xr - FFT == DFT", np.allclose(x_r, x_r_fft))

print("X == X_fft?", np.allclose(X, X_fft))
print("xr == x_r_fft?", np.allclose(x_r, x_r_fft))

f1 = 125
x_new = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)
X_new = A @ x_new

plt.figure(figsize=(12, 6))
plt.stem(freqs, np.abs(X_new))
plt.title("Widmo amplitudowe dla f1 = 125 Hz")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")
plt.show()
