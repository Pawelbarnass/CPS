import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import chebwin, blackman, hamming

fs = 1000
N = 100
f1, f2 = 100, 125
A1, A2 = 1, 0.0001

t = np.linspace(0, (N-1)/fs, N)
x = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t)

f = np.arange(0, 500, 0.1)

X_dtft = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * f / fs * np.arange(N))) / N for f in f])

plt.figure(figsize=(5, 5))
plt.plot(f, np.abs(X_dtft), label="DTFT (Bez okna)")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo DTFT")
plt.grid()
plt.legend()
plt.show()

rect_window = np.ones(N)
hamming_window = hamming(N)
blackman_window = blackman(N)
chebyshev_100 = chebwin(N, at=100)
chebyshev_120 = chebwin(N, at=120)

X_rect = np.array([np.sum(x * rect_window * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])
X_hamming = np.array([np.sum(x * hamming_window * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])
X_blackman = np.array([np.sum(x * blackman_window * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])
X_cheby_100 = np.array([np.sum(x * chebyshev_100 * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])
X_cheby_120 = np.array([np.sum(x * chebyshev_120 * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])

plt.figure(figsize=(5, 5))
plt.plot(f, np.abs(X_rect), label="Prostokątne")
plt.plot(f, np.abs(X_hamming), label="Hamming")
plt.plot(f, np.abs(X_blackman), label="Blackman")
plt.plot(f, np.abs(X_cheby_100), label="Czebyszew 100dB")
plt.plot(f, np.abs(X_cheby_120), label="Czebyszew 120dB")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo dla różnych okien")
plt.legend()
plt.grid()
plt.show()

N = 1000
t = np.linspace(0, (N-1)/fs, N)
x = A1 * np.cos(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t)

chebyshev_1000_100 = chebwin(N, at=100)
chebyshev_1000_120 = chebwin(N, at=120)

X_cheby_1000_100 = np.array([np.sum(x * chebyshev_1000_100 * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])
X_cheby_1000_120 = np.array([np.sum(x * chebyshev_1000_120 * np.exp(-1j * 2 * np.pi * freq / fs * np.arange(N))) for freq in f])

plt.figure(figsize=(5, 5))
plt.plot(f, np.abs(X_cheby_1000_100), label="Czebyszew 100dB")
plt.plot(f, np.abs(X_cheby_1000_120), label="Czebyszew 120dB")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo DTFT dla N=1000, okna Czebyszewa")
plt.legend()
plt.grid()
plt.show()