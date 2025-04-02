import numpy as np
import matplotlib.pyplot as plt

# Parametry sygnału
fpr = 1000  # częstotliwość próbkowania [Hz]
N = 100     # liczba próbek
dt = 1/fpr  # krok czasowy
t = dt * np.arange(N)  # wektor czasu (0, dt, 2dt, ..., (N-1)dt)

# Generowanie sygnału sinusoidalnego
A = 5
f0 = 50  # częstotliwość sygnału [Hz]
x = A * np.cos(2 * np.pi * f0 * t)

# Wykres sygnału w dziedzinie czasu
plt.figure()
plt.plot(t, x, 'b-o', markersize=4)
plt.xlabel('t [s]')
plt.title('x(t)')
plt.grid(True)
plt.show()

# Obliczenie FFT
X = np.fft.fft(x)         # Transformata Fouriera
f = (fpr/N) * np.arange(N)  # wektor częstotliwości

# Wykres widma amplitudowego
plt.figure()
plt.plot(f, (1/N) * np.real(X), 'b-o', markersize=4)  # Normalizacja przez N
plt.xlabel('f [Hz]')
plt.title('|X(k)|')
plt.grid(True)
plt.show()