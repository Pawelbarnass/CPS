import numpy as np
import matplotlib.pyplot as plt

# Parametry sygnału
N = 100          # Liczba próbek
fpr = 1000       # Częstotliwość próbkowania [Hz]
t = np.arange(N) / fpr  # Oś czasu

# Generowanie dwóch rzeczywistych sygnałów testowych
f1, f2 = 50, 100  # Częstotliwości
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.cos(2 * np.pi * f2 * t)

# Łączenie sygnałów w zespolony
y = x1 + 1j * x2
Y = np.fft.fft(y)

# Inicjalizacja tablic wynikowych
X1 = np.zeros(N, dtype=complex)
X2 = np.zeros(N, dtype=complex)

# Obsługa składowej stałej (k=0) i Nyquista (k=N/2 jeśli N parzyste)
X1[0] = (Y[0] + np.conj(Y[0])) / 2
X2[0] = (Y[0] - np.conj(Y[0])) / (2j)

if N % 2 == 0:  # Jeśli N parzyste, obsługa Nyquista
    X1[N//2] = (Y[N//2] + np.conj(Y[N//2])) / 2
    X2[N//2] = (Y[N//2] - np.conj(Y[N//2])) / (2j)

# Obliczenia dla pozostałych częstotliwości
for k in range(1, N//2 + (0 if N % 2 == 0 else 1)):
    if k == N//2:  # Skip if already handled
        continue
        
    # Wzór: X1[k] = (Y[k] + conj(Y[N-k]))/2 i X2[k] = (Y[k] - conj(Y[N-k]))/(2j)
    X1[k] = (Y[k] + np.conj(Y[N-k])) / 2
    X2[k] = (Y[k] - np.conj(Y[N-k])) / (2j)
    
    # Zachowanie symetrii hermitowskiej dla rzeczywistych sygnałów
    X1[N-k] = np.conj(X1[k])
    X2[N-k] = np.conj(X2[k])

# Obliczenia referencyjne FFT
X1_ref = np.fft.fft(x1)
X2_ref = np.fft.fft(x2)

# Weryfikacja poprawności
print("Czy X1 się zgadza?", np.allclose(X1, X1_ref, atol=1e-10))
print("Czy X2 się zgadza?", np.allclose(X2, X2_ref, atol=1e-10))