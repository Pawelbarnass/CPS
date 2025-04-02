import numpy as np
import time

def dft(x):
    """Bezpośrednia implementacja DFT o złożoności O(N^2)."""
    N = len(x)
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N):
        for n in range(N):
            X[k] += x[n] * np.exp(-2j * np.pi * k * n / N)
    return X

def combine(X1, X2, N):
    """Łączy dwa mniejsze DFT w większy, wykorzystując współczynniki obrotowe."""
    X = np.zeros(N, dtype=np.complex128)
    for k in range(N // 2):
        W = np.exp(-2j * np.pi * k / N)
        X[k] = X1[k] + W * X2[k]
        X[k + N // 2] = X1[k] - W * X2[k]
    return X

# Generowanie sygnału losowego
np.random.seed(42)
x = np.random.rand(1024)

# Pomiar czasu - Metoda 1: Bezpośrednie DFT
start_time = time.time()
X_direct = dft(x)
direct_time = time.time() - start_time
print(f"Czas wykonania DFT bezpośrednio: {direct_time:.6f} s")

# Pomiar czasu - Metoda 2: FFT z jednym poziomem podziału (512 punktów)
start_time = time.time()
x_even = x[::2]
x_odd = x[1::2]
X1 = dft(x_even)
X2 = dft(x_odd)
X_fft = combine(X1, X2, 1024)
fft_single_time = time.time() - start_time
print(f"Czas wykonania FFT z jednym podziałem: {fft_single_time:.6f} s")
print(f"Przyspieszenie względem DFT bezpośredniego: {direct_time/fft_single_time:.2f}x")

# Pomiar czasu - Metoda 3: FFT z dwoma poziomami podziału (256 punktów)
start_time = time.time()
x_even_even = x_even[::2]
x_even_odd = x_even[1::2]
x_odd_even = x_odd[::2]
x_odd_odd = x_odd[1::2]

X11 = dft(x_even_even)
X12 = dft(x_even_odd)
X1_split = combine(X11, X12, 512)

X21 = dft(x_odd_even)
X22 = dft(x_odd_odd)
X2_split = combine(X21, X22, 512)

X_fft_split = combine(X1_split, X2_split, 1024)
fft_double_time = time.time() - start_time
print(f"Czas wykonania FFT z dwoma podziałami: {fft_double_time:.6f} s")
print(f"Przyspieszenie względem DFT bezpośredniego: {direct_time/fft_double_time:.2f}x")

# Porównanie wyników
print("\nDokładność wyników:")
print("Porównanie X_direct i X_fft:", np.allclose(X_direct, X_fft))
print("Porównanie X_fft i X_fft_split:", np.allclose(X_fft, X_fft_split))
print("Porównanie X_direct i X_fft_split:", np.allclose(X_direct, X_fft_split))

# Podsumowanie przyspieszenia
print("\nPodsumowanie przyspieszenia:")
print(f"FFT z jednym podziałem: {direct_time/fft_single_time:.2f}x szybciej niż DFT bezpośredni")
print(f"FFT z dwoma podziałami: {direct_time/fft_double_time:.2f}x szybciej niż DFT bezpośredni")
print(f"FFT z dwoma podziałami: {fft_single_time/fft_double_time:.2f}x szybciej niż FFT z jednym podziałem")