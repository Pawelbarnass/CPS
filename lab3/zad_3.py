import numpy as np
import matplotlib.pyplot as plt
from time import sleep

# Parametry sygnału
fs = 1000  # Częstotliwość próbkowania [Hz]
N = 100    # Liczba próbek
t = np.arange(N) / fs  # Wektor czasu [s]

# Generowanie sygnału: suma trzech sinusoid
f1, f2, f3 = 50, 100, 150  # Częstotliwości składowych [Hz]
A1, A2, A3 = 50, 100, 150  # Amplitudy składowych

# Tworzenie oryginalnego sygnału x
x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t) + A3 * np.sin(2 * np.pi * f3 * t)

# Budowanie macierzy DCT (A) i IDCT (S)
A = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A[k, n] = s_k * np.cos((np.pi * k / N) * (n + 0.5))

S = A.T  # Macierz IDCT jest transponowaną macierzą DCT

# Wyświetlanie każdego wiersza A i odpowiadającej kolumny S
plt.figure(figsize=(10, 6))
for i in range(N):
    plt.clf()
    plt.plot(A[i, :], 'b-', label=f'Wiersz {i} macierzy A (DCT)')
    plt.plot(S[:, i], 'r--', label=f'Kolumna {i} macierzy S (IDCT)')
    plt.title(f'Funkcje bazowe DCT/IDCT: indeks {i}')
    plt.legend()
    plt.grid(True)
    plt.ylim(-0.3, 0.3)
    plt.draw()
    plt.pause(0.05)  # Krótka pauza dla widoczności
    # Usuń sleep() dla pełnego wyświetlania lub odkomentuj dla manualnego trybu
    # sleep(0.5)  # Dłuższa pauza dla lepszej obserwacji

# Analiza sygnału: y = A @ x
y = A @ x

# Wyświetlanie współczynników transformaty
plt.figure(figsize=(12, 8))
plt.subplot(211)
plt.stem(y)
plt.title('Współczynniki DCT sygnału')
plt.xlabel('Indeks n')
plt.ylabel('Amplituda')
plt.grid(True)

# Skalowanie osi poziomej w Hz
f_axis = np.arange(N) * fs / (2*N)
plt.subplot(212)
plt.stem(f_axis, y)
plt.title('Współczynniki DCT z osią częstotliwości')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid(True)
plt.tight_layout()

# Sprawdzenie rekonstrukcji
x_reconstructed = S @ y
print(f"Maksymalna różnica po rekonstrukcji: {np.max(np.abs(x - x_reconstructed))}")
print(f"Czy rekonstrukcja jest perfekcyjna? {np.allclose(x, x_reconstructed)}")

# Modyfikacja: zmiana f2 na 105 Hz
f2_mod = 110
x_mod = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2_mod * t) + A3 * np.sin(2 * np.pi * f3 * t)
y_mod = A @ x_mod

plt.figure(figsize=(12, 6))
plt.subplot(211)
plt.stem(f_axis, y)
plt.title('DCT dla oryginalnego sygnału (f2 = 100 Hz)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid(True)

plt.subplot(212)
plt.stem(f_axis, y_mod)
plt.title('DCT dla zmodyfikowanego sygnału (f2 = 105 Hz)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid(True)
plt.tight_layout()

# Sprawdzenie rekonstrukcji dla zmodyfikowanego sygnału
x_mod_reconstructed = S @ y_mod
print(f"\nDla f2 = 105 Hz:")
print(f"Maksymalna różnica po rekonstrukcji: {np.max(np.abs(x_mod - x_mod_reconstructed))}")
print(f"Czy rekonstrukcja jest perfekcyjna? {np.allclose(x_mod, x_mod_reconstructed)}")

# Przesunięcie wszystkich częstotliwości o 2.5 Hz
f1_shift, f2_shift, f3_shift = f1 + 2.5, f2 + 2.5, f3 + 2.5
x_shift = A1 * np.sin(2 * np.pi * f1_shift * t) + A2 * np.sin(2 * np.pi * f2_shift * t) + A3 * np.sin(2 * np.pi * f3_shift * t)
y_shift = A @ x_shift

plt.figure(figsize=(12, 6))
plt.stem(f_axis, y_shift)
plt.title('DCT dla sygnału z przesunięciem o 2.5 Hz')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid(True)

# Sprawdzenie rekonstrukcji dla sygnału z przesunięciem
x_shift_reconstructed = S @ y_shift
print(f"\nPo przesunięciu o 2.5 Hz:")
print(f"Maksymalna różnica po rekonstrukcji: {np.max(np.abs(x_shift - x_shift_reconstructed))}")
print(f"Czy rekonstrukcja jest perfekcyjna? {np.allclose(x_shift, x_shift_reconstructed)}")

plt.show()