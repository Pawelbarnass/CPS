import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate

# -------------------------------
# 1. Wczytanie sygnału z pliku MAT
# -------------------------------
mat_data = scipy.loadmat('adsl_x.mat')
x = mat_data['x'].flatten()
print(f"Długość sygnału: {len(x)} próbek")

# -------------------------------
# 2. Ustawienie parametrów sygnału
# -------------------------------
M = 32    # długość prefiksu
N = 512   # liczba próbek bloku bez prefiksu
block_length = M + N  # całkowita długość bloku = 544
L = len(x)

# -------------------------------
# 3. Obliczenie metryki korelacji
# -------------------------------
# Każdy prefiks jest kopią OSTATNICH M próbek bloku danych
# Dla każdej pozycji n, sprawdzamy korelację między segmentem [n:n+M] 
# a segmentem [n+N:n+N+M] (segment oddalony o długość danych)
peaks = []
for n in range(L - M + 1):
    # Obliczamy korelację między potencjalnym prefiksem a końcówką bloku danych
   corr = correlate(x, x[n:n+M], mode='valid')  # Correlation of prefix with entire signal
   max_val = np.max(np.abs(corr))  # Find maximum value
   max_pos = np.where(np.abs(corr) == max_val)[0]
   print (max_pos)
   print(len(max_pos))  # Find all positions with maximum value
   if len(max_pos) >= 2:  # If there are at least 2 maximum positions
        peaks.append(max_pos) # Sto
# -------------------------------
# 4. Wykrywanie pozycji prefiksów
# -------------------------------
# Znajdź wysokie wartości korelacji
# Użyj height parameter by odrzucić niskie wartości

print("Wykryte pozycje (indeksy) początków prefiksów:")
print(peaks)
print("Odległości między kolejnymi prefiksami:")
print(np.diff(peaks))
print(len(corr))

# -------------------------------
# Wizualizacja fragmentu sygnału z zaznaczonymi prefiksami
