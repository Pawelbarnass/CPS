import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

# -------------------------------
# 1. Wczytanie sygnału z pliku MAT
# -------------------------------
# Upewnij się, że plik 'adsl_x.mat' znajduje się w bieżącym katalogu roboczym.
mat_data = scipy.io.loadmat('adsl_x.mat')
# Zakładamy, że sygnał jest zapisany w zmiennej o nazwie 'adsl_x'
x = mat_data['x'].flatten()  # spłaszczamy do jednowymiarowego wektora
print(len(x))
# -------------------------------
# 2. Ustawienie parametrów sygnału
# -------------------------------
M = 32    # długość prefiksu
N = 512   # liczba próbek bloku bez prefiksu
block_length = M + N  # całkowita długość bloku
L = len(x)

# -------------------------------
# 3. Obliczenie metryki korelacji
# -------------------------------
# Dla każdego możliwego indeksu n obliczamy sumę iloczynów:
# sum_{i=0}^{M-1} x[n+i] * x[n+N+i]
# Jeśli sygnał jest zespolony, można użyć: np.dot(x[n:n+M], np.conj(x[n+N:n+N+M]))
metric = np.array([np.dot(x[n:n+M], x[n+N:n+N+M]) for n in range(L - N - M + 1)])

# -------------------------------
# 4. Wykrywanie pozycji prefiksów
# -------------------------------
# Używamy find_peaks, ustawiając minimalną odległość między pikami bliską długości bloku
# oraz próg wykrywania (tutaj przykładowo połowa maksymalnej wartości metryki)
peaks, properties = find_peaks(metric, distance=block_length * 0.8, height=0.5 * np.max(metric))

print("Wykryte pozycje (indeksy) początków prefiksów:")
print(peaks)

# Wizualizacja metryki korelacji wraz z zaznaczonymi pikami
plt.figure(figsize=(10,4))
plt.plot(metric, label='Metryka korelacji')
plt.plot(peaks, metric[peaks], "x", label='Wykryte prefiksy')
plt.xlabel('Indeks próbek')
plt.ylabel('Wartość korelacji')
plt.title('Wykrywanie prefiksu w sygnale ADSL')
plt.legend()
plt.grid(True)
plt.show()
plt.figure(figsize=(12, 6))

# Choose one of the detected peaks (prefix positions)
if len(peaks) > 0:
    start_pos = peaks[0]  # First detected prefix
    end_pos = start_pos + block_length + M  # Show one complete block plus the next prefix
    
    # Ensure we don't go beyond signal length
    end_pos = min(end_pos, len(x))
    
    # Plot the signal segment
    t = np.arange(start_pos, end_pos)
    plt.plot(t, x[start_pos:end_pos], 'b-', label='Signal')
    
    # Highlight the prefixes
    for peak in peaks:
        if peak >= start_pos and peak + M <= end_pos:
            plt.axvspan(peak, peak + M, color='yellow', alpha=0.3, label='Prefix' if peak == peaks[0] else "")
            
    # Highlight corresponding data portions for the first prefix
    if start_pos + N + M <= end_pos:
        plt.axvspan(start_pos + N, start_pos + N + M, color='green', alpha=0.3, label='Matching data portion')
    
    plt.title('ADSL Signal Structure with Detected Prefixes')
    plt.xlabel('Sample index')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Verify prefix by comparing values
    prefix = x[start_pos:start_pos+M]
    matching_portion = x[start_pos+N:start_pos+N+M]
    
    print("\nVerifying first prefix:")
    print(f"Correlation between prefix and matching data: {np.corrcoef(prefix, matching_portion)[0,1]:.6f}")
    

# -------------------------------
# 5. (Opcjonalnie) Własna funkcja korelacji wzajemnej
# -------------------------------
def my_xcorr(x, y):
    """
    Oblicza korelację wzajemną dwóch sygnałów x i y.
    Definicja: r_xy[k] = sum_n x[n] * conj(y[n+k])
    Przyjmuje, że x i y są 1D tablicami o tej samej długości.
    Zwraca korelację dla lagów od -(N-1) do N-1.
    """
    N = len(x)
    r = np.zeros(2 * N - 1, dtype=complex)
    # Lag k przyjmujemy z przedziału -(N-1) ... (N-1)
    for k in range(-N + 1, N):
        sum_val = 0
        for n in range(N):
            m = n + k
            if 0 <= m < N:
                sum_val += x[n] * np.conjugate(y[m])
        r[k + N - 1] = sum_val
    return r

# Przykład użycia funkcji my_xcorr:
# Obliczamy korelację pomiędzy dwoma fragmentami prefiksu i odpowiadającej mu części payloadu (dla jednego bloku)
# Wybieramy pierwszy blok (przyjmujemy, że sygnał zaczyna się od początku bloku)
pref = x[0:M]
payload_end = x[N:N+M]
corr = my_xcorr(pref, payload_end)

print("\nPrzykładowa korelacja wzajemna (my_xcorr) między prefiksem a odpowiadającym fragmentem payloadu:")
print(corr)