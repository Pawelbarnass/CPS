import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from time import sleep


# Wczytywanie pliku audio
fs, x = wavfile.read('name_bpsk.wav')
# Konwersja do wartości zmiennoprzecinkowych dla łatwiejszej obróbki
if x.dtype == np.int16:
    x = x.astype(np.float32) / 32768.0

# Jeśli sygnał jest stereo, bierzemy tylko jeden kanał
if len(x.shape) > 1:
    x = x[:, 0]

# Wyświetlanie całego sygnału
plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(x)), x)
plt.title('Sygnał mowy')
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()

# Tworzenie macierzy DCT (A) jak w zad_1
N = 256  # Długość fragmentu
A = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A[k, n] = s_k * np.cos((np.pi * k / N) * (n + 0.5))

# Wybór 10 fragmentów sygnału
# Można zmodyfikować te indeksy po wizualnej inspekcji sygnału
fragment_starts = []
total_samples = len(x)

# Automatyczne wybranie fragmentów równomiernie rozłożonych w sygnale
if total_samples >= N * 10:  # Sprawdzenie czy sygnał jest wystarczająco długi
    step = (total_samples - N) // 10
    fragment_starts = [i * step for i in range(10)]
else:
    # Jeśli sygnał jest za krótki, wybieramy mniej fragmentów lub z zakładkami
    overlap = max(0, (10 * N - total_samples) // 9)  # Minimalna zakładka potrzebna
    step = N - overlap
    fragment_starts = [i * step for i in range(min(10, total_samples // step))]

# Analiza i wyświetlanie fragmentów
plt.figure(figsize=(12, 10))
for i, start in enumerate(fragment_starts):
    # Upewnienie się, że fragment nie wykracza poza sygnał
    if start + N <= len(x):
        # Wyodrębnienie fragmentu
        fragment = x[start:start+N]
        
        # Obliczenie DCT
        y = A @ fragment
        
        # Skalowanie osi częstotliwości
        f_axis = np.arange(N) * fs / (2 * N)
        
        # Wyświetlanie fragmentu i jego DCT
        plt.clf()  # Czyszczenie wykresu
        
        # Fragment czasowy
        plt.subplot(211)
        plt.plot(np.arange(N), fragment)
        plt.title(f'Fragment {i+1} (próbki {start}-{start+N-1})')
        plt.xlabel('Numer próbki w fragmencie')
        plt.ylabel('Amplituda')
        plt.grid(True)
        
        # Wynik DCT
        plt.subplot(212)
        plt.stem(f_axis[:N//2], np.abs(y[:N//2]))  # Pokazujemy tylko połowę współczynników (do fs/2)
        plt.title(f'DCT fragmentu {i+1}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Amplituda')
        plt.grid(True)
        
        plt.tight_layout()
        plt.draw()
        plt.pause(1)  # Pauza 1 sekunda między fragmentami
        
        # Opcjonalnie: zapisywanie wykresów do plików
        # plt.savefig(f'fragment_{i+1}.png')

# Zatrzymanie ostatniego wykresu
plt.show()