import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from time import sleep


fs, x = wavfile.read('mowa_nagranie.wav')
if x.dtype == np.int16:
    x = x.astype(np.float32) / 32768.0

plt.figure(figsize=(12, 6))
plt.plot(np.arange(len(x)), x)
plt.title('Sygnał mowy')
plt.xlabel('Numer próbki')
plt.ylabel('Amplituda')
plt.grid(True)
plt.show()

N = 256  
A = np.zeros((N, N))
for k in range(N):
    s_k = np.sqrt(1/N) if k == 0 else np.sqrt(2/N)
    for n in range(N):
        A[k, n] = s_k * np.cos((np.pi * k / N) * (n + 0.5))

fragment_starts = []
total_samples = len(x)

if total_samples >= N * 10: 
    step = (total_samples - N) // 10
    fragment_starts = [i * step for i in range(10)]
else:
    overlap = max(0, (10 * N - total_samples) // 9)
    step = N - overlap
    fragment_starts = [i * step for i in range(min(10, total_samples // step))]

plt.figure(figsize=(12, 10))
for i, start in enumerate(fragment_starts):
    if start + N <= len(x):
        fragment = x[start:start+N]
        y = A @ fragment
        f_axis = np.arange(N) * fs / (2 * N)
        plt.clf()
        
        plt.subplot(211)
        plt.plot(np.arange(N), fragment)
        plt.title(f'Fragment {i+1} (próbki {start}-{start+N-1})')
        plt.xlabel('Numer próbki w fragmencie')
        plt.ylabel('Amplituda')
        plt.grid(True)
        
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