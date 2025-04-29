import numpy as np
import matplotlib.pyplot as plt
from scipy.signal.windows import blackmanharris
from scipy.signal import welch

# Parametry filtra
fpr = 1200  # Częstotliwość próbkowania [Hz]
df = 200    # Szerokość pasma przepustowego [Hz]
fc = 300    # Częstotliwość środkowa pasma [Hz]
fl = fc - df/2  # Dolna częstotliwość odcięcia [Hz]
fh = fc + df/2  # Górna częstotliwość odcięcia [Hz]
N = 129     # Długość filtru (nieparzysta)
M = (N - 1) // 2  # Indeks środkowy

# Znormalizowane częstotliwości
fl_norm = fl / fpr
fh_norm = fh / fpr

# Idealna odpowiedź impulsowa filtra pasmowoprzepustowego
n = np.arange(N)
k = n - M  # Przesunięte indeksy

h_ideal = np.zeros(N)
for i in range(N):
    ki = k[i]
    if ki == 0:
        h_ideal[i] = 2 * (fh_norm - fl_norm)
    else:
        h_ideal[i] = (np.sin(2 * np.pi * fh_norm * ki) - np.sin(2 * np.pi * fl_norm * ki)) / (np.pi * ki)

# Definicje okien
windows = {
    'Prostokątne': np.ones(N),
    'Hanninga': np.hanning(N),
    'Hamminga': np.hamming(N),
    'Blackmana': np.blackman(N),
    'Blackmana-Harrisa': blackmanharris(N)
}

# Tworzenie filtrów poprzez zastosowanie okien
filtry = {}
for nazwa, okno in windows.items():
    filtry[nazwa] = h_ideal * okno

# Obliczenie charakterystyk częstotliwościowych
num_freqs = 4096  # Liczba punktów FFT
czestotliwosci = np.fft.fftfreq(num_freqs, 1/fpr)
czestotliwosci = np.fft.fftshift(czestotliwosci)

plt.figure(figsize=(12, 8))
for i, (nazwa, h) in enumerate(filtry.items()):
    H = np.fft.fftshift(np.fft.fft(h, num_freqs))
    modul = 20 * np.log10(np.abs(H) + 1e-9)
    faza = np.unwrap(np.angle(H))
    
    # Wykres charakterystyki amplitudowej
    plt.subplot(2, 1, 1)
    plt.plot(czestotliwosci, modul, label=nazwa)
    
    # Wykres charakterystyki fazowej
    plt.subplot(2, 1, 2)
    plt.plot(czestotliwosci, faza, label=nazwa)

plt.subplot(2, 1, 1)
plt.title('Charakterystyka amplitudowa')
plt.xlim(0, fpr/2)
plt.ylim(-100, 5)
plt.ylabel('Amplituda (dB)')
plt.grid(True)
plt.legend()

plt.subplot(2, 1, 2)
plt.title('Charakterystyka fazowa')
plt.xlim(0, fpr/2)
plt.ylabel('Faza (radiany)')
plt.xlabel('Częstotliwość (Hz)')
plt.grid(True)
plt.legend()

plt.tight_layout()

# Obliczenie tłumienia w paśmie zaporowym
pasmo_zaporowe_niskie = (czestotliwosci >= 0) & (czestotliwosci <= fl)
pasmo_zaporowe_wysokie = (czestotliwosci >= fh) & (czestotliwosci <= fpr/2)
pasmo_zaporowe = pasmo_zaporowe_niskie | pasmo_zaporowe_wysokie

print("Poziom tłumienia w paśmie zaporowym:")
for nazwa in filtry:
    H = np.fft.fftshift(np.fft.fft(filtry[nazwa], num_freqs))
    modul = 20 * np.log10(np.abs(H) + 1e-9)
    tlumienie = np.min(modul[pasmo_zaporowe])
    print(f"{nazwa}: {tlumienie:.2f} dB")

# Generowanie sygnału testowego
t = np.arange(0, 3, 1/fpr)  # 1 sekunda
x = np.sin(2 * np.pi * 100 * t) + np.sin(2 * np.pi * 300 * t) + np.sin(2 * np.pi * 500 * t)
x = x - np.mean(x)  # Usunięcie składowej stałej

# Obliczenie widma sygnału oryginalnego
window = np.hanning(1024)
czestotliwosci_x, psd_x = welch(x, fs=fpr, window=window, nperseg=1024, detrend='constant')
psd_x_db = 10 * np.log10(psd_x + 1e-12)

plt.figure(figsize=(10, 5))
plt.plot(czestotliwosci_x, psd_x_db, label='Sygnał przed filtracją', linewidth=2)
plt.title('Widmo gęstości mocy sygnału przed filtracją')
plt.xlim(0, fpr/2)
plt.ylim(-60, 60)
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Gęstość mocy (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
plt.figure()
plt.plot(t, x)
plt.title("Sygnał w dziedzinie czasu")
plt.show()

# Filtracja i obliczenie widma dla każdego filtru
for nazwa in filtry:
    h = filtry[nazwa]
    przefiltrowany = np.convolve(x, h, mode='same')
    czestotliwosci_filtr, psd_filtr = welch(
        przefiltrowany,
        fs=fpr,
        window=window,
        nperseg=1024,
        detrend='constant'
    )
    psd_filtr_db = 10 * np.log10(psd_filtr + 1e-12)
    plt.plot(czestotliwosci_filtr, psd_filtr_db, label=nazwa)

plt.title('Widmo gęstości mocy sygnału po filtracji różnymi oknami')
plt.xlim(0, fpr/2)
plt.ylim(-60, 60)
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Gęstość mocy (dB)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.show()