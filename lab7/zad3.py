import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
from scipy import signal
from scipy.signal import firwin, remez, freqz, lfilter, decimate
from scipy.fft import fft, fftfreq

# Załóżmy, że mamy wczytany sygnał hybrydowy FM (tutaj przykładowe dane)
# W praktyce należy wczytać plik .raw lub .mat np.:
# data = np.fromfile('samples_100MHz_fs3200kHz.raw', dtype=np.int16)
# Dla pliku .mat:
# mat_data = loadmat('stereo_samples_fs1000kHz_LR_IQ.mat')
# signal = mat_data['sygnal']

# Generowanie przykładowego sygnału (dla testu)
fs = 3200000  # częstotliwość próbkowania
t = np.arange(0, 1, 1/fs)
fs = 3200000  # częstotliwość próbkowania
filename = r'C:\Users\wiedzmok\CPS\samples_100MHz_fs3200kHz_5MB.raw'
sygnal = np.fromfile(filename, dtype=np.int16).astype(np.float32)

# Normalizacja (opcjonalnie)
sygnal = sygnal / np.max(np.abs(sygnal))
# --------------------------------------------
# 1. Odzyskanie sygnału pilota
# --------------------------------------------
# Projekt filtra BP 19 kHz (przykład dla okna Hanninga)
nyq = 0.5 * fs
cutoff = [18900, 19100]  # pasmo przepustowe
n = 129  # długość filtra (nieparzysta dla liniowej fazy)

# Projekt filtra FIR BP
b = firwin(n, cutoff, window='hann', pass_zero=False, fs=fs)

# Filtracja
pilot_filtered = lfilter(b, 1.0, sygnal)

# Obliczenie widma
f, Pxx = signal.welch(pilot_filtered, fs, nperseg=1024)
Pxx_db = 10*np.log10(Pxx)

# Znajdź dokładną częstotliwość pilota
mask = (f >= 18000) & (f <= 20000)
fpi = f[mask][np.argmax(Pxx_db[mask])]

print(f'Znaleziona częstotliwość pilota: {fpi/1000:.2f} kHz')

# --------------------------------------------
# 2. Projekt filtra BP dla sygnału L-R
# --------------------------------------------
f_center = 2 * fpi  # środek pasma
bandwidth = 15000    # przykładowa szerokość pasma

# Wymagania tłumienia
ripple = 0.1   # zafalowania w paśmie [dB]
atten = 60     # tłumienie w paśmie zaporowym [dB]

# Projekt filtra metodą Remez
bp_low = f_center - bandwidth/2
bp_high = f_center + bandwidth/2
b_stereo = firwin(2001, [bp_low, bp_high], pass_zero=False, fs=fs)

# Charakterystyka filtra
w, h = freqz(b_stereo, fs=fs)
plt.plot(w, 20*np.log10(np.abs(h)))
plt.title('Charakterystyka filtra stereo')
plt.xlabel('Częstotliwość [Hz]')

# --------------------------------------------
# 3. Filtracja i przesunięcie częstotliwości
# --------------------------------------------
# Filtracja sygnału L-R
l_r_filtered = lfilter(b_stereo, 1.0, sygnal)

# Przesunięcie w dół przez mnożenie z cosinusem
t_shift = np.arange(len(l_r_filtered)) / fs
carrier = np.cos(2*np.pi*2*fpi*t_shift)
l_r_shifted = l_r_filtered * carrier

# Widma
plt.figure()
plt.psd(l_r_filtered, Fs=fs, label='Przed przesunięciem')
plt.psd(l_r_shifted, Fs=fs, label='Po przesunięciu')
plt.xlim(0, 100000)
plt.legend()
plt.show()

# --------------------------------------------
# 4. Decymacja z filtrem antyaliasingowym
# --------------------------------------------
# Filtr antyaliasingowy (przykład)
b_anti = firwin(101, 15000, fs=fs)
l_r_antialiased = lfilter(b_anti, 1.0, l_r_shifted)

# Decymacja
dec_factor = int(fs / 30000)
l_r_decimated = decimate(l_r_antialiased, dec_factor, ftype='fir')

# --------------------------------------------
# 5. Kompensacja opóźnień i rekonstrukcja
# --------------------------------------------
# Oblicz opóźnienia (przykład)
delay_LR = (len(b_stereo)-1)//2 + (len(b_anti)-1)//2
delay_LR_samples = delay_LR // dec_factor

# Przesuń sygnał L+R (przykład)
# (w praktyce należy przefiltrować sygnał L+R odpowiednim filtrem LP)
b_lp = firwin(101, 15000, fs=fs)
ym = lfilter(b_lp, 1.0, sygnal)
ym_shifted = np.roll(ym, -delay_LR_samples)

# Odtworzenie kanałów
ys = l_r_decimated
y1 = 0.5*(ym_shifted[:len(ys)] + ys)
y2 = 0.5*(ym_shifted[:len(ys)] - ys)

# Wykresy porównawcze
plt.figure()
plt.plot(y1[:10000], label='Lewy kanał')
plt.plot(y2[:10000], label='Prawy kanał')
plt.legend()
plt.show()

# Sprawdzenie przesłuchu
cross_talk = 20*np.log10(np.max(np.abs(y1 - y2)))
print(f'Poziom przesłuchu: {cross_talk:.2f} dB')
print("max y1:", np.max(np.abs(y1)), "max y2:", np.max(np.abs(y2)))
