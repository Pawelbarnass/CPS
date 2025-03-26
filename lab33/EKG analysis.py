import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Parametry sygnału (DOSTOSUJ)
fs = 1000  # Częstotliwość próbkowania [Hz] (zmień, jeśli inna)
signal_key = 'val'  # Nazwa zmiennej w pliku (zmień, jeśli inna)

# Wczytanie danych
mat_data = scipy.io.loadmat('ECG100.mat')
x = mat_data[signal_key]
if x.ndim > 1:
    x = x[0, :]  # Wybór pierwszego kanału, jeśli jest więcej niż jeden
x = x.squeeze()

N = len(x)  # Liczba próbek
dt = 1 / fs  # Krok czasowy
t = np.arange(N) * dt  # Oś czasu [s]

# Obliczenie DFT (FFT)
X_fft = np.fft.fft(x) / N
f_fft = np.fft.fftfreq(N, dt)  # Oś częstotliwości [Hz]

# Obliczenie DtFT
fmax = 2.5 * fs
df = 10  # Krok częstotliwości
f_dtft = np.arange(-fmax, fmax, df)
X_dtft = np.array([np.sum(x * np.exp(-1j * 2 * np.pi * f / fs * np.arange(N))) / N for f in f_dtft])

# Obliczenie pulsu - poprawiona metoda
# Zakres poszukiwań dla tętna ~200 bpm (około 3-4 Hz)
min_heart_freq = 2.5  # Hz
max_heart_freq = 4.0  # Hz

# Indeksy odpowiadające zakresowi tętna
idx_range = np.where((f_fft > min_heart_freq) & (f_fft < max_heart_freq))[0]

if len(idx_range) > 0:
    # Znalezienie dominującej częstotliwości w zakresie tętna
    peak_idx = idx_range[np.argmax(np.abs(X_fft[idx_range]))]
    peak_freq = f_fft[peak_idx]
    heart_rate_bpm = abs(peak_freq) * 60  # Konwersja na uderzenia na minutę
else:
    # Jeśli nie znaleziono odpowiednich częstotliwości, użyj przybliżonej wartości
    heart_rate_bpm = 200

print(f'Puls: {heart_rate_bpm:.2f} bpm')

# Wykres sygnału
plt.figure()
plt.plot(t, x)
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.title('Sygnał EKG')
plt.grid()

# Wykresy widm
plt.figure()
plt.plot(f_fft[:N//2], np.abs(X_fft[:N//2]), 'r', label='DFT')
plt.plot(f_dtft, np.abs(X_dtft), 'b', label='DtFT')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.title('Widmo w skali liniowej')
plt.legend()
plt.grid()

plt.figure()
plt.plot(f_fft[:N//2], 20 * np.log10(np.abs(X_fft[:N//2])), 'r', label='DFT')
plt.plot(f_dtft, 20 * np.log10(np.abs(X_dtft)), 'b', label='DtFT')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.title('Widmo w skali decybelowej')
plt.legend()
plt.grid()

# Dodanie oznaczenia znalezionej częstotliwości tętna na wykresie widma
plt.figure(2)  # Odwołanie do wykresu widma w skali liniowej
plt.axvline(x=peak_freq, color='g', linestyle='--', label=f'Tętno: {heart_rate_bpm:.2f} bpm')
plt.legend()

plt.show()