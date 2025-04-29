import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import firwin, lfilter, freqz, welch

# Wczytanie pliku MAT
ecg_data = loadmat('ECG100.mat')
ecg_signal = ecg_data['val'][0]  # Zakładając, że sygnał jest w kluczu 'val'
fs = 360  # Częstotliwość próbkowania

# Obliczenie widma
n = len(ecg_signal)
frequencies = np.fft.fftfreq(n, 1/fs)[:n//2]
fft_values = np.fft.fft(ecg_signal)[:n//2]
plt.figure(figsize=(12, 6))
plt.plot(frequencies, np.abs(fft_values))
plt.title('Widmo częstotliwościowe sygnału EKG')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.grid()
plt.show()

# Projektowanie filtra FIR
cutoff = 45  # Częstotliwość odcięcia
numtaps = 101  # Długość filtra (nieparzysta)
nyquist = 0.5 * fs
taps = firwin(numtaps, cutoff/nyquist, window='hamming')

# Kompensacja opóźnienia
delay = (numtaps - 1) // 2

# Filtracja
filtered_signal = lfilter(taps, 1.0, ecg_signal)
filtered_signal = filtered_signal[delay:]

# Przycięcie oryginalnego sygnału
original_trimmed = ecg_signal[:len(filtered_signal)]

# Wykres sygnałów
plt.figure(figsize=(12, 6))
plt.plot(original_trimmed, label='Oryginalny')
plt.plot(filtered_signal, label='Przefiltrowany', alpha=0.7)
plt.title('Porównanie sygnałów przed i po filtracji')
plt.xlabel('Próbki')
plt.ylabel('Amplituda')
plt.legend()
plt.grid()
plt.show()

# Dodanie szumu
noise_level = 0.5
noisy_signal = original_trimmed + noise_level * np.random.randn(len(original_trimmed))

# Filtracja zaszumionego sygnału
filtered_noisy = lfilter(taps, 1.0, noisy_signal)
filtered_noisy = filtered_noisy[delay:]

# Wykres z szumem
plt.figure(figsize=(12, 6))
plt.plot(original_trimmed, label='Oryginalny')
plt.plot(filtered_noisy, label='Przefiltrowany zaszumiony', alpha=0.7)
plt.title('Porównanie sygnałów z szumem')
plt.xlabel('Próbki')
plt.ylabel('Amplituda')
plt.legend()
plt.grid()
plt.show()

# Test różnych parametrów filtra
cutoffs = [35, 45, 55]  # Różne częstotliwości odcięcia
numtaps_list = [51, 101, 201]  # Różne długości filtrów

plt.figure(figsize=(15, 10))
for i, (cutoff, numtaps) in enumerate(zip(cutoffs, numtaps_list), 1):
    taps = firwin(numtaps, cutoff/nyquist, window='hamming')
    delay = (numtaps - 1) // 2
    filtered = lfilter(taps, 1.0, noisy_signal)[delay:]
    
    plt.subplot(3, 1, i)
    plt.plot(original_trimmed, label='Oryginalny')
    plt.plot(filtered, label=f'cutoff={cutoff} Hz, taps={numtaps}', alpha=0.7)
    plt.legend()
    plt.grid()

plt.suptitle('Porównanie różnych parametrów filtra')
plt.tight_layout()
plt.show()