import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.signal import bilinear_zpk, zpk2tf, freqz, freqs, lfilter
mat = loadmat('butter.mat')
z = mat['z'].flatten()  # zera (z-zera)
p = mat['p'].flatten()  # bieguny (p-bieguny)
k = mat['k'].item()     # wzmocnienie (k-współczynnik)
fs = 16000  # częstotliwość próbkowania
digital_z, digital_p, digital_k = bilinear_zpk(z, p, k, fs)
b, a = zpk2tf(digital_z, digital_p, digital_k)  # współczynniki H(z)
f_analog = np.linspace(0, fs/2, 8000)
w_analog = 2 * np.pi * f_analog
_, h_analog = freqs(zpk2tf(z, p, k)[0], zpk2tf(z, p, k)[1], worN=w_analog)
w_digital, h_digital = freqz(b, a, worN=f_analog, fs=fs)
plt.figure()
plt.plot(f_analog, 20 * np.log10(np.abs(h_analog)), label='Analogowy')
plt.plot(f_analog, 20 * np.log10(np.abs(h_digital)), label='Cyfrowy')
plt.axvline(1189, color='r', linestyle='--', label='Dolna częstotliwość graniczna')
plt.axvline(1229, color='g', linestyle='--', label='Górna częstotliwość graniczna')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.legend()
plt.grid()
plt.show()
t = np.arange(0, 1, 1/fs)
signal = np.sin(2 * np.pi * 1209 * t) + np.sin(2 * np.pi * 1272 * t)
y_manual = np.zeros_like(signal)
for n in range(len(signal)):
    y_manual[n] = b[0] * signal[n]
    for i in range(1, len(b)):
        if n - i >= 0:
            y_manual[n] += b[i] * signal[n - i]
    for i in range(1, len(a)):
        if n - i >= 0:
            y_manual[n] -= a[i] * y_manual[n - i]
y_lfilter = lfilter(b, a, signal)
plt.figure()
plt.plot(t[:200], signal[:200], label='Sygnał oryginalny')
plt.plot(t[:200], y_manual[:200], label='Filtr własny')
plt.plot(t[:200], y_lfilter[:200], label='Filtr scipy', linestyle='--')
plt.xlabel('Czas [s]')
plt.ylabel('Amplituda')
plt.legend()
plt.grid()
plt.show()
freq = np.fft.fftfreq(len(signal), 1/fs)
fft_signal = np.abs(np.fft.fft(signal))
fft_manual = np.abs(np.fft.fft(y_manual))
fft_lfilter = np.abs(np.fft.fft(y_lfilter))

plt.figure()
plt.plot(freq[:len(freq)//2], 20 * np.log10(fft_signal[:len(freq)//2]), label='Oryginał')
plt.plot(freq[:len(freq)//2], 20 * np.log10(fft_manual[:len(freq)//2]), label='Filtr własny')
plt.plot(freq[:len(freq)//2], 20 * np.log10(fft_lfilter[:len(freq)//2]), label='Filtr scipy', linestyle='--')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.legend()
plt.grid()
plt.show()
error = np.max(np.abs(y_manual - y_lfilter))
print(f'Maksymalny błąd między implementacjami: {error}')