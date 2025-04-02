import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

fs = 10000   
fn = 50      
fm = 1       
df = 5       
duration = 1  


t = np.arange(0, duration, 1/fs)

modulating_signal = np.sin(2 * np.pi * fm * t)
beta = df / fm
modulated_signal = np.sin(2 * np.pi * fn * t + beta * np.sin(2 * np.pi * fm * t))
plt.figure(figsize=(10, 6))
plt.plot(t, modulated_signal, 'b', label='Sygnał zmodulowany (SFM)')
plt.plot(t, modulating_signal, 'r', label='Sygnał modulujący')
plt.title('Sygnał zmodulowany i modulujący')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

fs_new = 25 
sampling_interval = int(fs / fs_new)
t_sampled = t[::sampling_interval]
modulated_signal_sampled = modulated_signal[::sampling_interval]

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(t, modulated_signal, 'b-', label='Sygnał "analogowy"')
plt.plot(t_sampled, modulated_signal_sampled, 'ro', label=f'Sygnał próbkowany (fs={fs_new} Hz)')
plt.title('Porównanie sygnału oryginalnego i próbkowanego')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda')
plt.legend()
plt.grid(True)

interpolation = interp1d(t_sampled, modulated_signal_sampled, kind='linear', fill_value="extrapolate")
interpolated_signal = interpolation(t)
error = modulated_signal - interpolated_signal

plt.subplot(2, 1, 2)
plt.plot(t, error)
plt.title('Błędy spowodowane próbkowaniem')
plt.xlabel('Czas (s)')
plt.ylabel('Błąd')
plt.grid(True)
plt.tight_layout()
plt.show()

def spectrum(signal_data, sampling_rate):
    """Calculate and return the power density spectrum using Welch's method"""
    f, Pxx = signal.welch(signal_data, sampling_rate, nperseg=min(1024, len(signal_data)))
    return f, Pxx

f_orig, Pxx_orig = spectrum(modulated_signal, fs)

f_sampled, Pxx_sampled = spectrum(modulated_signal_sampled, fs_new)

plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.semilogy(f_orig, Pxx_orig)
plt.title('Widmo gęstości mocy - sygnał oryginalny')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Moc/Częstotliwość (dB/Hz)')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.semilogy(f_sampled, Pxx_sampled)
plt.title('Widmo gęstości mocy - sygnał próbkowany')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Moc/Częstotliwość (dB/Hz)')
plt.grid(True)

plt.tight_layout()
plt.show()