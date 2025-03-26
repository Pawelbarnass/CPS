import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import sounddevice as sd
from scipy import signal

# Load audio files
fs_car, car_signal = wav.read('car.wav')
fs_bird, bird_signal = wav.read('bird.wav')

# Convert to float
car_signal = car_signal.astype(np.float32)
bird_signal = bird_signal.astype(np.float32)

# Resample if sample rates are different
if fs_car != fs_bird:
    # Use car's sample rate as the target
    num_samples = int(len(bird_signal) * fs_car / fs_bird)
    bird_signal = signal.resample(bird_signal, num_samples)
    print(f"Resampled bird signal from {fs_bird}Hz to {fs_car}Hz")
    fs = fs_car
else:
    fs = fs_car

# Adjust signal lengths (pad shorter with zeros)
N = max(len(car_signal), len(bird_signal))
car_signal = np.pad(car_signal, (0, N - len(car_signal)))
bird_signal = np.pad(bird_signal, (0, N - len(bird_signal)))

# Calculate DFT for both signals and exclude DC component
car_fft = fft(car_signal) / N
car_fft[0] = 0  # Zero out DC component

bird_fft = fft(bird_signal) / N
bird_fft[0] = 0  # Zero out DC component

freqs = np.fft.fftfreq(N, d=1/fs)

# Display DFT spectra
plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(car_fft[:N//2]), label='Car (niskie f)', color='r')
plt.xlim(0, 500)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.title('Widmo DFT dźwięku samochodu')
plt.grid()
plt.legend()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(bird_fft[:N//2]), label='Bird (wysokie f)', color='b')
plt.xlim(1000, 3000)
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.title('Widmo DFT dźwięku ptaka')
plt.grid()
plt.legend()
plt.show()

# Sum the signals
combined_signal = car_signal + bird_signal
combined_fft = fft(combined_signal) / N
combined_fft[0] = 0  # Zero out DC component

# Display DFT of sum
plt.figure(figsize=(10, 4))
plt.plot(freqs[:N//2], np.abs(combined_fft[:N//2]), label='Suma (Car + Bird)', color='g')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda')
plt.title('Widmo DFT sumy sygnałów')
plt.grid()
plt.legend()
plt.show()

# Remove low frequencies (car-related)
cutoff = 500  # Hz cutoff
car_indices = np.abs(freqs) < cutoff
filtered_fft = combined_fft.copy()
filtered_fft[car_indices] = 0  # Zero out low frequencies

# IDFT (convert back to signal)
filtered_signal = np.real(ifft(filtered_fft) * N)

# Normalize for playback
normalized_signal = filtered_signal / np.max(np.abs(filtered_signal))
sd.play(normalized_signal, fs)
sd.wait()

# Save filtered audio
wav.write('filtered_bird.wav', fs, (filtered_signal * 32767).astype(np.int16))

# Display filtered signal
plt.figure(figsize=(10, 4))
plt.plot(filtered_signal, label='Po usunięciu niskich f', color='purple')
plt.xlabel('Próbki')
plt.ylabel('Amplituda')
plt.title('Sygnał po filtracji DFT')
plt.grid()
plt.legend()
plt.show()