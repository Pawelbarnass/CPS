import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parameters from task 1.A
amplitude = 230  # V
frequency = 50   # Hz
duration = 0.1   # seconds

# Define sampling frequencies
fs_high = 10000  # 10 kHz "pseudo-analog" sampling rate
fs_low = 200     # 200 Hz sampling rate for reconstruction

# Generate time vectors
t_high = np.arange(0, duration, 1/fs_high)
t_low = np.arange(0, duration, 1/fs_low)

# Generate the original high-resolution signal ("pseudo analog")
y_high = amplitude * np.sin(2 * np.pi * frequency * t_high)

# Generate the low-resolution sampled signal
y_low = amplitude * np.sin(2 * np.pi * frequency * t_low)

# Function to compute the normalized sinc function
def normalized_sinc(x):
    # Handle the special case where x=0
    if isinstance(x, np.ndarray):
        result = np.ones_like(x)
        mask = x != 0
        result[mask] = np.sin(np.pi * x[mask]) / (np.pi * x[mask])
        return result
    else:
        return 1.0 if x == 0 else np.sin(np.pi * x) / (np.pi * x)

# Reconstruct the signal using sinc interpolation
def reconstruct_signal(samples, sample_times, reconstruction_times):
    T = sample_times[1] - sample_times[0]  # Sampling period
    result = np.zeros_like(reconstruction_times, dtype=float)
    
    for i, t in enumerate(reconstruction_times):
        for n, t_n in enumerate(sample_times):
            result[i] += samples[n] * normalized_sinc((t - t_n) / T)
            
    return result

# Reconstruct the signal
y_reconstructed = reconstruct_signal(y_low, t_low, t_high)

# Calculate reconstruction error
error = y_high - y_reconstructed

# Plot the results
plt.figure(figsize=(12, 10))

# Plot original and reconstructed signals
plt.subplot(2, 1, 1)
plt.plot(t_high, y_high, 'b-', label='Sygnał "pseudo analogowy" (10 kHz)', alpha=0.7)
plt.plot(t_high, y_reconstructed, 'g-', label='Sygnał zrekonstruowany', alpha=0.7)
plt.plot(t_low, y_low, 'ro', label='Próbki sygnału (200 Hz)')
plt.title('Rekonstrukcja sygnału za pomocą splotu z funkcją sinc')
plt.xlabel('Czas (s)')
plt.ylabel('Amplituda (V)')
plt.legend()
plt.grid(True)

# Plot reconstruction error
plt.subplot(2, 1, 2)
plt.plot(t_high, error, 'r-')
plt.title('Błąd rekonstrukcji')
plt.xlabel('Czas (s)')
plt.ylabel('Błąd (V)')
plt.grid(True)

plt.tight_layout()
plt.show()

# Display some statistics about the error
print(f"Maximum absolute error: {np.max(np.abs(error)):.6f} V")
print(f"Mean absolute error: {np.mean(np.abs(error)):.6f} V")
print(f"Root mean square error: {np.sqrt(np.mean(error**2)):.6f} V")