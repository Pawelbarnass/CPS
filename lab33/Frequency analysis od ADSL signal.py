import scipy.io
import numpy as np
import matplotlib.pyplot as plt

file_path = "lab_03.mat"
mat_data = scipy.io.loadmat(file_path)

student_id = 43258
index = (student_id % 16) + 1
signal_key = f"x_{index}"

signal = mat_data[signal_key].flatten()

K = 8
N = 512
M = 32

total_frame_length = N + M
frames = np.zeros((K, N))
for m in range(K):
    start_idx = m * total_frame_length + M
    frames[m] = signal[start_idx:start_idx + N]

fft_frames = np.fft.fft(frames)

harmonic_indices = []
for i in range(K):
    threshold = 0.1 * np.max(np.abs(fft_frames[i]))
    indices = np.where(np.abs(fft_frames[i]) > threshold)[0]
    harmonic_indices.append(indices)

fig, axes = plt.subplots(K, 1, figsize=(10, 15))
for i in range(K):
    axes[i].plot(np.abs(fft_frames[i]), label=f'Ramka {i+1}')
    axes[i].scatter(harmonic_indices[i], np.abs(fft_frames[i])[harmonic_indices[i]], color='red', label='Używane harmoniczne')
    axes[i].set_ylabel("Amplituda")
    axes[i].legend()
    axes[i].grid()
    print(f"Ramka {i+1}: Używane harmoniczne - {harmonic_indices[i]}")

axes[-1].set_xlabel("Indeks harmonicznej (częstotliwość)")
plt.suptitle("Widmo częstotliwościowe ramek sygnału ADSL")
plt.show()
