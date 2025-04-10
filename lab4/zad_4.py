import numpy as np
import matplotlib.pyplot as plt

# Load the raw signal data
random_signal_path = 'C:\\Users\\wiedzmok\\CPS\\random_signal.txt'
signal = np.loadtxt(random_signal_path)

# Compute FFT in Python
N = len(signal)
python_fft = np.fft.fft(signal)

# Apply the normalization to match C++ implementation
# Both C++ files appear to use normalization by sqrt(N)
python_fft_normalized = python_fft / np.sqrt(N)

# Calculate magnitude and phase
python_magnitude = np.abs(python_fft_normalized)
python_phase = np.angle(python_fft_normalized)

# Load both C++ results
cpp_float = np.loadtxt('C:\\Users\\wiedzmok\\CPS\\lab4\\fft_results_float.dat')
cpp_double = np.loadtxt('C:\\Users\\wiedzmok\\CPS\\lab4\\fft_results.dat')

# Extract magnitude and phase from both C++ results
cpp_float_magnitude = cpp_float[:, 3]
cpp_float_phase = cpp_float[:, 4]
cpp_double_magnitude = cpp_double[:, 3]
cpp_double_phase = cpp_double[:, 4]

# Compare the first few results to verify normalization
print("Python vs C++ Float vs C++ Double magnitude comparison (first 5 elements):")
for i in range(5):
    print(f"Index {i}: Python={python_magnitude[i]:.6f}, C++ Float={cpp_float_magnitude[i]:.6f}, C++ Double={cpp_double_magnitude[i]:.6f}")

# Plot comparison
plt.figure(figsize=(14, 10))

# Magnitude comparison
plt.subplot(2, 1, 1)
plt.plot(python_magnitude[:100], label='Python FFT', alpha=0.7)
plt.plot(cpp_float_magnitude[:100], label='C++ Float FFT', alpha=0.7)
plt.plot(cpp_double_magnitude[:100], label='C++ Double FFT', alpha=0.7)
plt.title('FFT Magnitude Comparison')
plt.xlabel('Frequency Index')
plt.ylabel('Magnitude')
plt.grid(True)
plt.legend()

# Phase comparison
plt.subplot(2, 1, 2)
plt.plot(python_phase[:100], label='Python FFT', alpha=0.7)
plt.plot(cpp_float_phase[:100], label='C++ Float FFT', alpha=0.7) 
plt.plot(cpp_double_phase[:100], label='C++ Double FFT', alpha=0.7)
plt.title('FFT Phase Comparison')
plt.xlabel('Frequency Index')
plt.ylabel('Phase (radians)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

# Calculate differences
min_len = min(len(python_magnitude), len(cpp_float_magnitude))
float_mag_diff = np.abs(python_magnitude[:min_len] - cpp_float_magnitude[:min_len])
float_phase_diff = np.abs(python_phase[:min_len] - cpp_float_phase[:min_len])
float_phase_diff = np.minimum(float_phase_diff, 2*np.pi - float_phase_diff)  # Handle phase wrapping

# Statistics for float comparison
print("\nDifferences between Python and C++ Float:")
print(f"Maximum magnitude difference: {np.max(float_mag_diff):.6e}")
print(f"Average magnitude difference: {np.mean(float_mag_diff):.6e}")
print(f"Maximum phase difference: {np.max(float_phase_diff):.6e} radians")
print(f"Average phase difference: {np.mean(float_phase_diff):.6e} radians")

# Calculate differences for double precision
double_mag_diff = np.abs(python_magnitude[:min_len] - cpp_double_magnitude[:min_len])
double_phase_diff = np.abs(python_phase[:min_len] - cpp_double_phase[:min_len]) 
double_phase_diff = np.minimum(double_phase_diff, 2*np.pi - double_phase_diff)  # Handle phase wrapping

print("\nDifferences between Python and C++ Double:")
print(f"Maximum magnitude difference: {np.max(double_mag_diff):.6e}")
print(f"Average magnitude difference: {np.mean(double_mag_diff):.6e}")
print(f"Maximum phase difference: {np.max(double_phase_diff):.6e} radians")
print(f"Average phase difference: {np.mean(double_phase_diff):.6e} radians")