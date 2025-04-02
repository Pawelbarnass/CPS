import numpy as np
import matplotlib.pyplot as plt

N = 100
fs = 1000
f1, f2 = 100, 200
A1, A2 = 100, 200
phi1, phi2 = np.pi / 7, np.pi / 11

t = np.linspace(0, (N-1)/fs, N)
x = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)

k = np.arange(N).reshape((N, 1))
n = np.arange(N)
A = np.exp(-1j * 2 * np.pi * k * n / N) / np.sqrt(N)

X = A @ x
freqs = np.arange(N) * fs / N

plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.stem(freqs, X.real)
plt.title("Część rzeczywista")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 2)
plt.stem(freqs, X.imag)
plt.title("Część urojona")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 3)
plt.stem(freqs, np.abs(X))
plt.title("Moduł")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")

plt.subplot(2, 2, 4)
plt.stem(freqs, np.angle(X))
plt.title("Faza")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Kąt [rad]")

plt.tight_layout()
plt.show()

B = np.conj(A.T)
x_r = B @ X

print("Czy x == xr?", np.allclose(x, x_r))

X_fft = np.fft.fft(x) / np.sqrt(N)
x_r_fft = np.fft.ifft(X_fft) * np.sqrt(N)

print("X - FFT == DFT", np.allclose(X, X_fft))
print("xr - FFT == DFT", np.allclose(x_r, x_r_fft))

print("X == X_fft?", np.allclose(X, X_fft))
print("xr == x_r_fft?", np.allclose(x_r, x_r_fft))

f1 = 125
x_new = A1 * np.cos(2 * np.pi * f1 * t + phi1) + A2 * np.cos(2 * np.pi * f2 * t + phi2)
X_new = A @ x_new

plt.figure(figsize=(12, 6))
plt.stem(freqs, np.abs(X_new))
plt.title("Widmo amplitudowe dla f1 = 125 Hz")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")
plt.show()

plt.figure(figsize=(12, 8))

# Plots dla części rzeczywistej, urojonej i modułu pozostają bez zmian...

# Lepszy sposób wyświetlania fazy:
plt.subplot(2, 2, 4)
amplitude_threshold = 0.05 * np.max(np.abs(X))  # Próg amplitudy (5% maksymalnej)

# Maska do filtrowania nieistotnych wartości fazy
mask = np.abs(X) > amplitude_threshold

# Przygotuj dane do wyświetlenia (tylko znaczące fazy)
phases = np.angle(X)
freqs_masked = freqs[mask]
phases_masked = phases[mask]

# Wariant 1: Punkty tylko dla znaczących wartości
plt.stem(freqs_masked, phases_masked, linefmt='C0-', markerfmt='C0o')

# Wariant 2: Wszystkie punkty, ale nieznaczące są przyciemnione (opcjonalnie)
plt.stem(freqs, phases, linefmt='gray', markerfmt='gray.', alpha=0.3)
plt.stem(freqs_masked, phases_masked, linefmt='C0-', markerfmt='C0o')

# Dodaj linie pomocnicze na poziomie ±π dla odniesienia
plt.axhline(y=np.pi, color='r', linestyle='--', alpha=0.3)
plt.axhline(y=-np.pi, color='r', linestyle='--', alpha=0.3)

plt.title("Faza (tylko dla znaczących amplitud)")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Faza [rad]")
plt.ylim([-np.pi-0.5, np.pi+0.5])  # Nieco większy zakres dla lepszej czytelności

# Dodatkowe wyświetlanie fazy w stopniach na drugiej osi Y
ax2 = plt.gca().twinx()
ax2.set_ylim([-np.pi-0.5, np.pi+0.5])
ax2.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
ax2.set_yticklabels(['-180°', '-90°', '0°', '90°', '180°'])
ax2.set_ylabel("Faza [stopnie]")

plt.tight_layout()
plt.show()
