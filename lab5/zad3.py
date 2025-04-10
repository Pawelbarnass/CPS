import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellipord, ellip, freqs, tf2zpk

# Parametry projektu
fs = 256e3  # Częstotliwość próbkowania [Hz]
f3dB = 64e3  # Częstotliwość odcięcia [Hz]
fstop = 128e3  # Częstotliwość zaporowa [Hz]
Rp = 3       # Tłumienie w paśmie przepustowym [dB]
Rs = 40      # Tłumienie w paśmie zaporowym [dB]

# Konwersja na rad/s (wymagane przez funkcje projektowe)
wp = 2 * np.pi * f3dB
ws = 2 * np.pi * fstop

# Projektowanie filtra
n, Wn = ellipord(wp, ws, Rp, Rs, analog=True)
b, a = ellip(n, Rp, Rs, Wn, btype='low', analog=True)

# Obliczenie biegunów i zer
z, p, k = tf2zpk(b, a)

# Charakterystyka częstotliwościowa
w = np.logspace(np.log10(2 * np.pi * 1e3), np.log10(2 * np.pi * 300e3), 1000)
w, H = freqs(b, a, worN=w)
f = w / (2 * np.pi)  # Konwersja rad/s → Hz

# Przygotowanie wykresów
plt.figure(figsize=(12, 8))

# Charakterystyka amplitudowa
plt.subplot(2, 1, 1)
plt.semilogx(f, 20 * np.log10(np.abs(H)))
plt.title('Charakterystyka amplitudowa')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Tłumienie [dB]')
plt.grid(True, which='both', linestyle='--')
plt.axvline(f3dB, color='r', linestyle='--', label='f3dB = 64 kHz')
plt.axvline(fstop, color='g', linestyle='--', label='fs/2 = 128 kHz')
plt.axhline(-3, color='r', linestyle=':', label='-3 dB')
plt.axhline(-40, color='g', linestyle=':', label='-40 dB')
plt.xlim(1e3, 300e3)
plt.ylim(-60, 3)
plt.legend()

# Rozkład biegunów i zer
plt.subplot(2, 1, 2)
plt.scatter(np.real(p), np.imag(p), marker='x', color='b', label='Bieguny')
plt.scatter(np.real(z), np.imag(z), marker='o', color='r', label='Zera', facecolors='none')
plt.title('Rozkład biegunów i zer')
plt.xlabel('Część rzeczywista')
plt.ylabel('Część urojona')
plt.grid(True)
plt.axis('equal')
plt.legend()

plt.tight_layout()
plt.show()

# Wyświetlenie parametrów filtra
print(f'Rząd filtra: n = {n}')
print(f'Częstotliwość graniczna: {Wn/(2*np.pi):.1f} Hz')