import numpy as np
import matplotlib.pyplot as plt

# 1. Definicja zer i biegunów
zeros = [1j*5, -1j*5, 1j*15, -1j*15]  # Zera transmitancji
poles = [-0.5 + 9.5j, -0.5 - 9.5j,     # Bieguny transmitancji
         -1 + 10j, -1 - 10j,
         -0.5 + 10.5j, -0.5 - 10.5j]

# 2. Generowanie współczynników wielomianów
num = np.poly(zeros)  # Współczynniki licznika
den = np.poly(poles)   # Współczynniki mianownika

# 3. Charakterystyka częstotliwościowa
omega = np.linspace(0, 20, 1000)  # Zakres pulsacji
s = 1j * omega                    # Zmienna s = jω
H = np.polyval(num, s) / np.polyval(den, s)

# 4. Wykresy
plt.figure(figsize=(15, 10))

# a) Zera i bieguny
plt.subplot(2, 2, 1)
plt.scatter(np.real(poles), np.imag(poles), marker='*', color='red', label='Bieguny')
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', color='blue', label='Zera')
plt.title('Płaszczyzna zespolona')
plt.xlabel('Re')
plt.ylabel('Im (jω)')
plt.grid(True)
plt.legend()

# b) Charakterystyka amplitudowa
plt.subplot(2, 2, 2)
plt.plot(omega, np.abs(H), label='|H(jω)|')
plt.plot(omega, 20 * np.log10(np.abs(H)), label='20log|H(jω)|')
plt.title('Charakterystyka amplitudowa')
plt.xlabel('ω [rad/s]')
plt.grid(True)
plt.legend()

# c) Charakterystyka fazowa
plt.subplot(2, 2, 3)
plt.plot(omega, np.angle(H, deg=True))
plt.title('Charakterystyka fazowa')
plt.xlabel('ω [rad/s]')
plt.ylabel('Faza [°]')
plt.grid(True)

plt.tight_layout()
plt.show()

# 5. Analiza
peak_gain = np.max(np.abs(H))
print(f"Maksymalne wzmocnienie: {peak_gain:.2f}")

# 6. Normalizacja wzmocnienia
H_normalized = H / peak_gain  # Normalizacja do wzmocnienia 1

# 7. Sprawdzenie w paśmie przepustowym (8-12 rad/s)
passband = (omega >= 8) & (omega <= 12)
min_attenuation = np.min(20 * np.log10(np.abs(H_normalized[passband])))
print(f"Minimalne tłumienie w paśmie: {min_attenuation:.2f} dB")
passband = (omega >= 8) & (omega <= 12)
phase_passband = np.angle(H[passband], deg=True)
omega_passband = omega[passband]

plt.figure()
plt.plot(omega_passband, phase_passband, 'r-')
plt.title('Faza w paśmie przepustowym')
plt.xlabel('ω [rad/s]')
plt.ylabel('Faza [°]')
plt.grid(True)
plt.show()