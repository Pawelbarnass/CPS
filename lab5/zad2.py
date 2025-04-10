import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Parametry projektowe
omega_3dB = 2 * np.pi * 100  # Częstotliwość graniczna [rad/s]
N_values = [2, 4, 6, 8]      # Rzędy filtrów
colors = ['b', 'g', 'r', 'c'] # Kolory dla wykresów

# Generowanie częstotliwości do analizy
f_log = np.logspace(1, 3, 1000)  # 10 Hz - 1000 Hz (skala log)
f_lin = np.linspace(0, 1000, 1000) # 0 Hz - 1000 Hz (skala liniowa)
omega_log = 2 * np.pi * f_log
omega_lin = 2 * np.pi * f_lin

# Inicjalizacja wykresów
plt.figure(figsize=(14, 8))

# Charakterystyki amplitudowe
for i, N in enumerate(N_values):
    # Obliczanie biegunów
    poles = []
    for k in range(1, N+1):
        theta = np.pi/2 + np.pi*(2*k-1)/(2*N)
        pole = omega_3dB * np.exp(1j*theta)
        poles.append(pole)
    
    # Tworzenie transmitancji
    den = np.poly(poles)
    num = np.prod(poles)
    
    # Obliczanie charakterystyki
    H_log = num / np.polyval(den, 1j*omega_log)
    H_lin = num / np.polyval(den, 1j*omega_lin)
    
    # Wykresy
    plt.subplot(2, 1, 1)
    plt.semilogx(f_log, 20*np.log10(np.abs(H_log)), color=colors[i], label=f'N={N}')
    
    plt.subplot(2, 1, 2)
    plt.plot(f_lin, 20*np.log10(np.abs(H_lin)), color=colors[i], label=f'N={N}')

# Formatowanie wykresów amplitudy
plt.subplot(2, 1, 1)
plt.title('Charakterystyka amplitudowa (skala log)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('20log|H| [dB]')
plt.grid(True)
plt.legend()
plt.axvline(100, color='k', linestyle='--')

plt.subplot(2, 1, 2)
plt.title('Charakterystyka amplitudowa (skala liniowa)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('20log|H| [dB]')
plt.grid(True)
plt.legend()
plt.axvline(100, color='k', linestyle='--')
plt.tight_layout()

# Charakterystyki fazowe
plt.figure(figsize=(10, 5))
for i, N in enumerate(N_values):
    # Obliczanie biegunów (ponownie)
    poles = []
    for k in range(1, N+1):
        theta = np.pi/2 + np.pi*(2*k-1)/(2*N)
        pole = omega_3dB * np.exp(1j*theta)
        poles.append(pole)
    
    den = np.poly(poles)
    num = np.prod(poles)
    
    H = num / np.polyval(den, 1j*omega_lin)
    phase = np.angle(H, deg=True)
    plt.plot(f_lin, phase, color=colors[i], label=f'N={N}')

plt.title('Charakterystyka fazowa')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Faza [°]')
plt.grid(True)
plt.legend()

# Odpowiedź impulsowa i skokowa (dla N=4)
N = 4
poles = []
for k in range(1, N+1):
    theta = np.pi/2 + np.pi*(2*k-1)/(2*N)
    pole = omega_3dB * np.exp(1j*theta)
    poles.append(pole)

den = np.poly(poles)
num = np.prod(poles)
sys = signal.lti(num, den)

t_imp, y_imp = signal.impulse(sys)
t_step, y_step = signal.step(sys)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_imp, y_imp)
plt.title('Odpowiedź impulsowa (N=4)')
plt.xlabel('Czas [s]')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(t_step, y_step)
plt.title('Odpowiedź skokowa (N=4)')
plt.xlabel('Czas [s]')
plt.grid(True)
plt.tight_layout()

plt.show()