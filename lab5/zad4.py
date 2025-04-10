import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import buttap, cheb1ap, cheb2ap, ellipap, freqs, lp2lp, lp2hp, lp2bp, lp2bs, tf2zpk, zpk2tf

# Parametry projektu
N = 8               # Rząd filtra prototypowego
f0 = 100            # Częstotliwość graniczna dla LP/HP [Hz]
f1, f2 = 10, 100    # Pasmo dla BP/BS [Hz]
Rp = 3              # Tłumienie w paśmie przepustowym [dB] (Czebyszew I/eliptyczny)
Rs = 100            # Tłumienie w paśmie zaporowym [dB] (Czebyszew II/eliptyczny)

# Lista typów filtrów prototypowych
prototype_filters = {
    'Butterworth': buttap,
    'Czebyszew I': lambda N: cheb1ap(N, Rp),
    'Czebyszew II': lambda N: cheb2ap(N, Rs),
    'Eliptyczny': lambda N: ellipap(N, Rp, Rs)
}

# Transformacje częstotliwościowe
transforms = {
    'LowPass': lambda b, a: lp2lp(b, a, 2*np.pi*f0),
    'HighPass': lambda b, a: lp2hp(b, a, 2*np.pi*f0),
    'BandPass': lambda b, a: lp2bp(b, a, 2*np.pi*np.sqrt(f1*f2), 2*np.pi*(f2 - f1)),
    'BandStop': lambda b, a: lp2bs(b, a, 2*np.pi*np.sqrt(f1*f2), 2*np.pi*(f2 - f1))
}

# Zakres częstotliwości
f = np.logspace(-1, 3, 1000)  # 0.1 Hz do 1000 Hz
w = 2 * np.pi * f             # Pulsacja [rad/s]

for filter_name, prototype_fn in prototype_filters.items():
    # Projektuj prototypowy filtr dolnoprzepustowy
    z_proto, p_proto, k_proto = prototype_fn(N)
    b_proto, a_proto = zpk2tf(z_proto, p_proto, k_proto)
    
    # Charakterystyka przed transformacją
    w_proto, H_proto = freqs(b_proto, a_proto, worN=w)
    
    for transform_name, transform_fn in transforms.items():
        # Transformacja filtra
        b_trans, a_trans = transform_fn(b_proto, a_proto)
        
        # Charakterystyka po transformacji
        w_trans, H_trans = freqs(b_trans, a_trans, worN=w)
        
        # Wykres charakterystyki amplitudowej
        plt.figure(figsize=(12, 6))
        plt.semilogx(f, 20 * np.log10(np.abs(H_proto)), 'r--', label='Prototyp LP')
        plt.semilogx(f, 20 * np.log10(np.abs(H_trans)), 'b-', label=f'{transform_name}')
        plt.title(f'{filter_name} → {transform_name}')
        plt.xlabel('Częstotliwość [Hz]')
        plt.ylabel('Tłumienie [dB]')
        plt.grid(True, which='both', linestyle='--')
        plt.legend()
        plt.show()

        # Rozkład biegunów i zer
        z_trans, p_trans, _ = tf2zpk(b_trans, a_trans)
        plt.figure(figsize=(8, 8))
        plt.scatter(np.real(z_trans), np.imag(z_trans), marker='o', facecolors='none', edgecolors='r', label='Zera')
        plt.scatter(np.real(p_trans), np.imag(p_trans), marker='x', color='b', label='Bieguny')
        plt.title(f'Bieguny i zera: {filter_name} → {transform_name}')
        plt.xlabel('Część rzeczywista')
        plt.ylabel('Część urojona')
        plt.grid(True)
        plt.axis('equal')
        plt.legend()
        plt.show()

# Funkcja pomocnicza do konwersji zer/biegunów na współczynniki TF
def zpk2tf(z, p, k):
    b = np.poly(z) * k
    a = np.poly(p)
    return b, a