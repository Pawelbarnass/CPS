import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import ellip, freqs

def projektuj_filtr(case, Rp=3, Rs=40, N_start=4):
    # Parametry w zależności od przypadku
    if case == "test":
        center = 96e6      # 96 MHz
        delta = 1e6        # ±1 MHz
        stop_delta = 1e6   # odstęp od pasma przepustowego
    elif case == "docelowy":
        center = 96e6      # 96 MHz
        delta = 0.1e6      # ±100 kHz
        stop_delta = 0.1e6 # odstęp od pasma przepustowego
    
    # Częstotliwości graniczne
    passband = [center - delta, center + delta]
    stopband = [passband[0] - stop_delta, passband[1] + stop_delta]
    
    # Konwersja na rad/s
    Wp = np.array(passband) * 2 * np.pi
    Ws = np.array(stopband) * 2 * np.pi

    # Iteracyjne zwiększanie rzędu filtra
    for N in range(N_start, 15):
        # Projektuj filtr eliptyczny
        b, a = ellip(N, Rp, Rs, Wp, btype='bandpass', analog=True)
        
        # Oblicz charakterystykę
        f = np.linspace(passband[0]-2*stop_delta, passband[1]+2*stop_delta, 10000)
        w = 2 * np.pi * f
        _, H = freqs(b, a, worN=w)
        H_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))
        
        # Sprawdź tłumienie w pasmie zaporowym
        mask_stop_low = (f < stopband[0])
        mask_stop_high = (f > stopband[1])
        attenuation_low = np.max(H_db[mask_stop_low])
        attenuation_high = np.max(H_db[mask_stop_high])

        mask_pass = (f >= passband[0]) & (f <= passband[1])
        ripple_pass = np.max(np.abs(H_db[mask_pass]))
        
        if (attenuation_low <= -Rs) and (attenuation_high <= -Rs) and (ripple_pass <= Rp):
            print(f"Case {case}: Użyty rząd filtra N = {N}")
            break
    else:
        print("Nie udało się spełnić wymagań nawet dla N=15!")
        return

    # Wykres charakterystyki
    plt.figure(figsize=(12, 6))
    plt.plot(f, H_db, label='Odpowiedź filtra')
    plt.axvspan(passband[0], passband[1], color='0.9', label='Pasmo przepustowe')
    plt.axvline(stopband[0], c='r', ls='--', label='Granice pasma zaporowego')
    plt.axvline(stopband[1], c='r', ls='--')
    plt.axhline(-Rp, c='g', ls=':', label='-3 dB')
    plt.axhline(-Rs, c='m', ls=':', label='-40 dB')
    plt.xlim(passband[0]-2*stop_delta, passband[1]+2*stop_delta)
    #plt.ylim(-Rs-10, 3)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Tłumienie [dB]')
    plt.title(f'Charakterystyka filtra ({case})')
    plt.legend()
    plt.grid(True)
    plt.show()

# Projektuj filtry
projektuj_filtr("test", N_start=4)
projektuj_filtr("docelowy", N_start=14)