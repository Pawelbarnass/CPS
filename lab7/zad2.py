import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, kaiserord, spectrogram, lfilter
from scipy.signal import firwin, lfilter, freqz

# Parametry sygnału
fs = 48000  # Częstotliwość próbkowania [Hz]
czas_trwania = 1.0  # Długość sygnału [s]
t = np.linspace(0, czas_trwania, int(fs * czas_trwania), endpoint=False)

# 1. Projekt filtra mono (L+R) - dolnoprzepustowy
cutoff_lowpass = 15000  # Częstotliwość odcięcia [Hz]
stopband_lowpass = 19000
transition_width_lowpass = stopband_lowpass - cutoff_lowpass

# Zwiększamy tłumienie w paśmie zaporowym do 60 dB
ripple_db_lowpass = 5000

# Obliczanie parametrów okna Kaisera
width_norm_lowpass = transition_width_lowpass / (0.5 * fs)
N_lowpass, beta_lowpass = kaiserord(ripple_db_lowpass, width_norm_lowpass)
N_lowpass = N_lowpass + 1 if N_lowpass % 2 == 0 else N_lowpass  # Nieparzyste N

taps_lowpass = firwin(N_lowpass, cutoff_lowpass, window=('kaiser', beta_lowpass), fs=fs)
# 2. Projekt filtra pilota 19 kHz - pasmowoprzepustowy
passband_center = 19000  # Częstotliwość pilota
passband_width = 200  # Szerokość pasma [Hz]
transition_width_pilot = 100  # Szerokość przejścia [Hz]

# Obliczanie parametrów okna Kaisera
width_norm_pilot = transition_width_pilot / (0.5 * fs)
N_pilot, beta_pilot = kaiserord(60, width_norm_pilot)  # Większe tłumienie
N_pilot = N_pilot + 1 if N_pilot % 2 == 0 else N_pilot

taps_bandpass = firwin(N_pilot, [passband_center - passband_width/2, passband_center + passband_width/2],
                       window=('kaiser', beta_pilot), pass_zero=False, fs=fs)

# Generowanie sygnału testowego
f_pilot = 19000  # Pilot
f_interferencja = 20000  # Sygnał zakłócający
# Generowanie szumu białego
np.random.seed(0)
szum = np.random.randn(len(t))

# Filtr pasmowoprzepustowy 30 Hz – 15 kHz (np. rząd 801, okno Kaisera)
taps_band = firwin(801, [30, 15000], pass_zero=False, fs=fs, window=('kaiser', 8.6))
sygnal_mono = lfilter(taps_band, 1.0, szum)

# Sygnał testowy: mono + pilot + zakłócenie
sygnal = (sygnal_mono +
          0.5 * np.sin(2 * np.pi * f_pilot * t) +
          0.3 * np.sin(2 * np.pi * f_interferencja * t))

# Filtracja sygnału
# Filtr mono
przefiltrowany_mono = lfilter(taps_lowpass, 1.0, sygnal)

# Filtr pilota z korekcją opóźnienia
przefiltrowany_pilot = lfilter(taps_bandpass, 1.0, sygnal)
opoznienie = (N_pilot - 1) // 2
przefiltrowany_pilot = przefiltrowany_pilot[opoznienie:]
t_pilot = t[:len(przefiltrowany_pilot)]

# Analiza widmowa
def rysuj_widmo(sygnal, tytul, fs):
    plt.figure(figsize=(12, 4))
    n = len(sygnal)
    frekw = np.fft.rfftfreq(n, 1/fs)
    widmo = np.abs(np.fft.rfft(sygnal))
    plt.plot(frekw, 20 * np.log10(widmo + 1e-10))
    plt.title(tytul)
    plt.xlim(0, 24000)
    plt.ylim(-60, 60)
    plt.xlabel('Częstotliwość [Hz]')
    plt.ylabel('Amplituda [dB]')
    plt.grid(True)

rysuj_widmo(sygnal, 'Widmo sygnału oryginalnego', fs)
rysuj_widmo(przefiltrowany_mono, 'Widmo po filtrze mono', fs)
rysuj_widmo(przefiltrowany_pilot, 'Widmo po filtrze pilota', fs)

# Analiza czasowo-częstotliwościowa (spektrogram)
def rysuj_spektrogram(sygnal, tytul, fs):
    f, t_spec, Sxx = spectrogram(sygnal, fs, nperseg=1024, noverlap=512)
    plt.figure(figsize=(12, 4))
    plt.pcolormesh(t_spec, f, 10 * np.log10(Sxx + 1e-10), shading='auto')
    plt.title(tytul)
    plt.ylim(18000, 20000)
    plt.xlabel('Czas [s]')
    plt.ylabel('Częstotliwość [Hz]')
    plt.colorbar(label='Gęstość mocy [dB]')

rysuj_spektrogram(przefiltrowany_pilot, 'Spektrogram sygnału pilota', fs)
plt.show()
w, h = freqz(taps_lowpass, worN=2048, fs=fs)
plt.figure(figsize=(10, 5))
plt.plot(w, 20 * np.log10(np.abs(h) + 1e-12))
plt.title('Charakterystyka amplitudowa filtru mono (dolnoprzepustowego)')
plt.xlabel('Częstotliwość [Hz]')
plt.ylabel('Amplituda [dB]')
plt.xlim(0, fs/2)
plt.ylim(-100, 5)
plt.grid(True)
plt.tight_layout()
plt.show()