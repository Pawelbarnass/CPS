import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import butter, lfilter, freqz, stft, tf2zpk
import sounddevice as sd
# Wczytanie plików audio
fs_speech, speech = wavfile.read('C:/Users/wiedzmok/CPS/beach_and_seagulls.wav')
fs_bird, bird = wavfile.read('C:/Users/wiedzmok/CPS/cnt109.wav')

# Konwersja do mono i dopasowanie długości
def prepare_signal(signal):
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    return signal / np.max(np.abs(signal))

speech = prepare_signal(speech)
bird = prepare_signal(bird)

max_len = max(len(speech), len(bird))
speech = np.pad(speech, (0, max_len - len(speech)))
bird = np.pad(bird, (0, max_len - len(bird)))

# Mieszanie sygnałów
mix = (speech + bird) / 2
def plot_fft(signal, fs, title):
    n = len(signal)
    freq = np.fft.rfftfreq(n, 1/fs)
    fft = np.abs(np.fft.rfft(signal))
    plt.plot(freq, 20*np.log10(fft))
    plt.title(title)
    plt.xlabel('Częstotliwość (Hz)')
    plt.ylabel('Amplituda (dB)')

plt.figure(figsize=(12, 8))
plt.subplot(311); plot_fft(speech, fs_speech, 'Mowa')
plt.subplot(312); plot_fft(bird, fs_speech, 'Ptak')
plt.subplot(313); plot_fft(mix, fs_speech, 'Mieszanka')
plt.tight_layout()
plt.show()

def plot_spectrogram(signal, fs, title):
    f, t, Zxx = stft(signal, fs, nperseg=1024)
    plt.pcolormesh(t, f, 20*np.log10(np.abs(Zxx) + 1e-10), shading='gouraud')
    plt.title(title)
    plt.ylabel('Częstotliwość (Hz)')
    plt.xlabel('Czas (s)')

print("fs_speech =", fs_speech)  # Debug print

# Parametry filtru
cutoff = min(1000, int(0.45 * fs_speech))  # Ensure cutoff < Nyquist
order = 9     # Rząd filtru

# Projektowanie filtru Butterworth
nyquist = 0.5 * fs_speech
normal_cutoff = cutoff / nyquist
b, a = butter(order, normal_cutoff, btype='low')
plt.figure(figsize=(12, 8))
plt.subplot(311); plot_spectrogram(speech, fs_speech, 'Mowa')
plt.subplot(312); plot_spectrogram(bird, fs_speech, 'Ptak')
plt.subplot(313); plot_spectrogram(mix, fs_speech, 'Mieszanka')
plt.tight_layout()
plt.show()

# Charakterystyka częstotliwościowa
w, h = freqz(b, a, worN=2000)
plt.plot((w/np.pi) * nyquist, 20*np.log10(np.abs(h)))
plt.title('Charakterystyka filtru')
plt.xlabel('Częstotliwość (Hz)')
plt.ylabel('Wzmocnienie (dB)')
plt.grid()
plt.show()

# Wykres biegunów i zer
zeros, poles, _ = tf2zpk(b, a)
plt.scatter(np.real(zeros), np.imag(zeros), marker='o', label='Zera')
plt.scatter(np.real(poles), np.imag(poles), marker='x', label='Bieguny')
plt.title('Płaszczyzna Z')
plt.xlabel('Część rzeczywista')
plt.ylabel('Część urojona')
plt.grid()
plt.legend()
plt.show()
# Filtracja sygnału
filtered = lfilter(b, a, mix)

# Wizualizacja wyników
plt.figure(figsize=(12, 4))
plt.subplot(121); plot_fft(filtered, fs_speech, 'Przefiltrowany sygnał')
plt.subplot(122); plot_spectrogram(filtered, fs_speech, 'Przefiltrowany sygnał')
plt.tight_layout()
plt.show()

# Odtwarzanie dźwięku
print("Odtwarzanie oryginalnej mieszanki...")
sd.play(mix, fs_speech)
sd.wait()
print("Odtwarzanie przefiltrowanego sygnału...")
sd.play(filtered, fs_speech)
sd.wait()