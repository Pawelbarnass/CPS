import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.fftpack import dct, idct
import sounddevice as sd
import time
import os

# Parametry
fs = 8000  # Częstotliwość próbkowania [Hz]
duration = 4  # Czas nagrania [s]

# Ścieżka do pliku z nagraniem
filename = "mowa_nagranie.wav"

# Część 1: Nagrywanie głosu (jeśli plik nie istnieje)
if not os.path.exists(filename):
    print(f"Nagrywanie {duration} sekund mowy. Proszę mówić do mikrofonu...")
    try:
        # Nagrywanie dźwięku
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
        sd.wait()  # Poczekaj na zakończenie nagrywania
        recording = recording.flatten()
        
        # Normalizacja nagrania
        recording = recording / np.max(np.abs(recording))
        
        # Zapisanie nagrania do pliku WAV
        wavfile.write(filename, fs, recording.astype(np.float32))
        print(f"Nagranie zapisane do pliku {filename}")
        x = recording
        
    except Exception as e:
        print(f"Błąd podczas nagrywania: {e}")
        print("Generowanie przykładowego sygnału mowy...")
        # Generowanie przykładowego sygnału (jeśli nagrywanie się nie powiedzie)
        t = np.arange(0, duration, 1/fs)
        x = 0.3 * np.sin(2*np.pi*180*t) * np.sin(2*np.pi*1*t)
        x += 0.2 * np.sin(2*np.pi*300*t) * np.sin(2*np.pi*2*t)
        x += 0.1 * np.sin(2*np.pi*600*t)
        x *= np.exp(-t/2)  # Envelope
        x = x / np.max(np.abs(x))  # Normalizacja
else:
    # Wczytanie istniejącego nagrania
    fs, x = wavfile.read(filename)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    print(f"Wczytano nagranie z pliku {filename}")

# Wyświetlanie sygnału oryginalnego
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(x))/fs, x)
plt.title("Sygnał oryginalny")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Odtworzenie oryginalnego sygnału
print("Odtwarzanie oryginalnego sygnału...")
sd.play(x, fs)
sd.wait()

# Obliczenie DCT całego sygnału
c = dct(x, type=2, norm='ortho')

# Wyświetlenie współczynników DCT
plt.figure(figsize=(12, 4))
plt.stem(np.arange(len(c))/len(c)*fs/2, c, markerfmt=" ")
plt.title("Współczynniki DCT")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Część 2: Rekonstrukcja z 25% pierwszych współczynników
percent_25 = int(len(c) * 0.25)
c_25 = np.copy(c)
c_25[percent_25:] = 0  # Zerowanie 75% współczynników (wszystkie oprócz pierwszych 25%)

# Rekonstrukcja sygnału
x_25 = idct(c_25, type=2, norm='ortho')

# Wyświetlanie rekonstruowanego sygnału
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(x_25))/fs, x_25)
plt.title("Sygnał zrekonstruowany z 25% pierwszych współczynników DCT")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Odtworzenie rekonstruowanego sygnału
print("Odtwarzanie sygnału zrekonstruowanego z 25% pierwszych współczynników...")
sd.play(x_25, fs)
sd.wait()

# Część 3: Rekonstrukcja z 75% ostatnich współczynników
c_75_last = np.copy(c)
c_75_last[:percent_25] = 0  # Zerowanie pierwszych 25% współczynników

# Rekonstrukcja sygnału
x_75_last = idct(c_75_last, type=2, norm='ortho')

# Wyświetlanie rekonstruowanego sygnału
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(x_75_last))/fs, x_75_last)
plt.title("Sygnał zrekonstruowany z 75% ostatnich współczynników DCT")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Odtworzenie rekonstruowanego sygnału
print("Odtwarzanie sygnału zrekonstruowanego z 75% ostatnich współczynników...")
sd.play(x_75_last, fs)
sd.wait()

# Część 4: Dodanie zakłócenia sinusoidalnego
t = np.arange(len(x))/fs
x_noisy = x + 0.5 * np.sin(2*np.pi*250*t)

# Wyświetlanie sygnału z zakłóceniem
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(x_noisy))/fs, x_noisy)
plt.title("Sygnał z zakłóceniem sinusoidalnym 250 Hz")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Odtworzenie sygnału z zakłóceniem
print("Odtwarzanie sygnału z zakłóceniem...")
sd.play(x_noisy, fs)
sd.wait()

# DCT sygnału z zakłóceniem
c_noisy = dct(x_noisy, type=2, norm='ortho')

# Wyświetlenie współczynników DCT sygnału z zakłóceniem
plt.figure(figsize=(12, 4))
plt.stem(np.arange(len(c_noisy))/len(c_noisy)*fs/2, c_noisy, markerfmt=" ")
plt.title("Współczynniki DCT sygnału z zakłóceniem")
plt.xlabel("Częstotliwość [Hz]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Usunięcie zakłócenia przez wyzerowanie współczynników w okolicy 250 Hz
# Obliczenie indeksu odpowiadającego częstotliwości 250 Hz
idx_250Hz = int(250 / (fs/2) * len(c_noisy))
margin = 25  # Margines wokół indeksu (do dostosowania)

c_filtered = np.copy(c_noisy)
c_filtered[idx_250Hz-margin:idx_250Hz+margin+1] = 0

# Rekonstrukcja sygnału bez zakłócenia
x_filtered = idct(c_filtered, type=2, norm='ortho')

# Wyświetlanie odfiltrowanego sygnału
plt.figure(figsize=(12, 4))
plt.plot(np.arange(len(x_filtered))/fs, x_filtered)
plt.title("Sygnał po usunięciu zakłócenia 250 Hz")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.grid(True)
plt.tight_layout()
plt.show()

# Odtworzenie odfiltrowanego sygnału
print("Odtwarzanie sygnału po usunięciu zakłócenia...")
sd.play(x_filtered, fs)
sd.wait()

# Porównanie wszystkich sygnałów
plt.figure(figsize=(12, 10))

plt.subplot(5, 1, 1)
plt.plot(np.arange(len(x))/fs, x)
plt.title("Sygnał oryginalny")
plt.grid(True)

plt.subplot(5, 1, 2)
plt.plot(np.arange(len(x_25))/fs, x_25)
plt.title("25% pierwszych współczynników DCT")
plt.grid(True)

plt.subplot(5, 1, 3)
plt.plot(np.arange(len(x_75_last))/fs, x_75_last)
plt.title("75% ostatnich współczynników DCT")
plt.grid(True)

plt.subplot(5, 1, 4)
plt.plot(np.arange(len(x_noisy))/fs, x_noisy)
plt.title("Sygnał z zakłóceniem 250 Hz")
plt.grid(True)

plt.subplot(5, 1, 5)
plt.plot(np.arange(len(x_filtered))/fs, x_filtered)
plt.title("Sygnał po filtracji zakłócenia")
plt.xlabel("Czas [s]")
plt.grid(True)

plt.tight_layout()
plt.show()

print("Analiza zakończona.")