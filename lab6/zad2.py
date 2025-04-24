import numpy as np
from scipy.io import wavfile

# Parametry DTMF
DTMF_FREQS = {
    '1': (697, 1209), '2': (697, 1336), '3': (697, 1477),
    '4': (770, 1209), '5': (770, 1336), '6': (770, 1477),
    '7': (852, 1209), '8': (852, 1336), '9': (852, 1477),
    '*': (941, 1209), '0': (941, 1336), '#': (941, 1477)
}

# Częstotliwość próbkowania i parametry
FS = 16000  # Hz
SEGMENT_LENGTH = int(0.5 * FS)  # 50 ms = 800 próbek
TOLERANCE = 20  # Hz

def goertzel(signal, target_freq, fs):
    """Implementacja algorytmu Goertzla do detekcji częstotliwości."""
    N = len(signal)
    k = int(0.5 + (N * target_freq) / fs)
    omega = 2 * np.pi * k / N
    coeff = 2 * np.cos(omega)
    
    q0 = q1 = q2 = 0
    for sample in signal:
        q0 = coeff * q1 - q2 + sample
        q2 = q1
        q1 = q0
    
    return q1**2 + q2**2 - q1*q2*coeff

def decode_dtmf(wav_file):
    fs, audio = wavfile.read(wav_file)
    audio = audio.astype(np.float32)
    
    decoded = []
    prev_char = None
    energy_threshold = 1e6  # Nowy próg energii (dostosuj w zależności od sygnału)
    
    for i in range(len(audio) // SEGMENT_LENGTH):
        start = i * SEGMENT_LENGTH
        end = start + SEGMENT_LENGTH
        segment = audio[start:end]
        
        energies = {}
        for freq in [697, 770, 852, 941, 1209, 1336, 1477]:
            energies[freq] = goertzel(segment, freq, FS)
        
        # Filtruj słabe sygnały
        valid_energies = {f: e for f, e in energies.items() if e > energy_threshold}
        
        low, high = None, None
        for freq in sorted(valid_energies, key=lambda x: valid_energies[x], reverse=True):
            if freq in [697, 770, 852, 941] and not low:
                low = freq
            elif freq in [1209, 1336, 1477] and not high:
                high = freq
        
        current_char = None
        if low and high:
            for char, (f_low, f_high) in DTMF_FREQS.items():
                # BUG FIX: missing closing parenthesis in the if statement
                if (abs(low - f_low) <= TOLERANCE and abs(high - f_high) <= TOLERANCE):
                    current_char = char
                    break
        
        if current_char and current_char != prev_char:
            decoded.append(current_char)
            prev_char = current_char
    
    return ''.join(decoded)
# Użycie: 
# decoded_sequence = decode_dtmf("s.wav")
# print("Zdekodowana sekwencja:", decoded_sequence)
# Przykładowe użycie
decoded_pin = decode_dtmf("s.wav")
print("Zdekodowany PIN:", decoded_pin)