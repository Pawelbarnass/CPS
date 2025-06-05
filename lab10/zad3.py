import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa # Dla LPC i wczytywania audio (alternatywa dla wav)

# --- Parametry ---
FRAME_SIZE = 400  # Długość ramki w próbkach
HOP_SIZE = 200    # Przesunięcie ramki w próbkach (FRAME_SIZE // 2 dla 50% overlap)
LPC_ORDER = 20    # Rząd filtru LPC
POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE = 7 # Rząd wielomianu do aproksymacji widma resztkowego

# --- Funkcje pomocnicze ---
def detect_voiced_unvoiced(frame, prev_pitch=[0], prev_energy=[0.1], energy_alpha=0.9, corr_thresh=0.3, cepstrum_thresh=0.1):
    # Energia
    energy = np.sum(frame**2) / len(frame)
    prev_energy[0] = energy_alpha * prev_energy[0] + (1 - energy_alpha) * energy
    energy_thresh = 0.5 * prev_energy[0]
    # Autokorelacja
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    corr[0] = 0
    fs = 8000
    min_lag = int(fs/400)
    max_lag = int(fs/60)
    pitch_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch_val = corr[pitch_lag]
    norm_peak = pitch_val / (np.sum(frame**2) + 1e-12)
    # Kepstrum
    spectrum = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(spectrum) + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum).real
    min_pitch = int(fs / 400)
    max_pitch = int(fs / 60)
    cep_peak = np.max(np.abs(cepstrum[min_pitch:max_pitch]))
    # Adaptacja tonu podstawowego
    if prev_pitch[0] > 0:
        if abs(pitch_lag - prev_pitch[0]) > 10:
            pitch_lag = prev_pitch[0]
    prev_pitch[0] = pitch_lag
    # Detekcja
    is_voiced = (energy > energy_thresh) and (norm_peak > corr_thresh) and (cep_peak > cepstrum_thresh)
    return is_voiced, pitch_lag

def load_wav(filename):
    """Wczytuje plik WAV."""
    try:
        sample_rate, signal = wav.read(filename)
        if signal.dtype == np.int16:
            signal = signal / 32768.0  # Normalizacja do zakresu [-1, 1]
        elif signal.dtype == np.int32:
            signal = signal / 2147483648.0
        # Jeśli stereo, bierzemy tylko jeden kanał
        if signal.ndim > 1:
            signal = signal[:, 0]
        return sample_rate, signal
    except Exception as e:
        print(f"Błąd przy wczytywaniu {filename}: {e}")
        # Próba z librosa jako fallback
        try:
            signal, sample_rate = librosa.load(filename, sr=None, mono=True)
            return sample_rate, signal
        except Exception as e_librosa:
            print(f"Błąd przy wczytywaniu {filename} z librosa: {e_librosa}")
            return None, None


def save_wav(filename, sample_rate, signal):
    """Zapisuje sygnał do pliku WAV."""
    if np.max(np.abs(signal)) > 1.0:
        print("Ostrzeżenie: Sygnał przekracza zakres [-1, 1]. Może wystąpić obcięcie.")
    signal_int16 = np.int16(signal * 32767)
    wav.write(filename, sample_rate, signal_int16)
    print(f"Zapisano sygnał do {filename}")

def plot_signals(original, reconstructed, title="Porównanie sygnałów"):
    """Rysuje oryginalny i zrekonstruowany sygnał."""
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(original)
    plt.title("Sygnał oryginalny")
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")

    plt.subplot(2, 1, 2)
    plt.plot(reconstructed)
    plt.title("Sygnał zrekonstruowany")
    plt.xlabel("Próbka")
    plt.ylabel("Amplituda")

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# --- Koder ---
def lpc_encoder(signal, frame_size, hop_size, lpc_order, mode="full_residual", poly_order_spec_env=7):
    """
    Koder LPC z sygnałem resztkowym.
    mode: 'full_residual' lub 'simplified_residual'
    """
    num_frames = (len(signal) - frame_size) // hop_size + 1
    window = scipy.signal.windows.hamming(frame_size)  # ZAMIANA na okno Hamminga

    lpc_coeffs_list = []
    excitation_params_list = []

    print(f"Kodowanie ({mode})...")
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal[start:end]

        if len(frame) < frame_size:
            continue

        frame_windowed = frame * window
        frame_windowed = frame_windowed - np.mean(frame_windowed)  # USUWANIE DC OFFSETU
        is_voiced, pitch = detect_voiced_unvoiced(frame_windowed)

        # 1. Obliczanie współczynników LPC (filtr A(z))
        try:
            a_lpc = librosa.lpc(frame_windowed, order=lpc_order)  # [1, a1, ..., ap]
        except Exception as e:
            print(f"Błąd w librosa.lpc dla ramki {i}: {e}. Używam zerowych współczynników.")
            a_lpc = np.zeros(lpc_order + 1)

        lpc_coeffs_list.append(a_lpc)  # Przechowuj pełny wektor [1, a1, ..., ap]

        # 2. Obliczanie sygnału resztkowego e[n] = x[n] * A(z)
        residual_frame = scipy.signal.lfilter(a_lpc, [1.0], frame_windowed)

        if mode == "full_residual":
            excitation_params_list.append(residual_frame)
        elif mode == "simplified_residual":
            spectrum_residual = np.abs(np.fft.fft(residual_frame, n=frame_size))
            half_spectrum_len = frame_size // 2 + 1
            spectrum_to_smooth = spectrum_residual[:half_spectrum_len]
            smoothing_window_len = 5
            if smoothing_window_len > 1:
                smooth_filter = np.ones(smoothing_window_len) / smoothing_window_len
                smoothed_magnitude_half = np.convolve(spectrum_to_smooth, smooth_filter, mode='same')
            else:
                smoothed_magnitude_half = spectrum_to_smooth
            x_axis = np.arange(half_spectrum_len)
            try:
                poly_coeffs = np.polyfit(x_axis, smoothed_magnitude_half, poly_order_spec_env)
            except (np.linalg.LinAlgError, ValueError) as e:
                print(f"Błąd polyfit w ramce {i}: {e}. Używam zerowych współczynników wielomianu.")
                poly_coeffs = np.zeros(poly_order_spec_env + 1)
            excitation_params_list.append(poly_coeffs)
        else:
            raise ValueError("Nieznany tryb kodera.")

    return lpc_coeffs_list, excitation_params_list, frame_size, hop_size, lpc_order

# --- Dekoder ---
def lpc_decoder(lpc_coeffs_list, excitation_params_list, frame_size, hop_size, lpc_order, original_signal_len,
                mode="full_residual", poly_order_spec_env=7):
    """
    Dekoder LPC z sygnałem resztkowym.
    """
    num_frames = len(lpc_coeffs_list)
    reconstructed_signal = np.zeros(original_signal_len)
    window = scipy.signal.windows.hamming(frame_size)  # ZAMIANA na okno Hamminga

    print(f"Dekodowanie ({mode})...")
    for i in range(num_frames):
        a_lpc = lpc_coeffs_list[i]  # [1, a1, ..., ap]
        excitation_param = excitation_params_list[i]

        filter_A_coeffs_den = a_lpc  # już [1, a1, ..., ap]

        if mode == "full_residual":
            excitation_frame = excitation_param
        elif mode == "simplified_residual":
            poly_coeffs = excitation_param
            half_spectrum_len = frame_size // 2 + 1
            x_axis = np.arange(half_spectrum_len)
            approx_magnitude_half = np.polyval(poly_coeffs, x_axis)
            approx_magnitude_half = np.maximum(approx_magnitude_half, 0)
            full_magnitude_spectrum = np.zeros(frame_size)
            full_magnitude_spectrum[0:half_spectrum_len] = approx_magnitude_half
            full_magnitude_spectrum[half_spectrum_len:] = approx_magnitude_half[frame_size//2-1:0:-1]
            random_phases = np.random.uniform(-np.pi, np.pi, half_spectrum_len)
            random_phases[0] = 0
            if frame_size % 2 == 0:
                random_phases[-1] = 0
            full_phase_spectrum = np.zeros(frame_size)
            full_phase_spectrum[0:half_spectrum_len] = random_phases
            full_phase_spectrum[half_spectrum_len:] = -random_phases[frame_size//2-1:0:-1]
            reconstructed_fft_coeffs = full_magnitude_spectrum * np.exp(1j * full_phase_spectrum)
            excitation_frame = np.real(np.fft.ifft(reconstructed_fft_coeffs, n=frame_size))
        else:
            raise ValueError("Nieznany tryb dekodera.")

        synthesized_frame = scipy.signal.lfilter([1.0], filter_A_coeffs_den, excitation_frame)

        start = i * hop_size
        end = start + frame_size
        if end <= original_signal_len:
            reconstructed_signal[start:end] += synthesized_frame * window

    # Normalizacja po overlap-add, aby uniknąć przesterowania
    max_val = np.max(np.abs(reconstructed_signal))
    if max_val > 1.0:
        reconstructed_signal /= max_val

    return reconstructed_signal


# --- Główny skrypt ---
if __name__ == "__main__":
    # Lista plików do przetworzenia
    # Upewnij się, że te pliki istnieją w tym samym katalogu co skrypt,
    # lub podaj pełne ścieżki.
    # Przykładowe nazwy, zmień na swoje:
    audio_files = ["C:\\Users\\wiedzmok\\CPS\\lab10\\mowa1.wav"] 
    # Możesz użyć librosa.util.example(), np. 'brahms' lub 'nutcracker'
    # audio_files = [librosa.ex('brahms'), librosa.ex('nutcracker'), librosa.ex('choice')]


    for audio_file in audio_files:
        print(f"\n--- Przetwarzanie pliku: {audio_file} ---")
        sample_rate, signal = load_wav(audio_file)
        
        if signal is None:
            print(f"Nie udało się wczytać pliku {audio_file}. Pomijam.")
            continue

        # 1. Kodowanie/dekodowanie z pełnym sygnałem resztkowym
        print("\n*** Metoda: Pełny sygnał resztkowy ***")
        lpc_coeffs_full, residual_frames_full, fs, hs, lpc_o = lpc_encoder(
            signal, FRAME_SIZE, HOP_SIZE, LPC_ORDER, mode="full_residual"
        )
        reconstructed_signal_full = lpc_decoder(
            lpc_coeffs_full, residual_frames_full, fs, hs, lpc_o, len(signal), mode="full_residual"
        )
        output_filename_full = f"{audio_file.split('.')[0]}_reconstructed_full_residual.wav"
        save_wav(output_filename_full, sample_rate, reconstructed_signal_full)
        plot_signals(signal, reconstructed_signal_full, title=f"Pełny resztkowy - {audio_file}")

        # 2. Opcjonalne: Kodowanie/dekodowanie z uproszczonym sygnałem resztkowym
        print("\n*** Metoda: Uproszczony sygnał resztkowy (obwiednia widmowa) ***")
        lpc_coeffs_simp, poly_coeffs_list_simp, fs_s, hs_s, lpc_o_s = lpc_encoder(
            signal, FRAME_SIZE, HOP_SIZE, LPC_ORDER, 
            mode="simplified_residual", 
            poly_order_spec_env=POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE
        )
        reconstructed_signal_simp = lpc_decoder(
            lpc_coeffs_simp, poly_coeffs_list_simp, fs_s, hs_s, lpc_o_s, len(signal), 
            mode="simplified_residual",
            poly_order_spec_env=POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE
        )
        output_filename_simp = f"{audio_file.split('.')[0]}_reconstructed_simplified_residual.wav"
        save_wav(output_filename_simp, sample_rate, reconstructed_signal_simp)
        plot_signals(signal, reconstructed_signal_simp, title=f"Uproszczony resztkowy - {audio_file}")

        print(f"Zakończono przetwarzanie {audio_file}")