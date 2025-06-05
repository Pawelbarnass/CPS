import numpy as np
import scipy.signal
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import librosa

# --- Parametry ---
FRAME_SIZE = 400
HOP_SIZE = 100
LPC_ORDER = 24
POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE = 7
def load_wav(filename):
    """Wczytuje plik WAV jako float32 w zakresie [-1, 1]."""
    try:
        sample_rate, signal = wav.read(filename)
        if signal.dtype == np.int16:
            signal = signal.astype(np.float32) / 32768.0
        elif signal.dtype == np.int32:
            signal = signal.astype(np.float32) / 2147483648.0
        elif signal.dtype == np.uint8:
            signal = (signal.astype(np.float32) - 128) / 128.0
        else:
            signal = signal.astype(np.float32)
        if signal.ndim > 1:
            signal = signal[:, 0]
        return sample_rate, signal
    except Exception as e:
        print(f"Błąd przy wczytywaniu {filename}: {e}")
        return None, None
    
def save_wav(filename, sample_rate, signal):
    """Zapisuje sygnał do pliku WAV jako int16."""
    signal = np.clip(signal, -1, 1)
    wav.write(filename, sample_rate, (signal * 32767).astype(np.int16))
# --- Detekcja głosek dźwięcznych (U/V) i tonu podstawowego ---
def detect_voiced_unvoiced(frame, prev_pitch=[0], prev_energy=[0.1], energy_alpha=0.9, corr_thresh=0.3, cepstrum_thresh=0.1):
    energy = np.sum(frame**2) / len(frame)
    prev_energy[0] = energy_alpha * prev_energy[0] + (1 - energy_alpha) * energy
    energy_thresh = 0.5 * prev_energy[0]
    corr = np.correlate(frame, frame, mode='full')
    corr = corr[len(corr)//2:]
    corr[0] = 0
    fs = 8000
    min_lag = int(fs/400)
    max_lag = int(fs/60)
    pitch_lag = np.argmax(corr[min_lag:max_lag]) + min_lag
    pitch_val = corr[pitch_lag]
    norm_peak = pitch_val / (np.sum(frame**2) + 1e-12)
    spectrum = np.fft.fft(frame)
    log_spectrum = np.log(np.abs(spectrum) + 1e-12)
    cepstrum = np.fft.ifft(log_spectrum).real
    min_pitch = int(fs / 400)
    max_pitch = int(fs / 60)
    cep_peak = np.max(np.abs(cepstrum[min_pitch:max_pitch]))
    if prev_pitch[0] > 0:
        if abs(pitch_lag - prev_pitch[0]) > 10:
            pitch_lag = prev_pitch[0]
    prev_pitch[0] = pitch_lag
    is_voiced = (energy > energy_thresh) and (norm_peak > corr_thresh) and (cep_peak > cepstrum_thresh)
    return is_voiced, pitch_lag

# --- Levinson-Durbin ---
def levinson_durbin(r, order):
    a = np.zeros(order+1)
    e = r[0]
    a[0] = 1.0
    for i in range(1, order+1):
        acc = r[i]
        for j in range(1, i):
            acc += a[j] * r[i-j]
        k = -acc / e
        a_new = a.copy()
        for j in range(1, i):
            a_new[j] += k * a[i-j]
        a_new[i] = k
        e *= (1 - k**2)
        a = a_new
    return a

# --- Koder ---
def lpc_encoder(signal, frame_size, hop_size, lpc_order, mode="full_residual", poly_order_spec_env=7):
    num_frames = (len(signal) - frame_size) // hop_size + 1
    window = scipy.signal.windows.hamming(frame_size)
    lpc_coeffs_list = []
    excitation_params_list = []
    voiced_flags = []
    pitch_list = []
    prev_pitch = [60]
    prev_energy = [0.1]
    print(f"Kodowanie ({mode})...")
    for i in range(num_frames):
        start = i * hop_size
        end = start + frame_size
        frame = signal[start:end]
        if len(frame) < frame_size:
            continue
        frame_windowed = frame * window
        frame_windowed = frame_windowed - np.mean(frame_windowed)
        # --- Detekcja U/V i tonu podstawowego
        is_voiced, pitch = detect_voiced_unvoiced(frame_windowed, prev_pitch, prev_energy)
        voiced_flags.append(is_voiced)
        pitch_list.append(pitch)
        # --- Levinson-Durbin
        r = np.correlate(frame_windowed, frame_windowed, mode='full')[frame_size-1:frame_size+LPC_ORDER]
        a_lpc = levinson_durbin(r, lpc_order)
        lpc_coeffs_list.append(a_lpc)
        # --- Sygnał resztkowy
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
    return lpc_coeffs_list, excitation_params_list, frame_size, hop_size, lpc_order, voiced_flags, pitch_list

# --- Dekoder z interpolacją LPC ---
def lpc_decoder(lpc_coeffs_list, excitation_params_list, frame_size, hop_size, lpc_order, original_signal_len,
                mode="full_residual", poly_order_spec_env=7, interp_steps=2, voiced_flags=None, pitch_list=None):
    num_frames = len(lpc_coeffs_list)
    reconstructed_signal = np.zeros(original_signal_len)
    window = scipy.signal.windows.hamming(frame_size)
    print(f"Dekodowanie ({mode})...")
    prev_lpc = lpc_coeffs_list[0]
    for i in range(num_frames):
        curr_lpc = lpc_coeffs_list[i]
        for j in range(interp_steps):
            alpha = j / interp_steps
            a_interp = (1-alpha)*prev_lpc + alpha*curr_lpc
            if mode == "full_residual" and voiced_flags is not None and pitch_list is not None:
                if voiced_flags[i]:
                    excitation_frame = np.zeros(frame_size)
                    pitch = max(20, min(frame_size-1, pitch_list[i]))
                    excitation_frame[::pitch] = 1.0
                else:
                    excitation_frame = np.random.randn(frame_size) * 0.3
            elif mode == "full_residual":
                excitation_param = excitation_params_list[i]
                excitation_frame = excitation_param
            elif mode == "simplified_residual":
                # ... (pozostały kod bez zmian)
                pass  # Twój kod do simplified_residual
            else:
                raise ValueError("Nieznany tryb dekodera.")
            synthesized_frame = scipy.signal.lfilter([1.0], a_interp, excitation_frame)
            pos = i*hop_size + j*(frame_size//interp_steps)
            end = pos + frame_size//interp_steps
            if end <= original_signal_len:
                reconstructed_signal[pos:end] += synthesized_frame[:frame_size//interp_steps] * window[:frame_size//interp_steps]
        prev_lpc = curr_lpc
    max_val = np.max(np.abs(reconstructed_signal))
    if max_val > 1.0:
        reconstructed_signal /= max_val
    return reconstructed_signal

# --- Główny skrypt ---
if __name__ == "__main__":
    audio_files = ["C:\\Users\\wiedzmok\\CPS\\lab10\\mowa1.wav"]
    for audio_file in audio_files:
        print(f"\n--- Przetwarzanie pliku: {audio_file} ---")
        sample_rate, signal = load_wav(audio_file)
        if signal is None:
            print(f"Nie udało się wczytać pliku {audio_file}. Pomijam.")
            continue
        print("\n*** Metoda: Pełny sygnał resztkowy ***")
        lpc_coeffs_full, residual_frames_full, fs, hs, lpc_o, voiced_flags, pitch_list = lpc_encoder(
            signal, FRAME_SIZE, HOP_SIZE, LPC_ORDER, mode="full_residual"
        )
        reconstructed_signal_full = lpc_decoder(
            lpc_coeffs_full, residual_frames_full, fs, hs, lpc_o, len(signal), mode="full_residual"
        )
        output_filename_full = f"{audio_file.split('.')[0]}4_reconstructed_full_residual.wav"
        save_wav(output_filename_full, sample_rate, reconstructed_signal_full)
        print("\n*** Metoda: Uproszczony sygnał resztkowy (obwiednia widmowa) ***")
        lpc_coeffs_simp, poly_coeffs_list_simp, fs_s, hs_s, lpc_o_s, _, _ = lpc_encoder(
            signal, FRAME_SIZE, HOP_SIZE, LPC_ORDER,
            mode="simplified_residual",
            poly_order_spec_env=POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE
        )
        reconstructed_signal_simp = lpc_decoder(
            lpc_coeffs_simp, poly_coeffs_list_simp, fs_s, hs_s, lpc_o_s, len(signal),
            mode="simplified_residual",
            poly_order_spec_env=POLY_ORDER_RESIDUAL_SPECTRAL_ENVELOPE
        )
        output_filename_simp = f"{audio_file.split('.')[0]}4_reconstructed_simplified_residual.wav"
        save_wav(output_filename_simp, sample_rate, reconstructed_signal_simp)
        print(f"Zakończono przetwarzanie {audio_file}")

    print("\nPorównanie jakości:")
    print("Odsłuchaj wygenerowane pliki *_reconstructed_full_residual.wav oraz *_reconstructed_simplified_residual.wav.")
    print("Porównaj je z oryginalnymi plikami.")
    print("Zwróć uwagę na:")
    print("  - Ogólną jakość dźwięku, naturalność.")
    print("  - Obecność artefaktów (np. metaliczność, szumy, zniekształcenia).")
    print("  - Zachowanie charakterystyki głosu/instrumentu.")
    print("Pełny sygnał resztkowy powinien dać lepszą jakość niż uproszczony, kosztem większej ilości danych do przesłania (cała ramka resztkowa vs. kilka współczynników wielomianu).")
    print("Uproszczony sygnał resztkowy (z obwiednią widmową i losową fazą) może brzmieć bardziej 'szumiąco' lub 'syntetycznie', ponieważ traci informację o fazie i dokładnej strukturze harmonicznych sygnału resztkowego.")