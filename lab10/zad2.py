import numpy as np
from scipy.io import wavfile
from scipy.signal.windows import hamming

# --- Parametry ---
FRAME_SIZE = 400
HOP_SIZE = 100
LPC_ORDER = 16

# --- Eksperymenty z pobudzeniem ---
def impulse_train(period, length):
    """Generuje pobudzenie impulsowe o zadanym okresie."""
    x = np.zeros(length)
    x[::period] = 1.0
    return x

def get_residual(signal, a_lpc):
    """Oblicza sygnał resztkowy przez filtrację odwrotną."""
    from scipy.signal import lfilter
    return lfilter(a_lpc, [1.0], signal)

def average_residual_periods(residual, period, num_periods=5):
    """Uśrednia kilka okresów sygnału resztkowego."""
    periods = []
    for i in range(num_periods):
        start = i * period
        end = start + period
        if end <= len(residual):
            periods.append(residual[start:end])
    return np.mean(periods, axis=0) if periods else np.zeros(period)

# --- Koder (prosty, tylko LPC) ---
def lpc_encoder(signal):
    frames = []
    lpc_list = []
    voiced_flags = []
    pitch_list = []
    for start in range(0, len(signal)-FRAME_SIZE, HOP_SIZE):
        frame = signal[start:start+FRAME_SIZE] * hamming(FRAME_SIZE)
        frame = frame - np.mean(frame)
        # Prosta detekcja dźwięczności i tonu podstawowego (autokorelacja)
        corr = np.correlate(frame, frame, mode='full')
        corr = corr[len(corr)//2:]
        corr[0] = 0
        peak = np.argmax(corr[40:200]) + 40  # szukaj pitch w zakresie 40-200 próbek
        pitch = peak if corr[peak] > 0.3 * corr[1] else 0
        is_voiced = pitch > 0
        # LPC
        r = np.correlate(frame, frame, mode='full')[FRAME_SIZE-1:FRAME_SIZE+LPC_ORDER]
        a_lpc = levinson_durbin(r, LPC_ORDER)
        frames.append(frame)
        lpc_list.append(a_lpc)
        voiced_flags.append(is_voiced)
        pitch_list.append(pitch)
    return frames, lpc_list, voiced_flags, pitch_list

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

# --- Dekoder z różnymi eksperymentami ---
def lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="noise", fixed_pitch=80, res_exc=None, coldvox=None):
    out = np.zeros(len(frames)*HOP_SIZE + FRAME_SIZE)
    for i, (frame, a_lpc, is_voiced, pitch) in enumerate(zip(frames, lpc_list, voiced_flags, pitch_list)):
        # --- Pobudzenie ---
        if mode == "noise":
            excitation = np.random.randn(FRAME_SIZE)
        elif mode == "impulse":
            excitation = impulse_train(pitch if is_voiced and pitch > 0 else fixed_pitch, FRAME_SIZE)
        elif mode == "impulse_half_pitch":
            excitation = impulse_train(2*pitch if is_voiced and pitch > 0 else fixed_pitch, FRAME_SIZE)
        elif mode == "impulse_fixed":
            excitation = impulse_train(fixed_pitch, FRAME_SIZE)
        elif mode == "residual_period" and res_exc is not None:
            # Jeden okres sygnału resztkowego
            excitation = np.tile(res_exc, FRAME_SIZE // len(res_exc) + 1)[:FRAME_SIZE]
        elif mode == "coldvox" and coldvox is not None:
            excitation = coldvox[i*FRAME_SIZE:(i+1)*FRAME_SIZE]
            if len(excitation) < FRAME_SIZE:
                excitation = np.pad(excitation, (0, FRAME_SIZE-len(excitation)))
        else:
            excitation = np.random.randn(FRAME_SIZE)
        # --- Synteza ---
        from scipy.signal import lfilter
        synth = lfilter([1.0], a_lpc, excitation)
        out[i*HOP_SIZE:i*HOP_SIZE+FRAME_SIZE] += synth
    out /= np.max(np.abs(out)+1e-12)
    return out

# --- MAIN ---
if __name__ == "__main__":
    # Wczytaj mowę i coldvox
    fs, signal = wavfile.read("C:\\Users\\wiedzmok\\CPS\\lab10\\mowa1.wav")
    if signal.dtype != np.float32:
        signal = signal.astype(np.float32) / np.max(np.abs(signal))
    frames, lpc_list, voiced_flags, pitch_list = lpc_encoder(signal)

    # Eksperyment 1: ignoruj V/UV, zawsze szum (bezdźwięczne)
    rec_noise = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="noise")
    wavfile.write("zad2mowa1_noise.wav", fs, (rec_noise*32767).astype(np.int16))

    # Eksperyment 2: V/UV, ale dla dźwięcznych dwukrotnie niższy pitch
    rec_half_pitch = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="impulse_half_pitch")
    wavfile.write("zad2mowa1_half_pitch.wav", fs, (rec_half_pitch*32767).astype(np.int16))

    # Eksperyment 3: V/UV, ale dla dźwięcznych stały pitch
    rec_fixed_pitch = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="impulse_fixed", fixed_pitch=80)
    wavfile.write("zad2mowa1_fixed_pitch.wav", fs, (rec_fixed_pitch*32767).astype(np.int16))

    # Eksperyment 4: ignoruj V/UV, pobudzenie z coldvox.wav
    fs_cold, coldvox = wavfile.read("C:\\Users\\wiedzmok\\CPS\\lab10\\coldvox.wav")
    if coldvox.dtype != np.float32:
        coldvox = coldvox.astype(np.float32) / np.max(np.abs(coldvox))
    rec_coldvox = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="coldvox", coldvox=coldvox)
    wavfile.write("zad2mowa1_coldvox.wav", fs, (rec_coldvox*32767).astype(np.int16))

    # Eksperyment 5: pobudzenie sygnałem resztkowym
    # Wybierz fragment dźwięczny
    voiced_idx = [i for i, v in enumerate(voiced_flags) if v and pitch_list[i] > 0]
    if voiced_idx:
        idx = voiced_idx[0]
        frame = frames[idx]
        a_lpc = lpc_list[idx]
        pitch = pitch_list[idx]
        residual = get_residual(frame, a_lpc)
        # Jeden okres
        res_period = residual[:pitch]
        rec_residual = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="residual_period", res_exc=res_period)
        wavfile.write("zad2mowa1_residual_period.wav", fs, (rec_residual*32767).astype(np.int16))
        # Uśredniony okres
        avg_res = average_residual_periods(residual, pitch, num_periods=5)
        rec_avg_res = lpc_decoder(frames, lpc_list, voiced_flags, pitch_list, mode="residual_period", res_exc=avg_res)
        wavfile.write("zad2mowa1_avg_residual.wav", fs, (rec_avg_res*32767).astype(np.int16))