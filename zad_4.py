import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

fs = 16000 
T = 0.1    
fc = 500   

def string_to_binary(text):
    binary_string = ''
    for char in text:
        binary_char = bin(ord(char))[2:].zfill(8) 
        binary_string += binary_char
    return binary_string

def modulate_bpsk(binary_string, fs=16000, T=0.1, fc=500):
    t_bit = np.arange(0, T, 1/fs)
    carrier_0 = np.sin(2 * np.pi * fc * t_bit)
    carrier_1 = -np.sin(2 * np.pi * fc * t_bit)  
    
    signal = np.array([])
    for bit in binary_string:
        if bit == '0':
            signal = np.append(signal, carrier_0)
        else:
            signal = np.append(signal, carrier_1)
    return signal

def decode_bpsk(signal, fs=16000, T=0.1, fc=500):
    samples_per_bit = int(T * fs)
    num_bits = len(signal) // samples_per_bit
    
    t_bit = np.arange(0, T, 1/fs)
    carrier = np.sin(2 * np.pi * fc * t_bit)
    
    decoded_bits = ''
    for i in range(num_bits):
        signal_segment = signal[i*samples_per_bit:(i+1)*samples_per_bit]
        correlation = np.sum(signal_segment * carrier)
        if correlation > 0:
            decoded_bits += '0'
        else:
            decoded_bits += '1'
    
    return decoded_bits

def binary_to_string(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

name = "Pawel"  

binary_name = string_to_binary(name)
print(f"Name: {name}")
print(f"ASCII Binary: {binary_name}")
modulated_signal = modulate_bpsk(binary_name)

plt.figure(figsize=(12, 6))
samples_per_bit = int(T * fs)
t_plot = np.arange(0, min(0.5, len(binary_name)*T), 1/fs) 
plt.plot(t_plot, modulated_signal[:len(t_plot)])
plt.title(f"BPSK Modulated Signal for '{name}'")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.grid(True)

for i in range(min(5, len(binary_name))):
    bit = binary_name[i]
    plt.axvspan(i*T, (i+1)*T, alpha=0.1, color='green' if bit=='0' else 'red')
    plt.text((i+0.5)*T, 1.2, f"Bit {i}: {bit}", horizontalalignment='center')

plt.tight_layout()
plt.show()

wavfile.write('name_bpsk.wav', fs, (modulated_signal / np.max(np.abs(modulated_signal))).astype(np.float32))

decoded_binary = decode_bpsk(modulated_signal)
decoded_name = binary_to_string(decoded_binary)
print(f"Decoded binary: {decoded_binary}")
print(f"Decoded name: {decoded_name}")

def modulate_qpsk(binary_string, fs=16000, T=0.1, fc=500):
    if len(binary_string) % 2 != 0:
        binary_string += '0' 
    
    t_bit = np.arange(0, T, 1/fs)
    signal = np.array([])
    
    for i in range(0, len(binary_string), 2):
        bit_pair = binary_string[i:i+2]
        
        if bit_pair == '00':
           
            carrier = np.cos(2 * np.pi * fc * t_bit + np.pi/4)
        elif bit_pair == '01':
            
            carrier = np.cos(2 * np.pi * fc * t_bit + 3*np.pi/4)
        elif bit_pair == '10':
           
            carrier = np.cos(2 * np.pi * fc * t_bit - np.pi/4)
        else:  
            carrier = np.cos(2 * np.pi * fc * t_bit - 3*np.pi/4)
            
        signal = np.append(signal, carrier)
        
    return signal
def decode_qpsk(signal, fs=16000, T=0.1, fc=500):

    samples_per_symbol = int(T * fs)
    num_symbols = len(signal) // samples_per_symbol
    
    t_symbol = np.arange(0, T, 1/fs)
    
    carrier_45 = np.cos(2 * np.pi * fc * t_symbol + np.pi/4)    # 00
    carrier_135 = np.cos(2 * np.pi * fc * t_symbol + 3*np.pi/4)  # 01
    carrier_315 = np.cos(2 * np.pi * fc * t_symbol - np.pi/4)    # 10
    carrier_225 = np.cos(2 * np.pi * fc * t_symbol - 3*np.pi/4)  # 11
    
    decoded_bits = ''
    
    for i in range(num_symbols):
        signal_segment = signal[i*samples_per_symbol:(i+1)*samples_per_symbol]
        
        corr_45 = np.sum(signal_segment * carrier_45)
        corr_135 = np.sum(signal_segment * carrier_135)
        corr_315 = np.sum(signal_segment * carrier_315)
        corr_225 = np.sum(signal_segment * carrier_225)
        
        correlations = [corr_45, corr_135, corr_315, corr_225]
        max_index = np.argmax(correlations)

        if max_index == 0:
            decoded_bits += '00'  # Phase 45°
        elif max_index == 1:
            decoded_bits += '01'  # Phase 135°
        elif max_index == 2:
            decoded_bits += '10'  # Phase 315°
        else:
            decoded_bits += '11'
    
    return decoded_bits
qpsk_signal = modulate_qpsk(binary_name)
wavfile.write('name_qpsk.wav', fs, (qpsk_signal / np.max(np.abs(qpsk_signal))).astype(np.float32))
decodet_qpsk_signal = decode_qpsk(qpsk_signal)
print(f"QPSK Decoded: {decodet_qpsk_signal}")

print("\nComparison:")
print(f"BPSK signal length: {len(modulated_signal)/fs:.2f} seconds")
print(f"QPSK signal length: {len(qpsk_signal)/fs:.2f} seconds")
print(f"Transmission speedup with QPSK: 2x")