import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

# Parameters
fs = 16000  # Sampling frequency in Hz
T = 0.1     # Duration of each bit in seconds (100 ms)
fc = 500    # Carrier frequency in Hz

# Function to convert a string to its binary ASCII representation
def string_to_binary(text):
    binary_string = ''
    for char in text:
        # Convert each character to its ASCII code and then to binary
        binary_char = bin(ord(char))[2:].zfill(8)  # Ensure 8 bits per character
        binary_string += binary_char
    return binary_string

# Function to modulate the binary sequence using BPSK
def modulate_bpsk(binary_string, fs=16000, T=0.1, fc=500):
    # Create time vector for one bit period
    t_bit = np.arange(0, T, 1/fs)
    # Create carrier signals for bit 0 and bit 1
    carrier_0 = np.sin(2 * np.pi * fc * t_bit)
    carrier_1 = -np.sin(2 * np.pi * fc * t_bit)  # Negative sine = 180° phase shift
    
    signal = np.array([])
    for bit in binary_string:
        if bit == '0':
            signal = np.append(signal, carrier_0)
        else:  # bit == '1'
            signal = np.append(signal, carrier_1)
    return signal

# Function to decode BPSK modulated signal
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

# Function to convert binary back to ASCII text
def binary_to_string(binary):
    text = ''
    for i in range(0, len(binary), 8):
        byte = binary[i:i+8]
        if len(byte) == 8:
            text += chr(int(byte, 2))
    return text

# Name to transmit - change this to your name
name = "Pawel"  # Change to your name

# Convert name to binary
binary_name = string_to_binary(name)
print(f"Name: {name}")
print(f"ASCII Binary: {binary_name}")

# Modulate the signal
modulated_signal = modulate_bpsk(binary_name)

# Plot the signal (first 4 bits for clarity)
plt.figure(figsize=(12, 6))
samples_per_bit = int(T * fs)
t_plot = np.arange(0, min(0.5, len(binary_name)*T), 1/fs)  # Show first 0.5 seconds or all bits
plt.plot(t_plot, modulated_signal[:len(t_plot)])
plt.title(f"BPSK Modulated Signal for '{name}'")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.grid(True)

# Mark bit regions and values
for i in range(min(5, len(binary_name))):
    bit = binary_name[i]
    plt.axvspan(i*T, (i+1)*T, alpha=0.1, color='green' if bit=='0' else 'red')
    plt.text((i+0.5)*T, 1.2, f"Bit {i}: {bit}", horizontalalignment='center')

plt.tight_layout()
plt.show()

# Save the audio file
wavfile.write('name_bpsk.wav', fs, (modulated_signal / np.max(np.abs(modulated_signal))).astype(np.float32))

# Function to play the sound at different sampling rates
def play_audio(signal, original_fs, playback_fs):
    """Play or export the signal at different sampling rates."""
    # Here we keep the original signal but tell the audio device to play it at a different rate
    # This will change the perceived pitch and speed
    print(f"Playing at {playback_fs/1000} kHz...")
    
    # For actual implementation, you would use:
    # sd.play(signal, playback_fs)
    # sd.wait()
    
    return Audio(signal, rate=playback_fs)

# Decode the signal to verify correctness
decoded_binary = decode_bpsk(modulated_signal)
decoded_name = binary_to_string(decoded_binary)
print(f"Decoded binary: {decoded_binary}")
print(f"Decoded name: {decoded_name}")

# Example of a more efficient modulation: QPSK (2 bits per symbol)
def modulate_qpsk(binary_string, fs=16000, T=0.1, fc=500):
    """Modulate using QPSK - 2 bits per symbol."""
    # Make sure we have an even number of bits (required for QPSK)
    if len(binary_string) % 2 != 0:
        binary_string += '0'  # Pad with a 0 if necessary
    
    t_bit = np.arange(0, T, 1/fs)
    signal = np.array([])
    
    # Process two bits at a time
    for i in range(0, len(binary_string), 2):
        bit_pair = binary_string[i:i+2]
        
        if bit_pair == '00':
            # Phase = 45°
            carrier = np.cos(2 * np.pi * fc * t_bit + np.pi/4)
        elif bit_pair == '01':
            # Phase = 135°
            carrier = np.cos(2 * np.pi * fc * t_bit + 3*np.pi/4)
        elif bit_pair == '10':
            # Phase = 315° (-45°)
            carrier = np.cos(2 * np.pi * fc * t_bit - np.pi/4)
        else:  # bit_pair == '11'
            # Phase = 225° (-135°)
            carrier = np.cos(2 * np.pi * fc * t_bit - 3*np.pi/4)
            
        signal = np.append(signal, carrier)
        
    return signal

# Create QPSK version for comparison
qpsk_signal = modulate_qpsk(binary_name)

print("\nComparison:")
print(f"BPSK signal length: {len(modulated_signal)/fs:.2f} seconds")
print(f"QPSK signal length: {len(qpsk_signal)/fs:.2f} seconds")
print(f"Transmission speedup with QPSK: 2x")