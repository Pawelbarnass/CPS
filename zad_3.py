import numpy as np
import scipy.io as scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, correlate

mat_data = scipy.loadmat('adsl_x.mat')
x = mat_data['x'].flatten()
print(f"Długość sygnału: {len(x)} próbek")
def simple_correlate(signal, segment):
    L_signal = len(signal)
    L_segment = len(segment)
    result_len = L_signal - L_segment + 1
    corr = np.zeros(result_len)
    segment_normalized = segment / np.sqrt(np.sum(segment**2))  
    for i in range(result_len):
        signal_segment = signal[i:i+L_segment]
        corr[i] = np.sum(signal_segment * segment_normalized)
    
    return corr
M = 32   
N = 512   
block_length = M + N  
L = len(x)


peaks = []
for n in range(L - M + 1):
   #corr = correlate(x, x[n:n+M], mode='valid')  
   corr = simple_correlate(x, x[n:n+M])
   max_val = np.max(np.abs(corr))  
   max_pos = np.where(np.abs(corr) == max_val)[0]
   print (max_pos)
   print(len(max_pos))
   if len(max_pos) >= 2:  
        peaks.append(max_pos)

print("Wykryte pozycje (indeksy) początków prefiksów:")
print(peaks)
print("Odległości między kolejnymi prefiksami:")
print(np.diff(peaks))
print(len(corr))