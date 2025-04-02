import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
amplitude = 230
frequency = 50 

def task_A():
    duration_A = 0.1

    fs1 = 10000
    t1 = np.arange(0, duration_A, 1/fs1)
    y1 = amplitude * np.sin(2 * np.pi * frequency * t1)
    
    fs2 = 500
    t2 = np.arange(0, duration_A, 1/fs2)
    y2 = amplitude * np.sin(2 * np.pi * frequency * t2)
    
    fs3 = 200
    t3 = np.arange(0, duration_A, 1/fs3)
    y3 = amplitude * np.sin(2 * np.pi * frequency * t3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t1, y1, 'b-', label=f'fs1 = {fs1} Hz')
    plt.plot(t2, y2, 'r-o', label=f'fs2 = {fs2} Hz')
    plt.plot(t3, y3, 'k-x', label=f'fs3 = {fs3} Hz')
    plt.title('A. Sinusoida 50 Hz z różnymi częstotliwościami próbkowania')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def task_B():
    duration_B = 1 
    
    fs1 = 10000
    t1 = np.arange(0, duration_B, 1/fs1)
    y1 = amplitude * np.sin(2 * np.pi * frequency * t1)
    
    fs2 = 500
    t2 = np.arange(0, duration_B, 1/fs2)
    y2 = amplitude * np.sin(2 * np.pi * frequency * t2)
    
    fs3 = 200
    t3 = np.arange(0, duration_B, 1/fs3)
    y3 = amplitude * np.sin(2 * np.pi * frequency * t3)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t1, y1, 'b-', label=f'fs1 = {fs1} Hz')
    plt.plot(t2, y2, 'r-o', label=f'fs2 = {fs2} Hz')
    plt.plot(t3, y3, 'k-x', label=f'fs3 = {fs3} Hz')
    plt.title('B. Sinusoida 50 Hz z różnymi częstotliwościami próbkowania')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.tight_layout()
    plt.show()

def task_C():
    
    fs = 100   
    duration = 1 
    t = np.arange(0, duration, 1/fs)
    
    plt.figure(figsize=(10, 5))
plt.ion()

for i, freq in enumerate(range(0, 301, 5)):
    print(f"Iteration {i+1}/{61}: frequency = {freq} Hz")
    y = amplitude * np.sin(2 * np.pi * freq * t)
    
    plt.clf()
    plt.plot(t, y)
    plt.title(f'Sinusoida {freq} Hz (fs = {fs} Hz)')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.grid(True)
    plt.tight_layout()
    
    plt.draw()
    plt.pause(0.3)  # Pause for 0.3 seconds before showing the next plot

    plt.ioff()  # Turn off interactive mode
    plt.show()  # K
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.sin(2 * np.pi * 5 * t), 'b-', label='5 Hz')
    plt.plot(t, amplitude * np.sin(2 * np.pi * 105 * t), 'r-', label='105 Hz')
    plt.plot(t, amplitude * np.sin(2 * np.pi * 205 * t), 'g-', label='205 Hz')
    plt.title('Porównanie sinusoid o częstotliwościach 5 Hz, 105 Hz i 205 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.sin(2 * np.pi * 95 * t), 'b-', label='95 Hz')
    plt.plot(t, amplitude * np.sin(2 * np.pi * 195 * t), 'r-', label='195 Hz')
    plt.plot(t, amplitude * np.sin(2 * np.pi * 295 * t), 'g-', label='295 Hz')
    plt.title('Porównanie sinusoid o częstotliwościach 95 Hz, 195 Hz i 295 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.sin(2 * np.pi * 95 * t), 'b-', label='95 Hz')
    plt.plot(t, amplitude * np.sin(2 * np.pi * 105 * t), 'r-', label='105 Hz')
    plt.title('Porównanie sinusoid o częstotliwościach 95 Hz i 105 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nPowtórzenie eksperymentu z funkcją cosinus:\n")
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.cos(2 * np.pi * 5 * t), 'b-', label='5 Hz')
    plt.plot(t, amplitude * np.cos(2 * np.pi * 105 * t), 'r-', label='105 Hz')
    plt.plot(t, amplitude * np.cos(2 * np.pi * 205 * t), 'g-', label='205 Hz')
    plt.title('Porównanie cosinusoid o częstotliwościach 5 Hz, 105 Hz i 205 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.cos(2 * np.pi * 95 * t), 'b-', label='95 Hz')
    plt.plot(t, amplitude * np.cos(2 * np.pi * 195 * t), 'r-', label='195 Hz')
    plt.plot(t, amplitude * np.cos(2 * np.pi * 295 * t), 'g-', label='295 Hz')
    plt.title('Porównanie cosinusoid o częstotliwościach 95 Hz, 195 Hz i 295 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    z
    plt.figure(figsize=(12, 6))
    plt.plot(t, amplitude * np.cos(2 * np.pi * 95 * t), 'b-', label='95 Hz')
    plt.plot(t, amplitude * np.cos(2 * np.pi * 105 * t), 'r-', label='105 Hz')
    plt.title('Porównanie cosinusoid o częstotliwościach 95 Hz i 105 Hz')
    plt.xlabel('Czas (s)')
    plt.ylabel('Amplituda (V)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

