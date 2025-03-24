import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Parametry
fs = 360  # Częstotliwość próbkowania (Hz)

# Wczytanie pliku
filename = "ekg100.txt"
with open(filename, "r") as file:
    signal = np.array([float(line.strip()) for line in file])

# Oś czasu
t = np.arange(len(signal)) / fs

# 1. Wizualizacja sygnału
plt.figure(figsize=(12, 4))
plt.plot(t, signal, label="Sygnał EKG")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Sygnał EKG")
plt.legend()
plt.grid()
plt.show()

# 2. Dyskretna Transformata Fouriera (DFT)
N = len(signal)
freqs = np.fft.fftfreq(N, 1/fs)  # Oś częstotliwości
X = fft(signal)  # DFT
X_magnitude = np.abs(X)[:N//2]  # Widmo amplitudowe (połowa widma)
freqs_half = freqs[:N//2]  # Zakres [0, fs/2]

# Wykres widma amplitudowego
plt.figure(figsize=(12, 4))
plt.plot(freqs_half, X_magnitude, label="Widmo amplitudowe")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo amplitudowe sygnału EKG")
plt.legend()
plt.grid()
plt.show()

# 3. Odwrotna DFT
signal_reconstructed = np.real(ifft(X))  # Rekonstrukcja sygnału

# Porównanie oryginału i odwrotnej DFT
plt.figure(figsize=(12, 4))
plt.plot(t, signal, label="Oryginalny sygnał")
plt.plot(t, signal_reconstructed, linestyle="dashed", label="Zrekonstruowany sygnał")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Porównanie sygnału oryginalnego i po IDFT")
plt.legend()
plt.grid()
plt.show()

# Różnica sygnałów
difference = signal - signal_reconstructed

plt.figure(figsize=(12, 4))
plt.plot(t, difference, label="Różnica sygnałów")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Różnica między oryginalnym a zrekonstruowanym sygnałem")
plt.legend()
plt.grid()
plt.show()
