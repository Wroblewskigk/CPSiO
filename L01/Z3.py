import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft

# Częstotliwość próbkowania (Hz)
fs = 360

#ZADANIE1###############################################################################################################

filename = "ekg100.txt"
with open(filename, "r") as file:
    signal = np.array([float(line.strip()) for line in file])

# Oś czasu
t = np.arange(len(signal)) / fs

plt.figure(figsize=(12, 4))
plt.plot(t, signal, label="Sygnał EKG")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Sygnał EKG")
plt.legend()
plt.grid()
plt.show()

#ZADANIE2###############################################################################################################

# N - Ilość próbek w sygnale signal
N = len(signal)
# Oś częstotliwości
frequencies = np.fft.fftfreq(N, 1 / fs)
# Transformata Fouriera sygnału signal
X = fft(signal)
XMagnitude = np.abs(X)[:N // 2]  # Widmo amplitudowe (połowa widma)
# Zakres [0, fs/2]
positiveFrequencies = frequencies[:N // 2]

# Wykres widma amplitudowego
plt.figure(figsize=(12, 4))
plt.plot(positiveFrequencies, XMagnitude, label="Widmo amplitudowe")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo amplitudowe sygnału EKG")
plt.legend()
plt.grid()
plt.show()

#ZADANIE3###############################################################################################################

# Rekonstrukcja sygnału odwrotną transformatą Fouriera
reconstructedSignal = np.real(ifft(X))

# Porównanie sygnału oryginalnego i sygnału po odwrotnej transformacie Fouriera
plt.figure(figsize=(12, 4))
plt.plot(t, signal, label="Oryginalny sygnał")
plt.plot(t, reconstructedSignal, linestyle="dashed", label="Zrekonstruowany sygnał")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Porównanie sygnału oryginalnego i sygnału po odwrotnej transformacie Fouriera")
plt.legend()
plt.grid()
plt.show()

# Różnica sygnałów
difference = signal - reconstructedSignal

plt.figure(figsize=(12, 4))
plt.plot(t, difference, label="Różnica sygnałów")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Różnica między oryginalnym a zrekonstruowanym sygnałem")
plt.legend()
plt.grid()
plt.show()
