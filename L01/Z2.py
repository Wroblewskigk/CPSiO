import matplotlib.pyplot as plt
import numpy as np

# Częstotliwość próbkowania (Hz)
fs = 1000
# Czas trwania sygnału (s)
T = 65536 / fs
# Wektor czasowy
t = np.linspace(0, T, 65536, endpoint=False)

#ZADANIE1###############################################################################################################

# Generowanie sygnału sinusoidalnego o częstotliwości 50 Hz
sinFrequency50Hz = 50  # Częstotliwość w Hz
sinSignal50Hz = np.sin(2 * np.pi * sinFrequency50Hz * t)
print(sinSignal50Hz)

#ZADANIE2###############################################################################################################

# Obliczanie i rysowanie transformaty Fouriera
sinSignalFFT = np.fft.fft(sinSignal50Hz)
frequencies = np.fft.fftfreq(len(sinSignal50Hz), 1 / fs)
sinSignalAmplitudeSpectrum = np.abs(sinSignalFFT) / len(sinSignal50Hz)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], sinSignalAmplitudeSpectrum[:len(frequencies) // 2])
plt.title("Widmo amplitudowe sygnału 50 Hz")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.grid()
plt.show()

#ZADANIE3###############################################################################################################

# Generowanie sygnału mieszanego z 50 Hz i 60 Hz
sinFrequency60Hz = 60
sinSignal60Hz = np.sin(2 * np.pi * sinFrequency60Hz * t)
mixedSinSignal = sinSignal50Hz + sinSignal60Hz

# Obliczanie i rysowanie transformaty Fouriera sygnału mieszanego
mixedSinSignalFFT = np.fft.fft(mixedSinSignal)
mixedSinSignalAmplitudeSpectrum = np.abs(mixedSinSignalFFT) / len(mixedSinSignal)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], mixedSinSignalAmplitudeSpectrum[:len(frequencies) // 2])
plt.title("Widmo amplitudowe sygnału mieszanego 50 Hz i 60 Hz")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.grid()
plt.show()

#ZADANIE4###############################################################################################################

# Eksperymentowanie z różnymi częstotliwościami próbkowania
samplingRates = [500, 2000, 5000]
for fsNew in samplingRates:
    # Nowy czas trwania
    TNew = 65536 / fsNew
    tNew = np.linspace(0, TNew, 65536, endpoint=False)
    signalNew = np.sin(2 * np.pi * sinFrequency50Hz * tNew) + np.sin(2 * np.pi * sinFrequency60Hz * tNew)

    fftNew = np.fft.fft(signalNew)
    frequencyNew = np.fft.fftfreq(len(signalNew), 1 / fsNew)
    amplitudeSpectrumNew = np.abs(fftNew) / len(signalNew)

    plt.figure(figsize=(10, 5))
    plt.plot(frequencyNew[:len(frequencyNew) // 2], amplitudeSpectrumNew[:len(frequencyNew) // 2])
    plt.title(f"Widmo amplitudowe (Częstotliwość próbkowania: {fsNew} Hz)")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.grid()
    plt.show()

#ZADANIE5###############################################################################################################

# Obliczanie odwrotnej transformaty Fouriera i porównanie z oryginalnymi sygnałami
reconstructedSinSignal = np.fft.ifft(sinSignalFFT).real
reconstructedMixedSinSignal = np.fft.ifft(mixedSinSignalFFT).real

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], sinSignal50Hz[:1000], label="Oryginalny sygnał 50Hz")
plt.plot(t[:1000], reconstructedSinSignal[:1000], '--', label="Odtworzony sygnał 50Hz")
plt.legend()
plt.title("Porównanie oryginalnego i odtworzonego sygnału 50Hz")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], mixedSinSignal[:1000], label="Oryginalny sygnał mieszany")
plt.plot(t[:1000], reconstructedMixedSinSignal[:1000], '--', label="Odtworzony sygnał mieszany")
plt.legend()
plt.title("Porównanie oryginalnego i odtworzonego sygnału mieszanego")
plt.show()
