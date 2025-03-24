import matplotlib.pyplot as plt
import numpy as np

# Definiowanie parametrów
fs = 1000  # Częstotliwość próbkowania (Hz)
T = 65536 / fs  # Czas trwania sygnału (s)
t = np.linspace(0, T, 65536, endpoint=False)  # Wektor czasowy

# 1. Generowanie sygnału sinusoidalnego o częstotliwości 50 Hz
freq1 = 50  # Częstotliwość w Hz
signal1 = np.sin(2 * np.pi * freq1 * t)
print(signal1)

# 2. Obliczanie i rysowanie transformaty Fouriera
fft_signal1 = np.fft.fft(signal1)
frequencies = np.fft.fftfreq(len(signal1), 1 / fs)
amplitude_spectrum1 = np.abs(fft_signal1) / len(signal1)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], amplitude_spectrum1[:len(frequencies) // 2])
plt.title("Widmo amplitudowe sygnału 50 Hz")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.grid()
plt.show()

# 3. Generowanie sygnału mieszanego z 50 Hz i 60 Hz
freq2 = 60  # Druga częstotliwość w Hz
signal2 = np.sin(2 * np.pi * freq2 * t)
mixed_signal = signal1 + signal2

# Obliczanie i rysowanie transformaty Fouriera sygnału mieszanego
fft_mixed = np.fft.fft(mixed_signal)
amplitude_spectrum_mixed = np.abs(fft_mixed) / len(mixed_signal)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], amplitude_spectrum_mixed[:len(frequencies) // 2])
plt.title("Widmo amplitudowe sygnału mieszanego 50 Hz i 60 Hz")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.grid()
plt.show()

# 4. Eksperymentowanie z różnymi częstotliwościami próbkowania
sampling_rates = [500, 2000, 5000]  # Różne częstotliwości próbkowania
for fs_new in sampling_rates:
    T_new = 65536 / fs_new  # Nowy czas trwania
    t_new = np.linspace(0, T_new, 65536, endpoint=False)
    signal_new = np.sin(2 * np.pi * freq1 * t_new) + np.sin(2 * np.pi * freq2 * t_new)

    fft_new = np.fft.fft(signal_new)
    freq_new = np.fft.fftfreq(len(signal_new), 1 / fs_new)
    amp_spec_new = np.abs(fft_new) / len(signal_new)

    plt.figure(figsize=(10, 5))
    plt.plot(freq_new[:len(freq_new) // 2], amp_spec_new[:len(freq_new) // 2])
    plt.title(f"Widmo amplitudowe (Częstotliwość próbkowania: {fs_new} Hz)")
    plt.xlabel("Częstotliwość (Hz)")
    plt.ylabel("Amplituda")
    plt.grid()
    plt.show()

# 5. Obliczanie odwrotnej transformaty Fouriera i porównanie z oryginalnymi sygnałami
reconstructed_signal1 = np.fft.ifft(fft_signal1).real
reconstructed_mixed = np.fft.ifft(fft_mixed).real

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], signal1[:1000], label="Oryginalny sygnał 50Hz")
plt.plot(t[:1000], reconstructed_signal1[:1000], '--', label="Odtworzony sygnał 50Hz")
plt.legend()
plt.title("Porównanie oryginalnego i odtworzonego sygnału 50Hz")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], mixed_signal[:1000], label="Oryginalny sygnał mieszany")
plt.plot(t[:1000], reconstructed_mixed[:1000], '--', label="Odtworzony sygnał mieszany")
plt.legend()
plt.title("Porównanie oryginalnego i odtworzonego sygnału mieszanego")
plt.show()
