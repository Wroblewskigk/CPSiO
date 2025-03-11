import matplotlib.pyplot as plt
import numpy as np

# Define parameters
fs = 1000  # Sampling frequency (Hz)
T = 65536 / fs  # Duration (s)
t = np.linspace(0, T, 65536, endpoint=False) #Set endpoint to true/Time vector

# 1. Generate a 50 Hz sinusoidal signal
freq1 = 50  # Frequency in Hz
signal1 = np.sin(2 * np.pi * freq1 * t)
print(signal1)

# 2. Compute and plot the Fourier Transform
fft_signal1 = np.fft.fft(signal1)
frequencies = np.fft.fftfreq(len(signal1), 1 / fs)
amplitude_spectrum1 = np.abs(fft_signal1) / len(signal1)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], amplitude_spectrum1[:len(frequencies) // 2])
plt.title("Amplitude Spectrum of 50Hz Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 3. Generate a mixed signal with 50 Hz and 60 Hz
freq2 = 60  # Second frequency in Hz
signal2 = np.sin(2 * np.pi * freq2 * t)
mixed_signal = signal1 + signal2

# Compute and plot the Fourier Transform of the mixed signal
fft_mixed = np.fft.fft(mixed_signal)
amplitude_spectrum_mixed = np.abs(fft_mixed) / len(mixed_signal)

plt.figure(figsize=(10, 5))
plt.plot(frequencies[:len(frequencies) // 2], amplitude_spectrum_mixed[:len(frequencies) // 2])
plt.title("Amplitude Spectrum of Mixed 50Hz and 60Hz Signal")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# 4. Experiment with different sampling frequencies
sampling_rates = [500, 2000, 5000]  # Different sampling frequencies
for fs_new in sampling_rates:
    T_new = 65536 / fs_new  # New duration
    t_new = np.linspace(0, T_new, 65536, endpoint=False)
    signal_new = np.sin(2 * np.pi * freq1 * t_new) + np.sin(2 * np.pi * freq2 * t_new)

    fft_new = np.fft.fft(signal_new)
    freq_new = np.fft.fftfreq(len(signal_new), 1 / fs_new)
    amp_spec_new = np.abs(fft_new) / len(signal_new)

    plt.figure(figsize=(10, 5))
    plt.plot(freq_new[:len(freq_new) // 2], amp_spec_new[:len(freq_new) // 2])
    plt.title(f"Amplitude Spectrum (Sampling Rate: {fs_new} Hz)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

# 5. Compute the inverse Fourier Transform and compare with the original signals
reconstructed_signal1 = np.fft.ifft(fft_signal1).real
reconstructed_mixed = np.fft.ifft(fft_mixed).real

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], signal1[:1000], label="Original 50Hz Signal")
plt.plot(t[:1000], reconstructed_signal1[:1000], '--', label="Reconstructed 50Hz Signal")
plt.legend()
plt.title("Comparison of Original and Reconstructed 50Hz Signal")
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(t[:1000], mixed_signal[:1000], label="Original Mixed Signal")
plt.plot(t[:1000], reconstructed_mixed[:1000], '--', label="Reconstructed Mixed Signal")
plt.legend()
plt.title("Comparison of Original and Reconstructed Mixed Signal")
plt.show()
