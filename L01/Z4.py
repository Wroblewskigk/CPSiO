import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

# Parametry
fs = 360  # Częstotliwość próbkowania (Hz)
lowcut = 5  # Filtr górnoprzepustowy (Hz)
highcut = 60  # Filtr dolnoprzepustowy (Hz)

# Wczytanie pliku (dwie kolumny: czas i amplituda)
filename = "ekg_noise.txt"
data = np.loadtxt(filename)
time = data[:, 0]
signal = data[:, 1]

# 1. Wykres oryginalnego sygnału
#plt.figure(figsize=(12, 4))
plt.plot(time, signal, label="Sygnał EKG z zakłóceniami")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Sygnał EKG z zakłóceniami")
plt.grid()
plt.show()

# 1.1 Widmo oryginalnego sygnału
N = len(signal)
freqs = np.fft.fftfreq(N, 1/fs)
X = fft(signal)
X_magnitude = np.abs(X)[:N//2]
freqs_half = freqs[:N//2]

plt.figure(figsize=(12, 4))
plt.plot(freqs_half, X_magnitude, label="Widmo amplitudowe przed filtracją")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo amplitudowe sygnału EKG przed filtracją")
plt.grid()
plt.show()

# # 2. Filtr dolnoprzepustowy Butterwortha
# def butter_lowpass_filter(data, cutoff, fs, order=4):
#     nyq = 0.5 * fs  # Częstotliwość Nyquista
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='low', analog=False)
#     return filtfilt(b, a, data)
#
# signal_lp = butter_lowpass_filter(signal, highcut, fs)
#
# # Wykres sygnału po filtrze dolnoprzepustowym
# plt.figure(figsize=(12, 4))
# plt.plot(time, signal, label="Oryginalny sygnał")
# plt.plot(time, signal_lp, label="Po filtrze LP (60 Hz)", linestyle="dashed")
# plt.xlabel("Czas (s)")
# plt.ylabel("Amplituda")
# plt.title("Sygnał po filtracji dolnoprzepustowej (60 Hz)")
# plt.legend()
# plt.grid()
# plt.show()
#
# # Widmo sygnału po filtracji LP
# X_lp = fft(signal_lp)
# X_lp_magnitude = np.abs(X_lp)[:N//2]
#
# plt.figure(figsize=(12, 4))
# plt.plot(freqs_half, X_lp_magnitude, label="Widmo po LP (60 Hz)")
# plt.xlabel("Częstotliwość (Hz)")
# plt.ylabel("Amplituda")
# plt.title("Widmo sygnału po filtracji dolnoprzepustowej (60 Hz)")
# plt.legend()
# plt.grid()
# plt.show()
#
# # 3. Filtr górnoprzepustowy Butterwortha
# def butter_highpass_filter(data, cutoff, fs, order=4):
#     nyq = 0.5 * fs
#     normal_cutoff = cutoff / nyq
#     b, a = butter(order, normal_cutoff, btype='high', analog=False)
#     return filtfilt(b, a, data)
#
# signal_bp = butter_highpass_filter(signal_lp, lowcut, fs)
#
# # Wykres sygnału po filtrze pasmowym (5-60 Hz)
# plt.figure(figsize=(12, 4))
# plt.plot(time, signal, label="Oryginalny sygnał")
# plt.plot(time, signal_bp, label="Po filtrze BP (5-60 Hz)", linestyle="dashed")
# plt.xlabel("Czas (s)")
# plt.ylabel("Amplituda")
# plt.title("Sygnał po filtracji pasmowej (5-60 Hz)")
# plt.legend()
# plt.grid()
# plt.show()
#
# # Widmo sygnału po filtracji BP
# X_bp = fft(signal_bp)
# X_bp_magnitude = np.abs(X_bp)[:N//2]
#
# plt.figure(figsize=(12, 4))
# plt.plot(freqs_half, X_bp_magnitude, label="Widmo po BP (5-60 Hz)")
# plt.xlabel("Częstotliwość (Hz)")
# plt.ylabel("Amplituda")
# plt.title("Widmo sygnału po filtracji pasmowej (5-60 Hz)")
# plt.legend()
# plt.grid()
# plt.show()
#
# # 4. Różnica między sygnałami przed i po filtracji
# diff = signal - signal_bp
#
# plt.figure(figsize=(12, 4))
# plt.plot(time, diff, label="Różnica przed i po filtracji")
# plt.xlabel("Czas (s)")
# plt.ylabel("Amplituda")
# plt.title("Różnica sygnału przed i po filtracji")
# plt.legend()
# plt.grid()
# plt.show()
