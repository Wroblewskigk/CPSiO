import matplotlib.pyplot as plt
import numpy as np
from scipy.fftpack import fft
from scipy.signal import butter, filtfilt

# Parametry programu
# Częstotliwość próbkowania [Hz]
fs = 360
# Ograniczenie dolne filtru górnoprzepustowego [Hz]
lowerBound = 5
# Ograniczenie górne filtru dolnoprzepustowego [Hz]
upperBound = 60

#ZADANIE1###############################################################################################################

# Wczytanie pliku z szumami
filename = "ekg_noise.txt"
noiseData = np.loadtxt(filename)
# Pierwsza kolumna pliku
time = noiseData[:, 0]
# Druga kolumna pliku
signalValues = noiseData[:, 1]

plt.plot(time, signalValues, label="Sygnał EKG z zakłóceniami")
plt.xlabel("Czas [s]")
plt.ylabel("Amplituda")
plt.title("Sygnał EKG z zakłóceniami")
plt.grid()
plt.show()

#ZADANIE2###############################################################################################################

# N - Ilość próbek w sygnale signalValues
N = len(signalValues)
frequencyBins = np.fft.fftfreq(N, 1 / fs)
# X - Wyliczona szybka transformata Fouriera
X = fft(signalValues)
# XMagnitude - Amplitudy szybkiej transformaty Fouriera
XMagnitude = np.abs(X)[:N//2]
# Transformata Fouriera generuje się w zakresie dodatnim i ujemnym (fs/2)
# zachowujemy tylko wartości z zakresu dodatniego poniewarz transformata
# jest w tych zakresach symetryczna
frequencyBinsPositiveHalf = frequencyBins[:N // 2]

plt.figure(figsize=(12, 4))
plt.plot(frequencyBinsPositiveHalf, XMagnitude, label="Widmo amplitudowe przed filtracją")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo amplitudowe sygnału EKG przed filtracją")
plt.grid()
plt.show()

def butterworthLowpassFilter(data, cutoffFrequency, order=4):
    nyquistFrequency = 0.5 * fs
    # Znormalizowana częstotliwość odcinana przez filter
    normalizedCutoffFrequency = cutoffFrequency / nyquistFrequency
    b, a = butter(order, normalizedCutoffFrequency, btype='low', analog=False)
    return filtfilt(b, a, data)

signalAfterLowpassFilter = butterworthLowpassFilter(signalValues, upperBound)

# Wykres sygnału po filtrze dolnoprzepustowym
plt.figure(figsize=(12, 4))
plt.plot(time, signalValues, label="Oryginalny sygnał")
plt.plot(time, signalAfterLowpassFilter, label="Po filtrze LP (60 Hz)") #linestyle="dashed"
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Sygnał po filtracji dolnoprzepustowej (60 Hz)")
plt.legend()
plt.grid()
plt.show()

# Widmo sygnału po filtracji dolnoprzepustowej
XAfterLowpassFilter = fft(signalAfterLowpassFilter)
XMagnitudeAfterLowpassFilter = np.abs(XAfterLowpassFilter)[:N // 2]

plt.figure(figsize=(12, 4))
plt.plot(frequencyBinsPositiveHalf, XMagnitude, label="Widmo przed filtracją", alpha=0.7)
plt.plot(frequencyBinsPositiveHalf, XMagnitudeAfterLowpassFilter, label="Widmo po LP (60 Hz)", linestyle="dashed")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Porównanie widm przed i po filtracji LP")
plt.legend()
plt.grid()
plt.show()

# Wypisanie maksymalnej i minimalnej wartośći amplitudy w celach
# sprawdzania poprawności kodu
print(f'Min: {np.min(signalValues)}, Max: {np.max(signalValues)}')

#ZADANIE3###############################################################################################################

def butterworthHighpassFilter(data, cutoffFrequency, order=4):
    nyquistFrequency = 0.5 * fs
    normalizedCutoffFrequency = cutoffFrequency / nyquistFrequency
    b, a = butter(order, normalizedCutoffFrequency, btype='high', analog=False)
    return filtfilt(b, a, data)

signalAfterHighpassFilter = butterworthHighpassFilter(signalAfterLowpassFilter, lowerBound)

# Wykres sygnału po filtrze pasmowym w zakresie od 5 do 60 Hz
plt.figure(figsize=(12, 4))
plt.plot(time, signalValues, label="Oryginalny sygnał")
plt.plot(time, signalAfterHighpassFilter, label="Po filtrze pasmowym (5-60 Hz)", linestyle="dashed")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Sygnał po filtracji pasmowej (5-60 Hz)")
plt.legend()
plt.grid()
plt.show()

# Widmo sygnału po filtracji pasmowej w zakresie od 5 do 60 Hz
XAfterBandpassFilter = fft(signalAfterHighpassFilter)
XMagnitudeAfterBandpassFilter = np.abs(XAfterBandpassFilter)[:N // 2]

plt.figure(figsize=(12, 4))
plt.plot(frequencyBinsPositiveHalf, XMagnitude, label="Widmo przed filtracją", color="red", alpha=0.6)
plt.plot(frequencyBinsPositiveHalf, XMagnitudeAfterBandpassFilter, label="Widmo po filtracji pasmowej (5-60 Hz)", linestyle="dashed", color="blue")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Porównanie widm przed i po filtracji pasmowej (5-60 Hz)")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(frequencyBinsPositiveHalf, XMagnitudeAfterBandpassFilter, label="Widmo po BP (5-60 Hz)")
plt.xlabel("Częstotliwość (Hz)")
plt.ylabel("Amplituda")
plt.title("Widmo sygnału po filtracji pasmowej (5-60 Hz)")
plt.legend()
plt.grid()
plt.show()

# Różnica między sygnałami przed i po filtracji
diff = signalValues - signalAfterHighpassFilter

plt.figure(figsize=(12, 4))
plt.plot(time, diff, label="Różnica przed i po filtracji")
plt.xlabel("Czas (s)")
plt.ylabel("Amplituda")
plt.title("Różnica sygnału przed i po filtracji")
plt.legend()
plt.grid()
plt.show()
