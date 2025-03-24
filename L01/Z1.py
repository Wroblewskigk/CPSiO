import numpy as np
import matplotlib.pyplot as plt

# Zmienne potrzebne do poprawnego działania programu
fileEKG1 = "ekg1.txt"
fileEKG100 = "ekg100.txt"
fs1 = 1000
fs100 = 360

def loadSignal(filename, fs):
    data = np.loadtxt(filename)
    # W przypadku sygnału jednokanałowego
    if data.ndim == 1:
        time = np.arange(len(data)) / fs
        return time, data
    # W przypadku sygnału wielokanałowego
    else:
        time = np.arange(data.shape[0]) / fs
        return time, data


def plotSignal(time, signal, signalStart=0, signalEnd=None, title="EKG", yMinValue=None, yMaxValue=None):
    # Jeżeli nie sprecyzowano, do której sekundy wyświetlić sygnał: wyświetl go w całości
    if signalEnd is None:
        signalEnd = time[-1]

    # Tworzenie maski z zakresem pomiędzy signalStart i signalEnd
    mask = (time >= signalStart) & (time <= signalEnd)
    print(mask)

    # Sprawdzanie, czy sygnał jest jedno, czy wielokanałowy
    if signal.ndim == 1:
        # Sygnał jednokanałowy
        numberOfChannels = 1
    else:
        # Sygnał wielokanałowy
        numberOfChannels = signal.shape[1]

    plt.figure(figsize=(12, 4 * numberOfChannels))

    if numberOfChannels == 1:
        plt.subplot(1, 1, 1)
        plt.plot(time[mask], signal[mask])
        plt.xlabel("Czas (s)")
        plt.ylabel("Amplituda")
        plt.title(f"{title} ({signalStart}-{signalEnd} s)")
        plt.grid()
        if yMinValue is not None and yMaxValue is not None:
            plt.ylim(yMinValue, yMaxValue)
    else:
        for i in range(numberOfChannels):
            plt.subplot(numberOfChannels, 1, i + 1)
            plt.plot(time[mask], signal[mask, i])
            plt.xlabel("Czas (s)")
            plt.ylabel(f"Amplituda (Lead {i + 1})")
            plt.title(f"{title} ({signalStart}-{signalEnd} s) - Kanał {i + 1}")
            plt.grid()
            if yMinValue is not None and yMaxValue is not None:
                plt.ylim(yMinValue, yMaxValue)

    plt.tight_layout()
    plt.show()


def plotAllSignal(time, signal, signalStart=0, signalEnd=None, title="EKG", yMinValue=None, yMaxValue=None):
    # Jeżeli nie sprecyzowano, do której sekundy wyświetlić sygnał: wyświetl go w całości
    if signalEnd is None:
        signalEnd = time[-1]
    # Tworzenie maski z zakresem pomiędzy signalStart i signalEnd
    mask = (time >= signalStart) & (time <= signalEnd)

    plt.figure(figsize=(12, 6))

    numberOfChannels = signal.shape[1] if signal.ndim > 1 else 1

    for i in range(numberOfChannels):
        plt.plot(time[mask], signal[mask, i], label=f"Kanał {i + 1}" if numberOfChannels > 1 else "Signal")

    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.title(f"{title} ({signalStart}-{signalEnd} s) - Wszystkie kanały")
    plt.grid()

    if yMinValue is not None and yMaxValue is not None:
        plt.ylim(yMinValue, yMaxValue)
    plt.show()

def savePlotSegmentToTxt(time, signal, signalStart, signalEnd, filename, fmt):
    # Tworzenie maski z zakresem pomiędzy signalStart i signalEnd
    mask = (time >= signalStart) & (time <= signalEnd)
    segment = signal[mask]
    np.savetxt(filename, segment, fmt=fmt)
    print(f"Zapisano wycinek do {filename}")

timeEKG1, signalEKG1 = loadSignal(fileEKG1, fs1)
timeEKG100, signalEKG100 = loadSignal(fileEKG100, fs100)

# Wizualizacja całego sygnału dla obu plików
plotSignal(timeEKG1, signalEKG1, title="EKG1 - Wielokanałowy")
plotSignal(timeEKG100, signalEKG100, title="EKG100 - Jednokanałowy")

# Wizualizacja zadanego przez signalStart i signalEnd segmentu sygnału
plotSignal(timeEKG1, signalEKG1, signalStart=1, signalEnd=2, title="EKG1 - Wielokanałowy", yMinValue=None, yMaxValue=None)
plotSignal(timeEKG100, signalEKG100, signalStart=1, signalEnd=2, title="EKG100 - Jednokanałowy", yMinValue=None, yMaxValue=None)

# Wizualizacja zadanego przez signalStart i signalEnd segmentu sygnałów wszystkich kanałów EKG1 na jednym wykresie
plotAllSignal(timeEKG1, signalEKG1, signalStart=1, signalEnd=2, title="Wszystkie kanały sygnału EKG1", yMinValue=None, yMaxValue=None)

# Zapis zadanych przez signalStart i signalEnd segmentów sygnałów do plików
savePlotSegmentToTxt(timeEKG1, signalEKG1, signalStart=1, signalEnd=2, filename="segment_ekg1.txt", fmt="%d")
savePlotSegmentToTxt(timeEKG100, signalEKG100, signalStart=1, signalEnd=2, filename="segment_ekg100.txt", fmt="%.7e")
