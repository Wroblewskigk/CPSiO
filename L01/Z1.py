import numpy as np
import matplotlib.pyplot as plt

def load_ecg(filename, fs):
    data = np.loadtxt(filename)
    if data.ndim == 1:
        time = np.arange(len(data)) / fs
        return time, data  # Pojedynczy kanał
    else:
        time = np.arange(data.shape[0]) / fs
        return time, data  # Wielokanałowy sygnał


def plot_ecg(time, signal, start=0, end=None, title="EKG", ymin=None, ymax=None):
    if end is None:
        end = time[-1]
    mask = (time >= start) & (time <= end)

    # Check if the signal is single-channel or multi-channel
    if signal.ndim == 1:
        n_channels = 1  # Single-channel
    else:
        n_channels = signal.shape[1]  # Multi-channel

    # Create a figure with subplots for multi-channel, or a single plot for single-channel
    plt.figure(figsize=(12, 4 * n_channels))

    if n_channels == 1:
        # For single-channel signal, plot it on one plot
        plt.subplot(1, 1, 1)
        plt.plot(time[mask], signal[mask])  # Plot the single channel
        plt.xlabel("Czas (s)")
        plt.ylabel("Amplituda")
        plt.title(f"{title} ({start}-{end} s)")
        plt.grid()

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
    else:
        # For multi-channel signal, plot each channel in a separate subplot
        for i in range(n_channels):
            plt.subplot(n_channels, 1, i + 1)  # One subplot for each channel
            plt.plot(time[mask], signal[mask, i])  # Plot each channel
            plt.xlabel("Czas (s)")
            plt.ylabel(f"Amplituda (Lead {i + 1})")
            plt.title(f"{title} ({start}-{end} s) - Lead {i + 1}")
            plt.grid()

            if ymin is not None and ymax is not None:
                plt.ylim(ymin, ymax)

    plt.tight_layout()
    plt.show()


def plot_all_channels_on_one_plot(time, signal, start=0, end=None, title="EKG", ymin=None, ymax=None):
    """Plot all channels of the multi-channel signal on one plot with optional Y-axis scaling."""
    if end is None:
        end = time[-1]
    mask = (time >= start) & (time <= end)

    plt.figure(figsize=(12, 6))

    n_channels = signal.shape[1] if signal.ndim > 1 else 1

    # Plot all channels on the same plot
    for i in range(n_channels):
        plt.plot(time[mask], signal[mask, i], label=f"Lead {i + 1}" if n_channels > 1 else "Signal")

    plt.xlabel("Czas (s)")
    plt.ylabel("Amplituda")
    plt.title(f"{title} ({start}-{end} s) - Wszystkie kanały")
    plt.grid()

    # Apply Y-axis limits if specified
    if ymin is not None and ymax is not None:
        plt.ylim(ymin, ymax)

    plt.show()

def save_segment(time, signal, start, end, filename, fmt):
    mask = (time >= start) & (time <= end)
    segment = signal[mask]
    np.savetxt(filename, segment, fmt=fmt)
    print(f"Zapisano wycinek do {filename}")


# Wczytanie plików
ekg1_file = "ekg1.txt"
ekg100_file = "ekg100.txt"
fs1 = 1000
fs100 = 360

time1, signal1 = load_ecg(ekg1_file, fs1)
time100, signal100 = load_ecg(ekg100_file, fs100)

# Wizualizacja całego sygnału dla obu plików
plot_ecg(time1, signal1, title="EKG1 - Wielokanałowy")
plot_ecg(time100, signal100, title="EKG100 - Jednokanałowy")

# Wizualizacja fragmentu (np. 1-3 sekundy)
plot_ecg(time1, signal1, start=1, end=3, title="EKG1 - Fragment 1-3s", ymin=None, ymax=None)
plot_ecg(time100, signal100, start=1, end=3, title="EKG100 - Fragment 1-3s", ymin=None, ymax=None)

# Wizualizacja wszystkich kanałów EKG1 na jednym wykresie
plot_all_channels_on_one_plot(time1, signal1, start=1, end=3, title="EKG1 - Wszystkie kanały", ymin=None, ymax=None)

# Zapis fragmentu do pliku
save_segment(time1, signal1, start=1, end=3, filename="segment_ekg1.txt", fmt="%d")
save_segment(time100, signal100, start=1, end=3, filename="segment_ekg100.txt", fmt="%.7e")
