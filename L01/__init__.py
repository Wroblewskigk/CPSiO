import matplotlib.pyplot as plt
import numpy as np

PLIK = "./ekg1.txt"
FS = 1000
XSCALE = 1
YSCALE = 1
BEGIN = 1000
END = 2000
RANGE = 12

# Z1
ekg1 = []  # Store all rows
with open(PLIK, "r") as file:
    for line in file:
        numbers = list(map(int, line.split()))  # Convert split strings into integers
        ekg1.append(numbers)  # Store the row

ekg1 = ekg1[BEGIN:END]
for columnIndex in range(0, 11):
    ypoints = [row[columnIndex] for row in ekg1 if len(row) > columnIndex]
    xpoints = list(range(len(ypoints)))
    plt.plot(xpoints, ypoints)
    plt.xlabel("Fs [kHz]")
    plt.ylabel("To w czym mierzymy EKG")
    plt.title("Wykres sygnału EKG mierzony z częstotliwością 1kHz")
    plt.show()