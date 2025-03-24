import matplotlib.pyplot as plt

filename = "./ekg1.txt"
fs = 1000
begin = 1000
end = 2000

ekg1 = []
with open(filename, "r") as file:
    for line in file:
        numbers = list(map(int, line.split()))
        ekg1.append(numbers)

ekg1 = ekg1[begin:end]
for columnIndex in range(0, 1):
    yPoints = [row[columnIndex] for row in ekg1 if len(row) > columnIndex]
    xPoints = list(range(len(yPoints)))
    plt.plot(xPoints, yPoints)
    plt.xlabel("Fs [kHz]")
    plt.ylabel("Amplituda")
    plt.title("Wykres sygnału EKG mierzony z częstotliwością 1kHz")
    plt.show()