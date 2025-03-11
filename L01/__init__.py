import matplotlib.pyplot as plt
import numpy as np

PLIK = "./egk1.txt"
FS = 1000
XSCALE = 1
YSCALE = 1
BEGIN = 10
END = 20
RANGE = 12

for i in range(1, RANGE):


xpoints = np.array([1, 8])
ypoints = np.array([3, 10])

plt.plot(xpoints, ypoints)
plt.show()