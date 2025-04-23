import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


#ZADANIE1###############################################################################################################

def loadAndDisplayImage(filename):
    image = Image.open(filename).convert('L')
    image.show()
    return np.array(image)

#ZADANIE2###############################################################################################################

def plotGrayLevels(image, coordinates, orientation='horizontal'):
    if orientation == 'horizontal':
        profile = image[coordinates, :]
    else:
        profile = image[:, coordinates]

    plt.plot(profile, 'k-')
    plt.title(f'Profil poziomu szarości ({orientation})')
    plt.xlabel('Pozycja piksela')
    plt.ylabel('Poziom szarości')
    plt.show()

#ZADANIE3###############################################################################################################

def cropImage(image, x1, y1, x2, y2, outputFilename):
    height, width = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x1 >= x2 or y1 >= y2:
        print("Błąd: Niepoprawne współrzędne podobszaru.")
        return

    croppedImage = Image.fromarray(image[y1:y2, x1:x2])
    croppedImagePath = os.path.join(os.getcwd(), outputFilename)
    croppedImage.save(croppedImagePath)
    print(f"Podobraz zapisany jako: {croppedImagePath}")
    croppedImage.show()


def transformationT(image, function):
    # Wektoryzuje podaną funkcję, umożliwiając jej zastosowanie do każdego elementu tablicy
    vectorizedFunction = np.vectorize(function)
    # Stosuje funkcję do każdego piksela obrazu i konwertuje wynik do 8-bitowych wartości całkowitych
    imageTransformed = vectorizedFunction(image).astype(np.uint8)
    # Tworzy obiekt obrazu PIL z przekształconej tablicy
    transformedImageToDisplay = Image.fromarray(imageTransformed)
    transformedImageToDisplay.show()
    # Zwraca przekształconą tablicę
    return imageTransformed

if __name__ == "__main__":
    filename = input("Podaj nazwę pliku obrazu: ")
    image = loadAndDisplayImage("./Images/" + filename)

    if image is not None:
        coordinates = int(input("Podaj współrzędną dla wykresu poziomu szarości: "))
        orientation = input("Podaj orientację (horizontal/vertical): ")
        plotGrayLevels(image, coordinates, orientation)

        x1, y1 = map(int, input("Podaj współrzędne (x1, y1) lewego górnego rogu podobszaru: ").split())
        x2, y2 = map(int, input("Podaj współrzędne (x2, y2) prawego dolnego rogu podobszaru: ").split())
        outputFilename = input("Podaj nazwę pliku dla zapisanego podobszaru: ")
        cropImage(image, x1, y1, x2, y2, outputFilename)

        print("Przekształcenie obrazu: T(r) = 255 - r")
        imageAfterTransformationT = transformationT(image, lambda r: 255 - r)