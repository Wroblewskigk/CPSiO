from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import os


def load_and_display_image(filename):
    image = Image.open(filename).convert('L')
    image.show()
    return np.array(image)


def plot_gray_level_profile(image, coord, orientation='horizontal'):
    if orientation == 'horizontal':
        profile = image[coord, :]
    else:
        profile = image[:, coord]

    plt.plot(profile, 'k-')
    plt.title(f'Profil poziomu szarości ({orientation})')
    plt.xlabel('Pozycja piksela')
    plt.ylabel('Poziom szarości')
    plt.show()


def select_and_save_subimage(image, x1, y1, x2, y2, output_filename):
    height, width = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x1 >= x2 or y1 >= y2:
        print("Błąd: Niepoprawne współrzędne podobszaru.")
        return

    sub_image = Image.fromarray(image[y1:y2, x1:x2])
    output_path = os.path.join(os.getcwd(), output_filename)
    sub_image.save(output_path)
    print(f"Podobraz zapisany jako: {output_path}")
    sub_image.show()


def transformation_T(image, func):
    vectorized_func = np.vectorize(func)
    transformed_image = vectorized_func(image).astype(np.uint8)
    transformed_pil = Image.fromarray(transformed_image)
    transformed_pil.show()
    return transformed_image


if __name__ == "__main__":
    filename = input("Podaj nazwę pliku obrazu: ")
    image = load_and_display_image("./Images/" + filename)

    if image is not None:
        coord = int(input("Podaj współrzędną dla wykresu poziomu szarości: "))
        orientation = input("Podaj orientację (horizontal/vertical): ")
        plot_gray_level_profile(image, coord, orientation)

        x1, y1 = map(int, input("Podaj współrzędne (x1, y1) lewego górnego rogu podobszaru: ").split())
        x2, y2 = map(int, input("Podaj współrzędne (x2, y2) prawego dolnego rogu podobszaru: ").split())
        output_filename = input("Podaj nazwę pliku dla zapisanego podobszaru: ")
        select_and_save_subimage(image, x1, y1, x2, y2, output_filename)

        print("Przekształcenie obrazu: T(r) = 255 - r")
        transformed_image = transformation_T(image, lambda r: 255 - r)