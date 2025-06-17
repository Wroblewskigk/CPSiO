import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Foldery
INPUT_DIR = "./Images"
OUTPUT_DIR = "./Images-converted-Z5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# 1. Wczytanie i wyświetlenie obrazu
# ----------------------------
def load_and_display_image(filename):
    image_path = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(image_path):
        print(f"Błąd: Nie znaleziono pliku {image_path}")
        return None

    image = Image.open(image_path).convert('L')
    image.show()
    return np.array(image)

# ----------------------------
# 2. Wykres poziomu szarości
# ----------------------------
def plot_gray_levels(image, coord, orientation='horizontal'):
    if orientation == 'horizontal':
        profile = image[coord, :]
    elif orientation == 'vertical':
        profile = image[:, coord]
    else:
        print("Nieprawidłowa orientacja.")
        return

    plt.plot(profile, 'k-')
    plt.title(f'Profil poziomu szarości ({orientation}), współrzędna={coord}')
    plt.xlabel('Pozycja piksela')
    plt.ylabel('Poziom szarości')
    plt.grid(True)
    plt.tight_layout()

    try:
        plt.show()
    except:
        plt.savefig(os.path.join(OUTPUT_DIR, f"profile_{orientation}_{coord}.png"))
        print("Zapisano wykres poziomu szarości.")

# ----------------------------
# 3. Wycinanie podobrazu
# ----------------------------
def crop_image(image, x1, y1, x2, y2, output_filename):
    height, width = image.shape
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)

    if x1 >= x2 or y1 >= y2:
        print("Błąd: Niepoprawne współrzędne podobszaru.")
        return

    cropped = Image.fromarray(image[y1:y2, x1:x2])
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    cropped.save(output_path)
    print(f"Podobraz zapisany jako: {output_path}")
    cropped.show()

# ----------------------------
# 4. Przekształcenie T(r) = 255 - r
# ----------------------------
def transformation_T(image, function):
    vectorized = np.vectorize(function)
    transformed = vectorized(image).astype(np.uint8)
    img = Image.fromarray(transformed)
    img.show()
    return transformed

# ----------------------------
# Główna funkcja
# ----------------------------
if __name__ == "__main__":
    filename = input("Podaj nazwę pliku obrazu: ")
    image = load_and_display_image(filename)

    if image is not None:
        try:
            coord = int(input("Podaj współrzędną dla wykresu poziomu szarości: "))
            orientation = input("Podaj orientację (horizontal/vertical): ").strip().lower()

            if orientation not in ['horizontal', 'vertical']:
                print("Niepoprawna orientacja. Wybierz 'horizontal' lub 'vertical'.")
            elif (orientation == 'horizontal' and (coord < 0 or coord >= image.shape[0])) or \
                 (orientation == 'vertical' and (coord < 0 or coord >= image.shape[1])):
                print("Błąd: współrzędna poza zakresem obrazu.")
            else:
                plot_gray_levels(image, coord, orientation)

            x1, y1 = map(int, input("Podaj współrzędne (x1, y1) lewego górnego rogu podobszaru: ").split())
            x2, y2 = map(int, input("Podaj współrzędne (x2, y2) prawego dolnego rogu podobszaru: ").split())
            output_filename = input("Podaj nazwę pliku do zapisu podobszaru (np. Z05_crop.tif): ")
            crop_image(image, x1, y1, x2, y2, output_filename)

            print("Wykonuję przekształcenie T(r) = 255 - r ...")
            transformed = transformation_T(image, lambda r: 255 - r)

            save_option = input("Czy zapisać wynik przekształcenia? (tak/nie): ").strip().lower()
            if save_option == 'tak':
                out_name = input("Podaj nazwę pliku (np. Z05_odwrocony.tif): ")
                Image.fromarray(transformed).save(os.path.join(OUTPUT_DIR, out_name))
                print(f"Zapisano wynik w: {os.path.join(OUTPUT_DIR, out_name)}")

        except Exception as e:
            print("Wystąpił błąd:", e)
