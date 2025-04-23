from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram(img_array):
    # Liczenie histogramu
    hist, bins = np.histogram(img_array.flatten(), 256, [0,256])
    cdf = hist.cumsum()  # Dystrybuanta
    cdf_masked = np.ma.masked_equal(cdf, 0)  # Pomijanie zer
    # Normalizacja CDF
    cdf_min = cdf_masked.min()
    cdf_max = cdf_masked.max()
    cdf_masked = (cdf_masked - cdf_min) * 255 / (cdf_max - cdf_min)
    cdf_final = np.ma.filled(cdf_masked, 0).astype('uint8')
    # Mapowanie wartości
    img_eq = cdf_final[img_array]
    return img_eq

def plot_histograms(original, equalized, title):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title('Oryginał')
    axes[0,0].axis('off')
    axes[0,1].hist(original.flatten(), bins=256, color='blue')
    axes[0,1].set_title('Histogram oryginału')
    axes[1,0].imshow(equalized, cmap='gray')
    axes[1,0].set_title('Po wyrównaniu')
    axes[1,0].axis('off')
    axes[1,1].hist(equalized.flatten(), bins=256, color='green')
    axes[1,1].set_title('Histogram po wyrównaniu')
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()

def process_image(image_path):
    img = Image.open(image_path).convert('L')
    img_array = np.array(img)
    img_eq = equalize_histogram(img_array)
    plot_histograms(img_array, img_eq, image_path)

# Lista obrazów do przetworzenia
images = [
    'chest_xray.tif',
    'pollen-dark.tif',
    'pollen-ligt.tif',
    'pollen-lowcontrast.tif',
    'pout.tif',
    'spectrum.tif'
]

for img_path in images:
    try:
        print(f"Przetwarzanie: {img_path}")
        process_image(img_path)
    except Exception as e:
        print(f"Błąd przetwarzania {img_path}: {e}")
