import numpy as np
import os
from skimage import io, transform
from skimage.metrics import structural_similarity
import matplotlib.pyplot as plt
import glob

def load_image(path):
    """Загружает изображение с диска в виде массива NumPy."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл '{path}' не найден.")
    return io.imread(path)

def to_grayscale(image):
    """Преобразует изображение в градации серого и uint8."""
    if image.ndim == 2:
        return (image / image.max() * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
    if image.shape[2] == 4:
        image = image[:, :, :3]  # отбрасываем альфа-канал
    gray = np.dot(image[..., :3], [0.2989, 0.5870, 0.1140])
    return np.clip(gray, 0, 255).astype(np.uint8)

def resize_image(image, new_shape):
    """Меняет размер изображения до new_shape (h,w)."""
    resized = transform.resize(image, new_shape, preserve_range=True, anti_aliasing=True)
    return np.clip(resized, 0, 255).astype(np.uint8)

def compute_mse(img1, img2):
    """Вычисляет среднеквадратичную ошибку."""
    diff = img1.astype(np.float64) - img2.astype(np.float64)
    return np.mean(diff ** 2)

def compute_ssim(img1, img2):
    """Вычисляет индекс структурного сходства (SSIM)."""
    return structural_similarity(img1, img2, data_range=img1.max() - img1.min())

def compare_images(img1, img2, threshold=0.8):
    """Сравнивает два изображения по SSIM и MSE и показывает результат."""
    h1, w1 = img1.shape
    h2, w2 = img2.shape
    if (h1, w1) != (h2, w2):
        img2 = resize_image(img2, (h1, w1))

    mse_val = compute_mse(img1, img2)
    ssim_val = compute_ssim(img1, img2)

    print(f"SSIM: {ssim_val:.3f}")
    print(f"MSE : {mse_val:.3f}")
    print("Вердикт:", "Похожие ✅" if ssim_val > threshold else "Разные ❌")

    # Визуализация
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(img1, cmap="gray"); axes[0].set_title("Изображение 1"); axes[0].axis("off")
    axes[1].imshow(img2, cmap="gray"); axes[1].set_title("Изображение 2"); axes[1].axis("off")
    fig.suptitle(f"SSIM: {ssim_val:.3f}   MSE: {mse_val:.2f}")
    plt.show()

def find_file(base_name):
    """Ищет файл с заданным базовым именем и любым расширением."""
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]:
        files = glob.glob(base_name + ext)
        if files:
            return files[0]
    raise FileNotFoundError(f"Файл {base_name}.[png/jpg/jpeg/bmp/tif] не найден в папке.")

if __name__ == "__main__":
    try:
        path1 = find_file("image1")
        path2 = find_file("image2")
        img1 = to_grayscale(load_image(path1))
        img2 = to_grayscale(load_image(path2))
        compare_images(img1, img2, threshold=0.8)
    except Exception as e:
        print("Ошибка:", e)