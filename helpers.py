from pathlib import Path
from PIL import Image


def ensure_dir(path: str) -> None:
    """Создает директорию, если она не существует"""
    Path(path).mkdir(parents=True, exist_ok=True)


def save_crop(image: Image.Image, bbox, dest_path: str) -> tuple:
    """
    Сохраняет обрезанное изображение по указанным координатам bbox

    Args:
        image: Исходное изображение
        bbox: Координаты (x_min, y_min, x_max, y_max) в пикселях
        dest_path: Путь для сохранения обрезанного изображения

    Returns:
        tuple: (путь к сохраненному файлу, размер изображения (ширина, высота))
    """
    x0, y0, x1, y1 = map(int, bbox)
    crop = image.crop((x0, y0, x1, y1))
    crop.save(dest_path)
    return dest_path, crop.size