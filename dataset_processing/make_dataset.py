"""
make_dataset.py

Распаковывает dataset_raw.rar и извлекает пары (изображение, текст) из всех PDF-файлов.
Изображения и .txt-файлы с контекстом сохраняются рядом в ../images,
чтобы pipeline.py мог немедленно их обработать.

Запуск: python dataset_processing/make_dataset.py
        (из корня проекта)
"""

import os
import math
import glob
import shutil
import subprocess

import fitz  # PyMuPDF


# ── Пути ──────────────────────────────────────────────────────────────────────
# Директория, в которую распакуем RAR
SCRIPT_DIR    = os.path.dirname(os.path.abspath(__file__))
RAR_PATH      = os.path.join(SCRIPT_DIR, "dataset_raw.rar")
TEMP_EXTRACT  = os.path.join(SCRIPT_DIR, "dataset_temp")

# Куда складываем пары (изображение + .txt) — сразу в images/ корня проекта
PROJECT_ROOT  = os.path.dirname(SCRIPT_DIR)
OUT_DIR       = os.path.join(PROJECT_ROOT, "images")


# ── Основная функция извлечения ────────────────────────────────────────────────
def extract_pdf_data(path, skip_first_images=0, skip_last_images=0,
                     out_dir="../images", pdf_index=0):
    """
    Извлекает изображения и ближайший к ним текст из PDF.
    Изображения и парные .txt-файлы сохраняются прямо в out_dir
    (без вложенных папок images/ и texts/), чтобы pipeline.py находил их по
    схеме img.stem + '.txt'.

    Параметры
    ---------
    path              : путь к PDF
    skip_first_images : сколько первых изображений пропустить
    skip_last_images  : сколько последних изображений пропустить
    out_dir           : куда сохранять файлы
    pdf_index         : уникальный индекс PDF (для именования файлов)

    Возвращает
    ----------
    (all_text, image_text_pairs)
    """
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(path)
    all_text = ""
    images_info = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        all_text += page.get_text() + "\n"

        # Текстовые блоки страницы
        blocks = page.get_text("blocks")
        text_blocks = [b for b in blocks if b[6] == 0]  # block_type == 0 → текст

        # Изображения на странице
        for img_info in page.get_images(full=True):
            xref = img_info[0]

            rects = page.get_image_rects(xref)
            if not rects:
                continue

            img_rect = rects[0]
            img_cx = (img_rect.x0 + img_rect.x1) / 2
            img_cy = (img_rect.y0 + img_rect.y1) / 2

            # Ближайший текстовый блок по Евклидову расстоянию
            closest_text = ""
            min_dist = float("inf")
            for tb in text_blocks:
                tb_cx = (tb[0] + tb[2]) / 2
                tb_cy = (tb[1] + tb[3]) / 2
                dist = math.hypot(img_cx - tb_cx, img_cy - tb_cy)
                if dist < min_dist:
                    min_dist = dist
                    closest_text = tb[4]

            # Байты изображения
            base_image  = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext   = base_image["ext"]

            images_info.append({
                "page":  page_num,
                "xref":  xref,
                "ext":   image_ext,
                "bytes": image_bytes,
                "text":  closest_text.strip(),
            })

    doc.close()

    # Применяем фильтр пропусков
    total_imgs = len(images_info)
    start_idx  = skip_first_images
    end_idx    = total_imgs - skip_last_images if skip_last_images else total_imgs

    if start_idx >= end_idx or end_idx <= 0:
        filtered_images = []
    else:
        filtered_images = images_info[start_idx:end_idx]

    image_text_pairs = []

    for i, img_data in enumerate(filtered_images):
        # Имена файлов: pdf_<idx>_image_<i>.<ext>  и  pdf_<idx>_image_<i>.txt
        base_name   = f"pdf_{pdf_index}_image_{i}"
        img_filename = base_name + "." + img_data["ext"]
        txt_filename = base_name + ".txt"

        img_path = os.path.join(out_dir, img_filename)
        txt_path = os.path.join(out_dir, txt_filename)

        with open(img_path, "wb") as f:
            f.write(img_data["bytes"])

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(img_data["text"])

        image_text_pairs.append({
            "image_path":   img_path,
            "text_path":    txt_path,
            "context_text": img_data["text"],
        })

    return all_text, image_text_pairs


# ── Точка входа ────────────────────────────────────────────────────────────────
def process_dataset_raw():
    if not os.path.exists(RAR_PATH):
        print(f"Ошибка: файл {RAR_PATH} не найден.")
        return

    # 1. Распаковываем RAR
    print(f"Распаковываю {RAR_PATH} → {TEMP_EXTRACT} ...")
    os.makedirs(TEMP_EXTRACT, exist_ok=True)
    subprocess.run(["tar", "xf", RAR_PATH, "-C", TEMP_EXTRACT], check=True)

    # 2. Находим все PDF
    pdf_files = sorted(glob.glob(os.path.join(TEMP_EXTRACT, "**", "*.pdf"), recursive=True))
    print(f"Найдено PDF-файлов: {len(pdf_files)}")

    # 3. Очищаем целевую директорию (images/)
    if os.path.exists(OUT_DIR):
        print(f"Очищаю {OUT_DIR} ...")
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # 4. Извлекаем данные из каждого PDF
    total_pairs = 0
    for idx, pdf_file in enumerate(pdf_files):
        print(f"  [{idx + 1}/{len(pdf_files)}] {os.path.basename(pdf_file)}")
        _, pairs = extract_pdf_data(
            pdf_file,
            skip_first_images=0,
            skip_last_images=0,
            out_dir=OUT_DIR,
            pdf_index=idx,
        )
        total_pairs += len(pairs)
        print(f"          → извлечено пар: {len(pairs)}")

    print(f"\n✅ Готово. Всего пар (изображение + текст): {total_pairs}")
    print(f"   Файлы сохранены в: {OUT_DIR}")


if __name__ == "__main__":
    process_dataset_raw()
