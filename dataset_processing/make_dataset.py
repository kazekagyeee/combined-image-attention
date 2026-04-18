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


# ── Настройки ─────────────────────────────────────────────────────────────────
MIN_CONTEXT_LEN = 400  # Минимальное количество символов контекста
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
    Извлекает изображения и расширенный контекст вокруг них из PDF.
    """
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(path)
    images_info = []
    all_text_blocks = [] # Список всех текстовых блоков во всем PDF
    all_text = ""        # Полный текст PDF (сохраняем для обратной совместимости)

    # 1. Сначала собираем ВСЕ текстовые блоки из всего документа
    # Это нужно для "межстраничного" поиска контекста
    for page_num in range(len(doc)):
        page = doc[page_num]
        page_text = page.get_text()
        all_text += page_text + "\n"
        
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] == 0: # только текст
                all_text_blocks.append({
                    "page": page_num,
                    "bbox": b[:4],
                    "text": b[4],
                    "y_mid": (b[1] + b[3]) / 2
                })

    # 2. Собираем информацию об изображениях
    for page_num in range(len(doc)):
        page = doc[page_num]
        for img_info in page.get_images(full=True):
            xref = img_info[0]
            rects = page.get_image_rects(xref)
            if not rects: continue

            img_rect = rects[0]
            img_cy = (img_rect.y0 + img_rect.y1) / 2

            # Находим "опорный" блок на этой же странице (самый близкий по Y)
            anchor_idx = -1
            min_y_dist = float("inf")
            
            for i, b in enumerate(all_text_blocks):
                if b["page"] == page_num:
                    dist = abs(b["y_mid"] - img_cy)
                    if dist < min_y_dist:
                        min_y_dist = dist
                        anchor_idx = i
            
            # Если на странице нет текста, ищем ближайший по индексу страницы блок
            if anchor_idx == -1 and all_text_blocks:
                # Находим первый блок на следующей странице или последний на предыдущей
                for i, b in enumerate(all_text_blocks):
                    if b["page"] > page_num:
                        anchor_idx = i
                        break
                if anchor_idx == -1: # Значит все блоки раньше
                    anchor_idx = len(all_text_blocks) - 1

            # 3. Расширяем контекст (вверх и вниз), пока не наберем MIN_CONTEXT_LEN
            context_parts = []
            if anchor_idx != -1:
                start_i = anchor_idx
                end_i = anchor_idx
                
                # Добавляем центральный блок
                current_text = all_text_blocks[anchor_idx]["text"]
                
                # Расширяем в обе стороны
                while len(current_text) < MIN_CONTEXT_LEN:
                    expanded = False
                    # Пробуем взять блок выше
                    if start_i > 0:
                        start_i -= 1
                        current_text = all_text_blocks[start_i]["text"].rstrip() + "\n" + current_text.lstrip()
                        expanded = True
                    
                    if len(current_text) >= MIN_CONTEXT_LEN: break
                    
                    # Пробуем взять блок ниже
                    if end_i < len(all_text_blocks) - 1:
                        end_i += 1
                        current_text = current_text.rstrip() + "\n" + all_text_blocks[end_i]["text"].lstrip()
                        expanded = True
                    
                    if not expanded: break
                
                final_text = current_text.strip()
            else:
                final_text = ""

            # Байты изображения
            base_image = doc.extract_image(xref)
            images_info.append({
                "page": page_num,
                "xref": xref,
                "ext": base_image["ext"],
                "bytes": base_image["image"],
                "text": final_text,
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
