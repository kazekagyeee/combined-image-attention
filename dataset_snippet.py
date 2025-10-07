import os
import zipfile
import pandas as pd
from pathlib import Path
import shutil

# === ПАРАМЕТРЫ ===
SOURCE_ZIP = "resources/products_dataset.zip"           # исходный архив
OUTPUT_ZIP = "dataset_snippet.zip"   # итоговый архив
EXTRACT_DIR = "tmp_dataset"
SNIPPET_DIR = "tmp_snippet"

TRAIN_LIMIT = 1000
TEST_LIMIT = 100

# === 1. Распаковка исходного архива ===
print("📂 Распаковываю архив...")
with zipfile.ZipFile(SOURCE_ZIP, "r") as zf:
    zf.extractall(EXTRACT_DIR)

# === 2. Пути ===
train_csv = Path(EXTRACT_DIR) / "train.csv"
test_csv = Path(EXTRACT_DIR) / "test.csv"
train_dir = Path(EXTRACT_DIR) / "train" / "train"
test_dir = Path(EXTRACT_DIR) / "test" / "test"

# === 3. Загружаем CSV и выбираем сниппеты ===
print("📑 Загружаю CSV...")
train_df = pd.read_csv(train_csv)
test_df = pd.read_csv(test_csv)

train_snip = train_df.head(TRAIN_LIMIT)
test_snip = test_df.head(TEST_LIMIT)

# === 4. Создаём структуру для сниппета ===
(Path(SNIPPET_DIR) / "train" / "train").mkdir(parents=True, exist_ok=True)
(Path(SNIPPET_DIR) / "test" / "test").mkdir(parents=True, exist_ok=True)

# === 5. Копируем изображения ===
def copy_images(df, src_dir, dst_dir):
    for fname in df['name'].values:  # предполагается, что в csv есть столбец "image"
        src_path = src_dir / fname
        dst_path = dst_dir / fname
        if src_path.exists():
            shutil.copy(src_path, dst_path)
        else:
            print(f"⚠️ Файл не найден: {src_path}")

print("🖼 Копирую изображения train...")
copy_images(train_snip, train_dir, Path(SNIPPET_DIR) / "train" / "train")

print("🖼 Копирую изображения test...")
copy_images(test_snip, test_dir, Path(SNIPPET_DIR) / "test" / "test")

# === 6. Сохраняем укороченные CSV ===
train_snip.to_csv(Path(SNIPPET_DIR) / "train.csv", index=False)
test_snip.to_csv(Path(SNIPPET_DIR) / "test.csv", index=False)

# === 7. Упаковываем результат ===
print("📦 Архивирую результат...")
with zipfile.ZipFile(OUTPUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, _, files in os.walk(SNIPPET_DIR):
        for file in files:
            full_path = Path(root) / file
            rel_path = full_path.relative_to(SNIPPET_DIR)
            zf.write(full_path, rel_path)

print(f"✅ Готово! Архив сохранён как: {OUTPUT_ZIP}")
