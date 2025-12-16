import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np
from pathlib import Path
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class OptimizedEmbedder:
    def __init__(self, model_name="Qwen/Qwen2.5-VL-7B-Instruct"):
        """
        Оптимизированная версия эмбеддера
        """
        print(f"Загрузка модели {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model.eval()  # Переводим модель в режим оценки

        print(f"Модель загружена на {self.device}")
        print(f"Тип данных модели: {self.model.dtype}")

    def get_batch_embeddings(self, full_images, crop_images, texts):
        """
        Получение эмбеддингов для батча
        """
        embeddings = []

        for full_img, crop_img, text in zip(full_images, crop_images, texts):
            try:
                # Подготавливаем сообщение
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": text}
                        ]
                    }
                ]

                text_prompt = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )

                # Обрабатываем
                inputs = self.processor(
                    images=[full_img, crop_img],
                    text=text_prompt,
                    return_tensors="pt"
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True
                    )

                    last_hidden_state = outputs.hidden_states[-1]
                    embedding = last_hidden_state.mean(dim=1)
                    embedding = embedding.cpu().numpy()
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

                embeddings.append(embedding[0])

            except Exception as e:
                print(f"Ошибка: {e}")
                embeddings.append(None)

        return embeddings


def process_large_dataset(base_folder, crops_folder, output_json, batch_size=4):
    """
    Обработка больших наборов данных с батч-обработкой
    """
    print("Сканирование файлов...")

    # Находим пары в базовой папке
    base_path = Path(base_folder)
    crops_path = Path(crops_folder)

    # Получаем все пары (изображение + текст)
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif']
    base_pairs = []

    for ext in image_exts:
        for img_file in base_path.glob(f"*{ext}"):
            txt_file = base_path / f"{img_file.stem}.txt"
            if txt_file.exists():
                base_pairs.append((img_file, txt_file))

    if not base_pairs:
        print("Нет пар в базовой папке!")
        return []

    # Получаем все crop изображения
    crop_files = []
    for ext in image_exts:
        crop_files.extend(crops_path.glob(f"*{ext}"))

    if not crop_files:
        print("Нет crop изображений!")
        return []

    print(f"Найдено: {len(base_pairs)} пар, {len(crop_files)} crops")

    # Инициализируем эмбеддер
    embedder = OptimizedEmbedder()

    results = []
    total_combinations = len(base_pairs) * len(crop_files)

    print(f"Всего комбинаций для обработки: {total_combinations}")

    # Прогресс-бар
    pbar = tqdm(total=total_combinations, desc="Извлечение эмбеддингов")

    # Обрабатываем каждую пару с каждым crop
    for full_img_path, text_path in base_pairs:
        # Читаем текст один раз для этой пары
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except:
            text = ""

        if not text:
            pbar.update(len(crop_files))
            continue

        # Обрабатываем crops батчами
        for i in range(0, len(crop_files), batch_size):
            batch_crops = crop_files[i:i + batch_size]

            # Загружаем изображения для батча
            full_images = []
            crop_images = []
            texts = []
            crop_paths = []

            try:
                full_image = Image.open(full_img_path).convert('RGB')
            except:
                continue

            for crop_path in batch_crops:
                try:
                    crop_image = Image.open(crop_path).convert('RGB')

                    full_images.append(full_image)
                    crop_images.append(crop_image)
                    texts.append(text)
                    crop_paths.append(crop_path)

                except:
                    continue

            if not full_images:
                continue

            # Получаем эмбеддинги для батча
            batch_embeddings = embedder.get_batch_embeddings(full_images, crop_images, texts)

            # Сохраняем результаты
            for crop_path, embedding in zip(crop_paths, batch_embeddings):
                if embedding is not None:
                    record = {
                        "crop_path": str(crop_path.absolute()),
                        "orig_path": str(full_img_path.absolute()),
                        "bbox": [],
                        "score": 0.0,
                        "caption": text,
                        "prompt_used": "",
                        "area": 0.0,
                        "rel_size_coeff": 0.0,
                        "crop_wh": [],
                        "text_embedding": embedding.tolist()
                    }
                    results.append(record)

            # Обновляем прогресс-бар
            pbar.update(len(batch_crops))

            # Промежуточное сохранение каждые 100 записей
            if len(results) % 100 == 0:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)

    pbar.close()

    # Финальное сохранение
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Сохранено {len(results)} записей в {output_json}")

    return results


# КОНФИГУРАЦИЯ
BASE_FOLDER = "../images"  # Замените на ваш путь
CROPS_FOLDER = "../out"  # Замените на ваш путь
OUTPUT_JSON = "base_model_embeddings.json"  # Замените на ваш путь

if __name__ == "__main__":
    # Установите: pip install tqdm

    print("Начинаем обработку...")
    results = process_large_dataset(
        base_folder=BASE_FOLDER,
        crops_folder=CROPS_FOLDER,
        output_json=OUTPUT_JSON,
        batch_size=2  # Можно увеличить если хватает памяти
    )

    # Вывод статистики
    if results:
        print(f"\nСтатистика:")
        print(f"  Всего записей: {len(results)}")
        print(f"  Размер эмбеддинга: {len(results[0]['text_embedding'])}")

        # Пример вывода первых нескольких эмбеддингов
        print(f"\nПример эмбеддинга (первые 5 значений):")
        for i in range(min(3, len(results))):
            emb = results[i]['text_embedding'][:5]
            print(f"  Запись {i + 1}: {emb}")