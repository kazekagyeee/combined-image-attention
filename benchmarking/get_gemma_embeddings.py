import torch
from transformers import AutoProcessor, Gemma3ForConditionalGeneration
from PIL import Image
import numpy as np
from pathlib import Path
import json
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class OptimizedGemmaEmbedder:
    def __init__(self, model_name="google/gemma-3-4b-it"):
        """
        Оптимизированный эмбеддер для Gemma 3 4B Instruct (VLM)
        """
        print(f"Загрузка модели {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Определяем dtype в зависимости от доступности GPU
        if self.device == "cuda" and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            print("Используется bfloat16 на CUDA")
        elif self.device == "cuda":
            torch_dtype = torch.float16
            print("Используется float16 на CUDA")
        else:
            torch_dtype = torch.float32
            print(f"Используется float32 на {self.device}")

        from_pretrained_kwargs = {"token": ""}

        # Загружаем модель и процессор
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map="auto" if self.device == "cuda" else None,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            **from_pretrained_kwargs
        )

        # Если CPU, перемещаем модель вручную
        if self.device == "cpu":
            self.model = self.model.to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, **from_pretrained_kwargs)
        self.model.eval()

        print(f"Модель загружена на {self.device}")
        print(f"Тип данных модели: {self.model.dtype}")

    def get_batch_embeddings(self, full_images, crop_images, texts):
        """
        Получение эмбеддингов для батча изображений и текста
        """
        embeddings = []

        for full_img, crop_img, text in zip(full_images, crop_images, texts):
            try:
                # Формируем сообщение в формате, который понимает Gemma 3
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": full_img},
                            {"type": "image", "image": crop_img},
                            {"type": "text",
                             "text": f"Проанализируй эти изображения. Описание основного изображения: {text}"}
                        ]
                    }
                ]

                # Применяем чат-шаблон и токенизируем
                inputs = self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt"
                ).to(self.device, dtype=self.model.dtype)

                input_len = inputs["input_ids"].shape[-1]

                # Получаем скрытые состояния
                with torch.inference_mode():
                    outputs = self.model(
                        **inputs,
                        output_hidden_states=True,
                        return_dict=True,
                        max_new_tokens=1  # Минимум токенов, так как нам нужны только эмбеддинги
                    )

                    # Берем скрытые состояния последнего слоя
                    last_hidden_state = outputs.hidden_states[-1]

                    # Усредняем по всем токенам (кроме padding)
                    # Внимание: для Gemma эмбеддинг может быть в другом месте
                    # Если это не работает, попробуйте outputs.last_hidden_state
                    embedding = last_hidden_state.mean(dim=1)

                    # Нормализуем
                    embedding = embedding.cpu().numpy()
                    embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

                embeddings.append(embedding[0])

            except Exception as e:
                print(f"Ошибка при обработке: {str(e)[:100]}...")
                embeddings.append(None)

        return embeddings


def process_large_dataset(base_folder, crops_folder, output_json, batch_size=2):
    """
    Обработка больших наборов данных с батч-обработкой для Gemma 3
    """
    print("Сканирование файлов...")

    # Находим пары в базовой папке
    base_path = Path(base_folder)
    crops_path = Path(crops_folder)

    # Получаем все пары (изображение + текст)
    image_exts = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp']
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
    embedder = OptimizedGemmaEmbedder()

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

        # Загружаем основное изображение один раз
        try:
            full_image = Image.open(full_img_path).convert('RGB')
            # Можете добавить ресайз до 896x896 как в документации
            # full_image = full_image.resize((896, 896))
        except Exception as e:
            print(f"Не удалось загрузить {full_img_path}: {e}")
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

            for crop_path in batch_crops:
                try:
                    crop_image = Image.open(crop_path).convert('RGB')
                    # Можете добавить ресайз до 896x896
                    # crop_image = crop_image.resize((896, 896))

                    full_images.append(full_image)
                    crop_images.append(crop_image)
                    texts.append(text)
                    crop_paths.append(crop_path)

                except Exception as e:
                    print(f"Не удалось загрузить {crop_path}: {e}")
                    continue

            if not full_images:
                pbar.update(len(batch_crops))
                continue

            # Получаем эмбеддинги для батча
            batch_embeddings = embedder.get_batch_embeddings(full_images, crop_images, texts)

            # Сохраняем результаты
            for crop_path, embedding in zip(crop_paths, batch_embeddings):
                if embedding is not None:
                    # Получаем размеры изображения для записи в JSON
                    try:
                        with Image.open(crop_path) as img:
                            width, height = img.size
                            crop_wh = [width, height]
                            area = width * height
                    except:
                        crop_wh = []
                        area = 0.0

                    record = {
                        "crop_path": str(crop_path.absolute()),
                        "orig_path": str(full_img_path.absolute()),
                        "bbox": [],
                        "score": 0.0,
                        "caption": text,
                        "prompt_used": f"Анализ изображений. Описание: {text[:50]}...",
                        "area": area,
                        "rel_size_coeff": 0.0,
                        "crop_wh": crop_wh,
                        "gemma_embedding": embedding.tolist()
                    }
                    results.append(record)

            # Обновляем прогресс-бар
            pbar.update(len(batch_crops))

            # Промежуточное сохранение каждые 50 записей
            if len(results) % 50 == 0:
                with open(output_json, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                print(f"\nПромежуточное сохранение: {len(results)} записей")

    pbar.close()

    # Финальное сохранение
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\nГотово! Сохранено {len(results)} записей в {output_json}")

    return results


# КОНФИГУРАЦИЯ
BASE_FOLDER = "../images"  # Замените на ваш путь
CROPS_FOLDER = "../out"  # Замените на ваш путь
OUTPUT_JSON = "gemma3_embeddings.json"  # Замените на ваш путь

if __name__ == "__main__":
    # Установите необходимые зависимости:
    # pip install torch transformers pillow tqdm numpy accelerate

    print("=" * 60)
    print("Gemma 3 4B VLM Embedder")
    print("=" * 60)

    results = process_large_dataset(
        base_folder=BASE_FOLDER,
        crops_folder=CROPS_FOLDER,
        output_json=OUTPUT_JSON,
        batch_size=2  # Начните с малого, можно увеличить при наличии памяти
    )

    # Вывод статистики
    if results:
        print(f"\nСтатистика:")
        print(f"  Всего записей: {len(results)}")
        print(f"  Размер эмбеддинга: {len(results[0]['gemma_embedding'])}")

        # Пример вывода первых нескольких эмбеддингов
        print(f"\nПример эмбеддинга (первые 5 значений):")
        for i in range(min(3, len(results))):
            emb = results[i]['gemma_embedding'][:5]
            print(f"  Запись {i + 1}: {emb}")
    else:
        print("\nНе удалось получить результаты. Проверьте пути к файлам.")