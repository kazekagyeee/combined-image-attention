import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')


class TextEmbedderBERT:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Инициализация BERT модели для извлечения эмбеддингов из текста
        """
        print(f"Загрузка модели {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загружаем токенизатор и модель BERT
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Переводим модель в режим оценки
        self.model.eval()
        print(f"Модель BERT загружена на устройство: {self.device}")
        print(f"Размерность эмбеддингов BERT: {self.model.config.hidden_size}")

    def get_text_embedding(self, text, pooling_strategy='cls'):
        """
        Получение эмбеддинга из текста с использованием BERT

        Аргументы:
            text: текст для обработки
            pooling_strategy: стратегия пулинга:
                - 'cls': использовать [CLS] токен
                - 'mean': усреднить по всем токенам (кроме специальных)
                - 'max': максимальный пулинг
        """
        try:
            # Токенизация текста
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512  # Максимальная длина для BERT
            ).to(self.device)

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Получаем скрытые состояния последнего слоя
                last_hidden_state = outputs.last_hidden_state

                # Применяем выбранную стратегию пулинга
                if pooling_strategy == 'cls':
                    # Используем эмбеддинг [CLS] токена (первый токен)
                    embedding = last_hidden_state[:, 0, :]

                elif pooling_strategy == 'mean':
                    # Усредняем по всем токенам (исключая специальные токены)
                    # Создаем маску для исключения padding токенов
                    attention_mask = inputs['attention_mask']

                    # Расширяем маску для соответствия размерности эмбеддингов
                    mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

                    # Умножаем эмбеддинги на маску (чтобы padding токены не влияли)
                    sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

                    # Усредняем
                    embedding = sum_embeddings / sum_mask

                elif pooling_strategy == 'max':
                    # Максимальный пулинг
                    embedding, _ = torch.max(last_hidden_state, dim=1)

                else:
                    raise ValueError(f"Неизвестная стратегия пулинга: {pooling_strategy}")

                # Преобразуем в numpy и нормализуем
                embedding = embedding.cpu().numpy()
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            return embedding[0]  # Возвращаем первый (и единственный) элемент батча

        except Exception as e:
            print(f"Ошибка при получении эмбеддинга BERT: {e}")
            import traceback
            traceback.print_exc()
            return None


def find_single_file_pair(folder_path, base_name=None):
    """
    Находит пару файлов (изображение + текст) в папке
    Если указан base_name - ищет конкретную пару, иначе первую найденную
    """
    folder = Path(folder_path)

    # Поддерживаемые форматы изображений
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']

    if base_name:
        # Ищем конкретную пару по имени
        for ext in image_extensions:
            image_path = folder / f"{base_name}{ext}"
            if image_path.exists():
                text_path = folder / f"{base_name}.txt"
                if text_path.exists():
                    return image_path, text_path
                else:
                    # Проверяем другие варианты расширений текста
                    for txt_ext in ['.txt', '.text', '.md']:
                        text_path = folder / f"{base_name}{txt_ext}"
                        if text_path.exists():
                            return image_path, text_path
    else:
        # Ищем первую подходящую пару
        for ext in image_extensions:
            for image_path in folder.glob(f"*{ext}"):
                # Проверяем существование текстового файла
                for txt_ext in ['.txt', '.text', '.md']:
                    text_path = folder / f"{image_path.stem}{txt_ext}"
                    if text_path.exists():
                        return image_path, text_path

    return None, None


def get_text_from_file(text_path):
    """
    Читает текст из файла
    """
    try:
        with open(text_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Ошибка чтения файла {text_path}: {e}")
        return ""


def get_multiple_text_embeddings(folder_path, output_json, base_names=None, pooling_strategy='cls'):
    """
    Рассчитывает эмбеддинги BERT для нескольких текстов из парных файлов

    Аргументы:
        folder_path: путь к папке с файлами
        output_json: путь для сохранения JSON файла
        base_names: список имен файлов (без расширений) для обработки
        pooling_strategy: стратегия пулинга для BERT
    """
    print("=" * 60)
    print("РАСЧЕТ ЭМБЕДДИНГОВ BERT ДЛЯ ТЕКСТОВ")
    print("=" * 60)

    # Инициализируем BERT эмбеддер
    embedder = TextEmbedderBERT()

    records = []

    # Если указаны конкретные имена файлов
    if base_names:
        file_names = base_names
    else:
        # Получаем список всех текстовых файлов в папке
        folder = Path(folder_path)
        text_files = list(folder.glob("*.txt")) + list(folder.glob("*.text")) + list(folder.glob("*.md"))
        # Берем только имена без расширений
        file_names = [f.stem for f in text_files]

    print(f"Найдено {len(file_names)} текстовых файлов для обработки")

    for base_name in file_names:
        print(f"\nОбработка: {base_name}")

        # Находим пару файлов
        image_path, text_path = find_single_file_pair(folder_path, base_name)

        if not text_path:
            print(f"Текстовый файл не найден для {base_name}, пропускаем...")
            continue

        # Читаем текст
        text = get_text_from_file(text_path)

        if not text:
            print(f"Текстовый файл пустой для {base_name}, пропускаем...")
            continue

        print(f"Текст (первые 100 символов): {text[:100]}...")

        # Получаем эмбеддинг BERT
        print(f"Извлечение эмбеддинга BERT (стратегия: {pooling_strategy})...")
        embedding = embedder.get_text_embedding(text, pooling_strategy)

        if embedding is None:
            print(f"Не удалось извлечь эмбеддинг для {base_name}!")
            continue

        # Создаем запись
        record = {
            "base_name": base_name,
            "text_path": str(text_path.absolute()) if text_path else "",
            "image_path": str(image_path.absolute()) if image_path else "",
            "text": text[:500] + "..." if len(text) > 500 else text,  # Сохраняем часть текста
            "text_length": len(text),
            "pooling_strategy": pooling_strategy,
            "embedding_dim": len(embedding),
            "text_embedding": embedding.tolist(),
            "embedding_stats": {
                "min": float(np.min(embedding)),
                "max": float(np.max(embedding)),
                "mean": float(np.mean(embedding)),
                "std": float(np.std(embedding))
            }
        }

        records.append(record)

        print(f"✓ Эмбеддинг успешно извлечен (размерность: {len(embedding)})")

    if not records:
        print("Не удалось обработать ни один файл!")
        return None

    # Сохраняем в JSON файл
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в: {output_json}")
    print(f"Всего обработано записей: {len(records)}")

    # Выводим сводную статистику
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА ЭМБЕДДИНГОВ")
    print("=" * 60)

    for i, record in enumerate(records):
        print(f"\n{i + 1}. {record['base_name']}:")
        print(f"   Длина текста: {record['text_length']} символов")
        print(f"   Размерность эмбеддинга: {record['embedding_dim']}")
        print(f"   Min: {record['embedding_stats']['min']:.6f}")
        print(f"   Max: {record['embedding_stats']['max']:.6f}")
        print(f"   Mean: {record['embedding_stats']['mean']:.6f}")

    return records


def calculate_single_bert_embedding(folder_path, output_json, base_name=None, pooling_strategy='cls'):
    """
    Рассчитывает эмбеддинг BERT для одной пары изображение+текст
    (Только текст используется для BERT)
    """
    print("=" * 60)
    print("РАСЧЕТ ЭМБЕДДИНГА BERT ДЛЯ ОДНОГО ТЕКСТА")
    print("=" * 60)

    # Находим пару файлов
    image_path, text_path = find_single_file_pair(folder_path, base_name)

    if not text_path:
        print(f"Не удалось найти текстовый файл в папке: {folder_path}")
        if base_name:
            print(f"Искали по имени: {base_name}")

        # Показываем какие файлы есть в папке
        folder = Path(folder_path)
        files = list(folder.iterdir())
        if files:
            print("Файлы в папке:")
            for f in files:
                print(f"  - {f.name}")
        return None

    print(f"Найдена пара файлов:")
    if image_path:
        print(f"  Изображение: {image_path.name}")
    print(f"  Текст: {text_path.name}")

    # Читаем текст
    text = get_text_from_file(text_path)

    if not text:
        print("Текстовый файл пустой!")
        return None

    print(f"Текст (первые 200 символов): {text[:200]}...")

    # Инициализируем BERT эмбеддер
    embedder = TextEmbedderBERT()

    # Получаем эмбеддинг BERT
    print(f"\nИзвлечение эмбеддинга BERT (стратегия: {pooling_strategy})...")
    embedding = embedder.get_text_embedding(text, pooling_strategy)

    if embedding is None:
        print("Не удалось извлечь эмбеддинг BERT!")
        return None

    print(f"Эмбеддинг BERT успешно извлечен!")
    print(f"Размерность эмбеддинга: {embedding.shape}")

    # Создаем запись в формате JSON
    record = {
        "base_name": base_name if base_name else text_path.stem,
        "text_path": str(text_path.absolute()),
        "image_path": str(image_path.absolute()) if image_path else "",
        "text": text[:1000] + "..." if len(text) > 1000 else text,
        "text_length": len(text),
        "pooling_strategy": pooling_strategy,
        "text_embedding": embedding.tolist(),
        "embedding_stats": {
            "min": float(np.min(embedding)),
            "max": float(np.max(embedding)),
            "mean": float(np.mean(embedding)),
            "std": float(np.std(embedding))
        }
    }

    # Сохраняем в JSON файл
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump([record], f, ensure_ascii=False, indent=2)

    print(f"\nРезультат сохранен в: {output_json}")

    # Выводим информацию об эмбеддинге
    print("\nИнформация об эмбеддинге BERT:")
    print(f"  Имя файла: {record['base_name']}")
    print(f"  Длина текста: {len(text)} символов")
    print(f"  Размерность эмбеддинга: {len(record['text_embedding'])}")
    print(f"  Стратегия пулинга: {pooling_strategy}")
    print(f"  Первые 5 значений: {record['text_embedding'][:5]}")
    print(f"  Минимальное значение: {record['embedding_stats']['min']:.6f}")
    print(f"  Максимальное значение: {record['embedding_stats']['max']:.6f}")
    print(f"  Среднее значение: {record['embedding_stats']['mean']:.6f}")
    print(f"  Стандартное отклонение: {record['embedding_stats']['std']:.6f}")

    return record


if __name__ == "__main__":
    # Конфигурация
    folder_path = "../images"
    output_json = "core_embedding.json"
    base_name = "image_20_2"  # Без расширения, можно указать None для обработки всех файлов

    # Выберите стратегию пулинга:
    # 'cls' - использует [CLS] токен (рекомендуется)
    # 'mean' - усредняет по всем токенам
    # 'max' - максимальный пулинг
    pooling_strategy = 'cls'

    # Обработка одного файла
    record = calculate_single_bert_embedding(
        folder_path=folder_path,
        output_json=output_json,
        base_name=base_name,
        pooling_strategy=pooling_strategy
    )

    # Или для обработки всех текстовых файлов в папке:
    # records = get_multiple_text_embeddings(
    #     folder_path=folder_path,
    #     output_json=output_json,
    #     base_names=None,  # None означает все файлы
    #     pooling_strategy=pooling_strategy
    # )