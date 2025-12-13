import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_distances
import warnings
from typing import List, Tuple, Optional
from collections import defaultdict

warnings.filterwarnings('ignore')


class TextEmbedder:
    def __init__(self, model_name="bert-base-uncased"):
        """
        Инициализация модели BERT и токенизатора
        """
        print(f"Загрузка модели {model_name}...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Загружаем токенизатор
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Загружаем модель BERT
        self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Переводим модель в режим оценки
        self.model.eval()
        print("Модель загружена!")

    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Получение эмбеддинга для текста с помощью BERT
        """
        try:
            # Токенизация текста
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)

            # Получение эмбеддингов
            with torch.no_grad():
                outputs = self.model(**inputs)

                # Получаем скрытые состояния последнего слоя
                last_hidden_state = outputs.last_hidden_state

                # Применяем attention mask для усреднения только по реальным токенам
                attention_mask = inputs['attention_mask']

                # Умножаем каждое скрытое состояние на соответствующий вес маски
                mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
                sum_embeddings = torch.sum(last_hidden_state * mask_expanded, 1)
                sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)

                # Усредненные эмбеддинги
                mean_embeddings = sum_embeddings / sum_mask

                # Конвертируем в numpy
                embedding = mean_embeddings.cpu().numpy()

                # Нормализуем
                embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            return embedding[0]  # Возвращаем первый (и единственный) элемент батча

        except Exception as e:
            print(f"Ошибка при обработке текста: {e}")
            return None

    def get_embedding_from_file(self, text_path: Path) -> Optional[np.ndarray]:
        """
        Получение эмбеддинга для текста из файла
        """
        # Загрузка текста
        try:
            with open(text_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except Exception as e:
            print(f"Ошибка загрузки текста {text_path}: {e}")
            return None

        if not text:
            print(f"Файл {text_path.name} пустой")
            return None

        # Получение эмбеддинга
        return self.get_embedding(text)


def find_text_files(directory: str) -> List[Path]:
    """
    Поиск текстовых файлов в указанной директории
    """
    directory = Path(directory)

    # Находим все файлы .txt
    txt_files = list(directory.glob("*.txt"))

    # Сортируем для консистентности
    txt_files.sort()

    print(f"Найдено {len(txt_files)} текстовых файлов")
    return txt_files


def filter_duplicate_embeddings(
        embeddings: List[np.ndarray],
        file_names: List[str],
        similarity_threshold: float = 0.99
) -> Tuple[List[np.ndarray], List[str], dict]:
    """
    Фильтрация идентичных и похожих эмбеддингов

    Args:
        embeddings: Список эмбеддингов
        file_names: Список имен файлов
        similarity_threshold: Порог схожести (1.0 - идеальное совпадение)

    Returns:
        filtered_embeddings: Отфильтрованные уникальные эмбеддинги
        filtered_names: Соответствующие имена файлов
        duplicates_info: Информация о дубликатах
    """
    if not embeddings:
        return [], [], {}

    # Инициализация
    filtered_embeddings = [embeddings[0]]
    filtered_names = [file_names[0]]
    duplicates_info = defaultdict(list)

    # Создаем хэш-таблицу для быстрого поиска дубликатов
    unique_embeddings = {0}  # Индексы уникальных эмбеддингов

    for i in range(1, len(embeddings)):
        is_duplicate = False

        # Проверяем на дубликаты
        for unique_idx in unique_embeddings:
            # Вычисляем косинусное сходство
            emb1 = embeddings[i].reshape(1, -1)
            emb2 = filtered_embeddings[unique_idx].reshape(1, -1)

            # Косинусное сходство = 1 - косинусное расстояние
            similarity = 1 - cosine_distances(emb1, emb2)[0][0]

            if similarity >= similarity_threshold:
                # Найден дубликат
                is_duplicate = True
                original_name = filtered_names[unique_idx]
                duplicate_name = file_names[i]
                duplicates_info[original_name].append(duplicate_name)
                print(f"  ⚠ Пропущен дубликат: {duplicate_name} похож на {original_name} (сходство: {similarity:.4f})")
                break

        if not is_duplicate:
            # Добавляем как уникальный
            filtered_embeddings.append(embeddings[i])
            filtered_names.append(file_names[i])
            unique_embeddings.add(len(filtered_embeddings) - 1)

    return filtered_embeddings, filtered_names, duplicates_info


def filter_consecutive_duplicates(
        embeddings: List[np.ndarray],
        file_names: List[str],
        similarity_threshold: float = 0.99
) -> Tuple[List[np.ndarray], List[str], dict]:
    """
    Фильтрация только последовательных дубликатов (более быстрый метод)
    """
    if not embeddings:
        return [], [], {}

    filtered_embeddings = []
    filtered_names = []
    duplicates_info = defaultdict(list)

    prev_embedding = None
    prev_name = None

    for emb, name in zip(embeddings, file_names):
        if prev_embedding is not None:
            # Проверяем сходство с предыдущим эмбеддингом
            similarity = 1 - cosine_distances(
                emb.reshape(1, -1),
                prev_embedding.reshape(1, -1)
            )[0][0]

            if similarity >= similarity_threshold:
                # Найден последовательный дубликат
                duplicates_info[prev_name].append(name)
                print(
                    f"  ⚠ Пропущен последовательный дубликат: {name} похож на {prev_name} (сходство: {similarity:.4f})")
                continue

        # Добавляем как уникальный
        filtered_embeddings.append(emb)
        filtered_names.append(name)
        prev_embedding = emb
        prev_name = name

    return filtered_embeddings, filtered_names, duplicates_info


def calculate_average_distance(embeddings: List[np.ndarray]) -> Optional[Tuple[float, float, List[float]]]:
    """
    Вычисление среднего расстояния между соседними эмбеддингами
    """
    if len(embeddings) < 2:
        print("Недостаточно эмбеддингов для вычисления расстояний")
        return None

    distances = []

    # Вычисляем косинусные расстояния между соседними эмбеддингами
    for i in range(len(embeddings) - 1):
        # Преобразуем в 2D массивы для cosine_distances
        emb1 = embeddings[i].reshape(1, -1)
        emb2 = embeddings[i + 1].reshape(1, -1)

        # Вычисляем косинусное расстояние
        distance = cosine_distances(emb1, emb2)[0][0]
        distances.append(distance)

    # Вычисляем среднее значение
    avg_distance = np.mean(distances)
    std_distance = np.std(distances)

    return avg_distance, std_distance, distances


def main(
        directory_path: str,
        model_name: str = "bert-base-uncased",
        deduplication_method: str = "all",  # "all", "consecutive", или "none"
        similarity_threshold: float = 0.99,
        min_text_length: int = 10
):
    """
    Основная функция

    Args:
        directory_path: Путь к директории с текстовыми файлами
        model_name: Название модели BERT
        deduplication_method: Метод удаления дубликатов ("all", "consecutive", "none")
        similarity_threshold: Порог схожести для определения дубликатов (0.99 = 99% сходство)
        min_text_length: Минимальная длина текста для обработки
    """
    # Инициализируем эмбеддер
    embedder = TextEmbedder(model_name)

    # Находим текстовые файлы
    text_files = find_text_files(directory_path)

    if not text_files:
        print("Не найдено текстовых файлов!")
        return

    # Извлекаем эмбеддинги
    all_embeddings = []
    all_file_names = []
    successful = 0
    skipped_short = 0

    print("\nИзвлечение эмбеддингов...")
    for txt_path in text_files:
        print(f"Обработка: {txt_path.stem}")

        # Проверка длины текста перед обработкой
        try:
            with open(txt_path, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if len(text) < min_text_length:
                    print(f"  ⚠ Пропущен (текст слишком короткий: {len(text)} символов)")
                    skipped_short += 1
                    continue
        except:
            pass

        embedding = embedder.get_embedding_from_file(txt_path)

        if embedding is not None:
            all_embeddings.append(embedding)
            all_file_names.append(txt_path.stem)
            successful += 1
            print(f"  ✓ Успешно (размерность: {embedding.shape})")
        else:
            print(f"  ✗ Ошибка")

    print(f"\nПредварительная статистика:")
    print(f"  Всего файлов: {len(text_files)}")
    print(f"  Успешно обработано: {successful}")
    print(f"  Пропущено (короткий текст): {skipped_short}")

    if successful < 2:
        print("Недостаточно успешных обработок для вычисления расстояний")
        return

    # Фильтрация дубликатов
    print(f"\nФильтрация дубликатов (метод: {deduplication_method})...")

    if deduplication_method == "all":
        filtered_embeddings, filtered_names, duplicates_info = filter_duplicate_embeddings(
            all_embeddings, all_file_names, similarity_threshold
        )
    elif deduplication_method == "consecutive":
        filtered_embeddings, filtered_names, duplicates_info = filter_consecutive_duplicates(
            all_embeddings, all_file_names, similarity_threshold
        )
    elif deduplication_method == "none":
        filtered_embeddings, filtered_names = all_embeddings, all_file_names
        duplicates_info = {}
    else:
        print(f"Неизвестный метод дедупликации: {deduplication_method}")
        return

    # Вывод статистики по дубликатам
    if duplicates_info:
        print(f"\nНайдено {sum(len(v) for v in duplicates_info.values())} дубликатов:")
        for original, duplicates in duplicates_info.items():
            print(f"  {original}: {len(duplicates)} дубликатов")
    else:
        print("Дубликаты не найдены")

    print(f"\nПосле фильтрации: {len(filtered_embeddings)} уникальных эмбеддингов")

    if len(filtered_embeddings) < 2:
        print("Недостаточно уникальных эмбеддингов для вычисления расстояний")
        return

    # Вычисляем среднее расстояние
    result = calculate_average_distance(filtered_embeddings)

    if result:
        avg_distance, std_distance, distances = result

        print(f"\n{'=' * 50}")
        print("РЕЗУЛЬТАТЫ:")
        print(f"{'=' * 50}")
        print(f"Всего файлов: {len(text_files)}")
        print(f"Обработано успешно: {successful}")
        print(f"Уникальных эмбеддингов: {len(filtered_embeddings)}")
        print(f"Найдено дубликатов: {sum(len(v) for v in duplicates_info.values())}")
        print(f"Размерность эмбеддингов: {filtered_embeddings[0].shape}")
        print(f"Количество расстояний: {len(distances)}")
        print(f"Среднее косинусное расстояние между соседями: {avg_distance:.6f}")
        print(f"Стандартное отклонение: {std_distance:.6f}")
        print(f"Минимальное расстояние: {min(distances):.6f}")
        print(f"Максимальное расстояние: {max(distances):.6f}")

        # Дополнительная статистика по расстояниям
        print(f"\nСтатистика по расстояниям:")
        percentiles = [25, 50, 75, 90, 95]
        for p in percentiles:
            value = np.percentile(distances, p)
            print(f"  {p}% перцентиль: {value:.6f}")

        print(f"\nВсе расстояния:")
        for i, dist in enumerate(distances):
            print(f"  {filtered_names[i]} → {filtered_names[i + 1]}: {dist:.6f}")

        return avg_distance, filtered_embeddings, filtered_names, duplicates_info


if __name__ == "__main__":
    # Укажите путь к директории с текстовыми файлами
    directory_path = "../images-all"  # Замените на путь к вашей директории

    # Настройки
    model_name = "bert-base-uncased"  # Измените при необходимости

    # Настройки дедупликации
    deduplication_method = "all"  # "all", "consecutive", или "none"
    similarity_threshold = 0.99  # Порог схожести для определения дубликатов (0.99 = 99% сходство)
    min_text_length = 10  # Минимальная длина текста для обработки

    print("=" * 50)
    print("НАСТРОЙКИ:")
    print("=" * 50)
    print(f"Модель: {model_name}")
    print(f"Метод дедупликации: {deduplication_method}")
    print(f"Порог схожести: {similarity_threshold}")
    print(f"Минимальная длина текста: {min_text_length}")
    print("=" * 50)

    # Запуск основной функции
    result = main(
        directory_path,
        model_name,
        deduplication_method,
        similarity_threshold,
        min_text_length
    )

    # Если нужно сохранить результаты для дальнейшего использования
    if result:
        avg_distance, embeddings, file_names, duplicates_info = result

        # Пример сохранения результатов
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)

        # Сохранение эмбеддингов
        np.save(output_dir / "embeddings.npy", embeddings)

        # Сохранение имен файлов
        with open(output_dir / "file_names.txt", "w", encoding="utf-8") as f:
            for name in file_names:
                f.write(f"{name}\n")

        # Сохранение информации о дубликатах
        if duplicates_info:
            with open(output_dir / "duplicates.txt", "w", encoding="utf-8") as f:
                f.write("Дубликаты:\n")
                for original, duplicates in duplicates_info.items():
                    f.write(f"\n{original}:\n")
                    for dup in duplicates:
                        f.write(f"  - {dup}\n")

        print(f"\nРезультаты сохранены в папке '{output_dir}'")