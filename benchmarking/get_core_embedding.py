import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np
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


def generate_embeddings_for_questions(questions, output_json, pooling_strategy='cls'):
    """
    Генерация эмбеддингов для списка вопросов

    Аргументы:
        questions: список вопросов для обработки
        output_json: путь для сохранения JSON файла
        pooling_strategy: стратегия пулинга для BERT
    """
    print("=" * 60)
    print("ГЕНЕРАЦИЯ ЭМБЕДДИНГОВ ДЛЯ ВОПРОСОВ")
    print("=" * 60)

    # Инициализируем BERT эмбеддер
    embedder = TextEmbedderBERT()

    records = []

    print(f"Найдено {len(questions)} вопросов для обработки")

    for i, question in enumerate(questions, 1):
        print(f"\nОбработка вопроса {i}/{len(questions)}:")
        print(f"Вопрос: {question[:100]}..." if len(question) > 100 else f"Вопрос: {question}")

        # Получаем эмбеддинг BERT
        print(f"Извлечение эмбеддинга BERT (стратегия: {pooling_strategy})...")
        embedding = embedder.get_text_embedding(question, pooling_strategy)

        if embedding is None:
            print(f"Не удалось извлечь эмбеддинг для вопроса {i}!")
            continue

        # Создаем запись согласно требуемой структуре
        record = {
            "question": question,
            "text_embedding": embedding.tolist()
        }

        records.append(record)

        print(f"✓ Эмбеддинг успешно извлечен (размерность: {len(embedding)})")

    if not records:
        print("Не удалось обработать ни один вопрос!")
        return None

    # Сохраняем в JSON файл
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nРезультаты сохранены в: {output_json}")
    print(f"Всего обработано записей: {len(records)}")

    # Выводим сводную статистику
    print("\n" + "=" * 60)
    print("СВОДНАЯ СТАТИСТИКА")
    print("=" * 60)

    for i, record in enumerate(records):
        embedding_array = np.array(record['text_embedding'])
        print(f"\n{i + 1}. Вопрос (первые 50 символов): {record['question'][:50]}...")
        print(f"   Размерность эмбеддинга: {len(embedding_array)}")
        print(f"   Min: {float(np.min(embedding_array)):.6f}")
        print(f"   Max: {float(np.max(embedding_array)):.6f}")
        print(f"   Mean: {float(np.mean(embedding_array)):.6f}")

    return records


def calculate_single_question_embedding(question, output_json, pooling_strategy='cls'):
    """
    Рассчитывает эмбеддинг BERT для одного вопроса

    Аргументы:
        question: вопрос для обработки
        output_json: путь для сохранения JSON файла
        pooling_strategy: стратегия пулинга для BERT
    """
    print("=" * 60)
    print("РАСЧЕТ ЭМБЕДДИНГА BERT ДЛЯ ОДНОГО ВОПРОСА")
    print("=" * 60)

    if not question or not question.strip():
        print("Вопрос не может быть пустым!")
        return None

    print(f"Вопрос: {question}")

    # Инициализируем BERT эмбеддер
    embedder = TextEmbedderBERT()

    # Получаем эмбеддинг BERT
    print(f"\nИзвлечение эмбеддинга BERT (стратегия: {pooling_strategy})...")
    embedding = embedder.get_text_embedding(question, pooling_strategy)

    if embedding is None:
        print("Не удалось извлечь эмбеддинг BERT!")
        return None

    print(f"Эмбеддинг BERT успешно извлечен!")
    print(f"Размерность эмбеддинга: {embedding.shape}")

    # Создаем запись в формате JSON
    record = {
        "question": question,
        "text_embedding": embedding.tolist()
    }

    # Сохраняем в JSON файл
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump([record], f, ensure_ascii=False, indent=2)

    print(f"\nРезультат сохранен в: {output_json}")

    # Выводим информацию об эмбеддинге
    print("\nИнформация об эмбеддинге BERT:")
    print(f"  Вопрос: {question[:100]}..." if len(question) > 100 else f"  Вопрос: {question}")
    print(f"  Размерность эмбеддинга: {len(record['text_embedding'])}")
    print(f"  Стратегия пулинга: {pooling_strategy}")
    print(f"  Первые 5 значений: {record['text_embedding'][:5]}")

    return record


if __name__ == "__main__":
    # Конфигурация
    output_json = "core_embeddings.json"

    # Список вопросов для обработки
    questions = [
        "Что нужно сделать, чтобы реквизиты организации заполнились автоматически при её создании?",
        "Какие данные необходимо указать при создании организации, чтобы реквизиты заполнились автоматически?",
        "При каких условиях реквизиты организации заполняются автоматически?",
        "Что произойдёт, если введённый ИНН отсутствует в государственном реестре?",
        "На каком этапе работы с карточкой организации выполняется автоматическое заполнение реквизитов?"
    ]

    # Выберите стратегию пулинга:
    # 'cls' - использует [CLS] токен (рекомендуется)
    # 'mean' - усредняет по всем токенам
    # 'max' - максимальный пулинг
    pooling_strategy = 'cls'

    # Обработка всех вопросов
    records = generate_embeddings_for_questions(
        questions=questions,
        output_json=output_json,
        pooling_strategy=pooling_strategy
    )

    # Или для обработки одного вопроса:
    # single_question = "Что нужно сделать, чтобы реквизиты организации заполнились автоматически при её создании?"
    # record = calculate_single_question_embedding(
    #     question=single_question,
    #     output_json=output_json,
    #     pooling_strategy=pooling_strategy
    # )