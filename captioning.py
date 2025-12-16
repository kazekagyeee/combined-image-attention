from abc import ABC, abstractmethod

from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModel, Gemma3ForConditionalGeneration
import torch
import numpy as np
import torch.nn.functional as F
from typing import List, Union, Optional

import config


class CaptionerBase(ABC):
    """Базовый класс для генератора текстовых описаний"""

    def __init__(self, device='cuda'):
        self.device = device

    @abstractmethod
    def describe(self, image: Image.Image, prompt: str = None, max_length=128) -> str:
        """
        Описывает изображение отталкиваясь от переданного промта

        :param image: изображение
        :param prompt: промт
        :param max_length: максимальная длина текста
        :return: текстовое описание изображения
        """
        pass

    @abstractmethod
    def describe_two_images(self, full_image: Image.Image, cropped_image: Image.Image,
                            prompt: str = None, max_length=256) -> str:
        """
        Генерирует описание, анализируя два изображения.

        Args:
            full_image (Image.Image): Полное изображение.
            cropped_image (Image.Image): Часть полного изображения.
            prompt (str): Текстовый запрос к модели (например, "What is the difference?").
            max_length (int): Максимальная длина ответа.

        Returns:
            str: Текстовый ответ модели.
        """
        pass


class CaptionerBLIP(CaptionerBase):
    """Генератор текстовых описаний для изображений (BLIP)."""

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device='cuda'):
        super().__init__(device)
        print("Loading captioning model:", model_name)
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)

    def describe_two_images(self, full_image: Image.Image, cropped_image: Image.Image, prompt: str = None,
                            max_length=256) -> str:
        pass

    def describe(self, image: Image.Image, prompt: str = None, label: str = None, max_length=128) -> str:
        """Генерирует описание для изображения. При наличии `prompt` — учитывает его."""
        if prompt is None:
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        else:
            # BLIP supports conditional generation with a prompt
            prompt = prompt.replace("{LABEL}", label)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption

class CaptionerQwen(CaptionerBase):
    """Генератор текстовых описаний для изображений с помощью Qwen2‑VL‑7B."""

    def __init__(self, model_name = "Qwen/Qwen2.5-VL-7B-Instruct", device='cuda'):
        super().__init__(device)
        print("Loading Qwen2‑VL model:", model_name)
        # Загружаем процессор (tokenizer + визуальную часть)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Загружаем модель
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",  # можно выбрать вручную, но “auto” часто хорошо
            torch_dtype=torch.float16  # рекомендовано для памяти
        )
        self.model.to(device)

    def describe(self, image: Image.Image, prompt: str = None, max_length=128) -> str:
        """
        Генерирует описание для изображения. При наличии `prompt` — вставляет его вместе с картинкой.
        prompt — это текст, который задает, что именно делать с изображением.
        """
        # Подготовка сообщения в формате, который ждет Qwen2‑VL
        # Qwen2VL ожидает "chat template" — список сообщений с ролями и контентом
        # Контент — это список dict, с type="image" и type="text"
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt or "Describe this image."},
                ],
            }
        ]
        # Преобразуем сообщения в текстовый токен-представление
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        # Обрабатываем изображение и текст вместе
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Генерируем
        # Генерируем
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            do_sample=True,  # Включить сэмплирование вместо жадного поиска
            temperature=0.6,  # Увеличить температуру (0.7-1.0)
            top_p=0.75,  # Использовать nucleus sampling
            top_k=75,  # Ограничить выбор топ-k токенов
            repetition_penalty=1.1,  # Штраф за повторения
            num_beams=1,  # Отключить beam search (уменьшает копирование)
            no_repeat_ngram_size=3,  # Запретить повторение n-грамм
        )

        # Декодируем весь вывод
        full_output = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            skip_prompt=True  # Пропускаем промпт при декодировании
        )[0]

        # Убираем возможные остатки промпта или меток ассистента
        if "<|im_start|>assistant" in full_output:
            caption = full_output.split("<|im_start|>assistant")[-1].strip()
        else:
            # Альтернативный подход: убираем все до последнего промпта
            user_text = prompt or "Describe this image."
            if user_text in full_output:
                parts = full_output.split(user_text)
                if len(parts) > 1:
                    caption = parts[-1].strip()
                else:
                    caption = full_output
            else:
                caption = full_output

        # Убираем возможные теги и лишние пробелы
        caption = caption.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()

        return caption

    def describe_two_images(self, full_image: Image.Image, cropped_image: Image.Image,
                            prompt: str = None, max_length=256) -> str:
        """
        Генерирует описание, анализируя два изображения.

        Args:
            full_image (Image.Image): Полное изображение.
            cropped_image (Image.Image): Часть полного изображения.
            prompt (str): Текстовый запрос к модели (например, "What is the difference?").
            max_length (int): Максимальная длина ответа.

        Returns:
            str: Текстовый ответ модели.
        """
        # Формируем мультимодальное сообщение с двумя изображениями
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image},
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        # Применяем шаблон чата и обрабатываем изображения
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Извлекаем изображения из сообщений для обработки
        image_inputs = [full_image, cropped_image]

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        # Генерируем ответ
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_length,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            do_sample=True,  # Включить сэмплирование вместо жадного поиска
            temperature=0.6,  # Увеличить температуру (0.7-1.0)
            top_p=0.75,  # Использовать nucleus sampling
            top_k=75,  # Ограничить выбор топ-k токенов
            repetition_penalty=1.1,  # Штраф за повторения
            num_beams=1,  # Отключить beam search (уменьшает копирование)
            no_repeat_ngram_size=3,  # Запретить повторение n-грамм
        )

        # Важный шаг: обрезаем input_ids (промпт) из generated_ids
        generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]

        # Декодируем только сгенерированную часть
        caption = self.processor.decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        ).strip()

        return caption


class CaptionerGLM(CaptionerBase):
    """Генератор текстовых описаний с помощью GLM-4.1V-9B-Thinking."""

    def __init__(self, model_name="THUDM/glm-4v-9b", device='cuda'):
        super().__init__(device)
        print("Loading GLM-4.1V-9B-Thinking model:", model_name)

        # Для GLM используем ChatGLMProcessor
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        ).eval()

        self.max_length = 512

    def describe(self, image: Image.Image, prompt: str = None, max_length=128) -> str:
        """
        Генерирует описание для одного изображения.
        """
        # GLM ожидает историю сообщений в формате (role, content)
        history = []
        query = prompt or "Describe this image."

        # Преобразуем PIL Image в base64 для GLM
        import base64
        from io import BytesIO

        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # Формируем сообщение с изображением
        response, history = self.model.chat(
            self.tokenizer,
            query=query,
            history=history,
            image=img_base64,
            do_sample=True,
            temperature=0.6,
            top_p=0.75,
            max_length=max_length,
        )

        return response.strip()

    def describe_two_images(self, full_image: Image.Image, cropped_image: Image.Image,
                            prompt: str = None, max_length=256) -> str:
        """
        Генерирует описание для двух изображений с использованием GLM.

        Примечание: GLM поддерживает мультимодальный ввод через base64 строки.
        Для двух изображений можно передать список или объединенное изображение.
        """
        import base64
        from io import BytesIO

        # Преобразуем оба изображения в base64
        def image_to_base64(img):
            buffered = BytesIO()
            img.save(buffered, format="JPEG")
            return base64.b64encode(buffered.getvalue()).decode()

        img1_base64 = image_to_base64(full_image)
        img2_base64 = image_to_base64(cropped_image)

        # GLM может обрабатывать несколько изображений в одном запросе
        history = []
        query = prompt or "Describe this 2 images."

        # Создаем текстовое представление с двумя изображениями
        response, history = self.model.chat(
            self.tokenizer,
            query=query,
            history=history,
            image=[img1_base64, img2_base64],  # Передаем список изображений
            do_sample=True,
            temperature=0.6,
            top_p=0.75,
            max_length=max_length,
        )

        return response.strip()


class CaptionerGemma3(CaptionerBase):
    """Генератор текстовых описаний с помощью Google Gemma 3 (4B-IT)."""

    def __init__(self, model_name="google/gemma-3-4b-it", device='cuda'):
        super().__init__(device)
        print(f"Loading Google Gemma 3 VLM: {model_name}")

        from_pretrained_kwargs = {"token": config.PipelineConfig.huggingface_token}

        self.processor = AutoProcessor.from_pretrained(model_name, **from_pretrained_kwargs)
        self.model = Gemma3ForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            **from_pretrained_kwargs
        ).eval()

        # Явно перемещаем модель на устройство
        self.model.to(self.device)

    def describe(self, image: Image.Image, prompt: str = None, max_length=128) -> str:
        """
        Генерирует описание для одного изображения.
        """
        text_prompt = prompt or "Опиши это изображение детально."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]

        # Применяем шаблон чата и токенизируем
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Генерируем ответ
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]  # Обрезаем промпт

        # Декодируем результат
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()

    def describe_two_images(self, full_image: Image.Image, cropped_image: Image.Image,
                            prompt: str = None, max_length=256) -> str:
        """
        Генерирует описание для двух изображений.
        """
        text_prompt = prompt or "Сравни эти два изображения. Опиши их сходства и различия."

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": full_image},
                    {"type": "image", "image": cropped_image},
                    {"type": "text", "text": text_prompt}
                ]
            }
        ]

        # Применяем шаблон чата и токенизируем
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(self.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        # Генерируем ответ
        with torch.inference_mode():
            generation = self.model.generate(
                **inputs,
                max_new_tokens=max_length,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                pad_token_id=self.processor.tokenizer.eos_token_id,
            )
            generation = generation[0][input_len:]  # Обрезаем промпт

        # Декодируем результат
        decoded = self.processor.decode(generation, skip_special_tokens=True)
        return decoded.strip()


class TextEmbedderBERT:
    """Контекстно-зависимые эмбеддинги текста с помощью BERT (CLS pooling или mean pooling)."""

    def __init__(self, model_name="bert-base-uncased", device='cpu', pooling='cls'):
        print("Loading text embedder (BERT):", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling = pooling

    def embed(self, texts: list) -> list:
        """Возвращает numpy массив эмбеддингов для списка текстов."""
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for t in texts:
                enc = self.tokenizer(t, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
                out = self.model(**enc)
                last_hidden = out.last_hidden_state  # (1, seq_len, hidden)
                attention_mask = enc['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
                if self.pooling == 'cls' and hasattr(last_hidden, 'size'):
                    vec = last_hidden[:,0,:]  # CLS token
                    vec = vec.squeeze(0).cpu().numpy()
                else:
                    # mean pooling with attention mask
                    masked = last_hidden * attention_mask
                    summed = masked.sum(dim=1)
                    denom = attention_mask.sum(dim=1).clamp(min=1e-9)
                    vec = (summed / denom).squeeze(0).cpu().numpy()
                embeddings.append(vec)
        return np.stack(embeddings, axis=0)


class TextEmbedderQwen:
    """Класс для получения текстовых эмбеддингов с помощью Qwen2.5-7B-embed-base.

    Эта модель специально обучена для создания семантических эмбеддингов текста
    и лучше подходит для задач поиска, кластеризации и сравнения текстов,
    чем генеративные модели вроде Qwen2.5-VL.
    """

    def __init__(
            self,
            model_name: str = "Qwen/Qwen2.5-7B-embed-base",
            device: str = 'cuda',
            pooling: str = 'mean',
            max_length: int = 512,
            normalize: bool = True
    ):
        """
        Args:
            model_name (str): Название модели в Hugging Face Hub.
            device (str): Устройство для вычислений ('cuda' или 'cpu').
            pooling (str): Стратегия пулинга:
                          - 'last_token': взять эмбеддинг последнего токена
                          - 'mean': усреднить эмбеддинги всех токенов
                          - 'cls': использовать специальный токен [CLS] (если есть)
            max_length (int): Максимальная длина входной последовательности.
            normalize (bool): Нормализовать эмбеддинги (приводить к длине 1).
        """
        print(f"Загрузка модели для эмбеддингов: {model_name}")

        self.device = device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling = pooling
        self.max_length = max_length
        self.normalize = normalize

        # Загрузка токенизатора и модели
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map="auto" if self.device == 'cuda' else None,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

        # Проверяем, поддерживает ли модель специальные токены для эмбеддингов
        self.has_instruction_token = '<|endoftext|>' in self.tokenizer.special_tokens_map.get('eos_token', '')

        print(f"Модель загружена на устройство: {self.device}")
        print(f"Используется пулинг: {pooling}")

    def embed(
            self,
            texts: List[str],
            instruction: Optional[str] = None,
            batch_size: int = 8
    ) -> np.ndarray:
        """Возвращает эмбеддинги для списка текстов.

        Args:
            texts (List[str]): Список текстов для векторизации.
            instruction (Optional[str]): Инструкция для модели (промпт).
                                       Например: "Представь этот текст для поиска: "
            batch_size (int): Размер батча для обработки.

        Returns:
            np.ndarray: Массив эмбеддингов формы (len(texts), hidden_size).
        """
        self.model.eval()
        all_embeddings = []

        # Формируем тексты с инструкцией, если она указана
        processed_texts = []
        if instruction:
            for text in texts:
                # Qwen2.5-embed модели часто используют специальный формат с инструкцией
                processed_texts.append(f"{instruction}\n{text}")
        else:
            processed_texts = texts

        # Обработка батчами для экономии памяти
        with torch.no_grad():
            for i in range(0, len(processed_texts), batch_size):
                batch_texts = processed_texts[i:i + batch_size]

                # Токенизация батча
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                # Прямой проход через модель
                outputs = self.model(**inputs, output_hidden_states=True)

                # Извлекаем скрытые состояния из последнего слоя
                last_hidden_states = outputs.last_hidden_state

                # Применяем выбранную стратегию пулинга
                batch_embeddings = self._apply_pooling(last_hidden_states, inputs['attention_mask'])

                # Нормализуем, если нужно
                if self.normalize:
                    batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

                all_embeddings.append(batch_embeddings.cpu().numpy())

        # Объединяем все батчи
        return np.vstack(all_embeddings)

    def _apply_pooling(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Применяет стратегию пулинга к скрытым состояниям."""

        if self.pooling == 'last_token':
            # Берем эмбеддинг последнего не-паддинг токена
            # Находим индексы последних токенов
            seq_lengths = attention_mask.sum(dim=1) - 1  # -1 для перехода от длины к индексу
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            embeddings = hidden_states[batch_indices, seq_lengths]

        elif self.pooling == 'mean':
            # Средний пулинг с учетом маски внимания
            attention_mask_expanded = attention_mask.unsqueeze(-1).float()
            hidden_states_masked = hidden_states * attention_mask_expanded

            # Суммируем и делим на длину
            sum_embeddings = hidden_states_masked.sum(dim=1)
            seq_lengths = attention_mask_expanded.sum(dim=1).clamp(min=1e-9)
            embeddings = sum_embeddings / seq_lengths

        elif self.pooling == 'cls':
            # Используем первый токен ([CLS] или аналогичный)
            # Для Qwen часто это токен начала последовательности
            embeddings = hidden_states[:, 0, :]

        else:
            raise ValueError(f"Неизвестная стратегия пулинга: {self.pooling}")

        return embeddings

    def embed_single(self, text: str, instruction: Optional[str] = None) -> np.ndarray:
        """Возвращает эмбеддинг для одного текста."""
        return self.embed([text], instruction=instruction)[0]

    def similarity(
            self,
            text1: Union[str, np.ndarray],
            text2: Union[str, np.ndarray],
            instruction: Optional[str] = None
    ) -> float:
        """Вычисляет косинусное сходство между двумя текстами или эмбеддингами.

        Args:
            text1: Первый текст или готовый эмбеддинг.
            text2: Второй текст или готовый эмбеддинг.
            instruction: Инструкция для модели (если передаются тексты).

        Returns:
            float: Косинусное сходство между -1 и 1.
        """
        # Если переданы тексты, получаем их эмбеддинги
        if isinstance(text1, str):
            emb1 = self.embed_single(text1, instruction)
        else:
            emb1 = text1

        if isinstance(text2, str):
            emb2 = self.embed_single(text2, instruction)
        else:
            emb2 = text2

        # Преобразуем в тензоры для вычислений
        emb1_tensor = torch.from_numpy(emb1).float()
        emb2_tensor = torch.from_numpy(emb2).float()

        # Вычисляем косинусное сходство
        similarity = F.cosine_similarity(emb1_tensor.unsqueeze(0), emb2_tensor.unsqueeze(0))

        return similarity.item()

    def get_model_info(self) -> dict:
        """Возвращает информацию о модели."""
        return {
            "model_name": self.model.config._name_or_path,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers,
            "num_attention_heads": self.model.config.num_attention_heads,
            "max_position_embeddings": self.model.config.max_position_embeddings,
            "pooling": self.pooling,
            "normalize": self.normalize
        }


# Дополнительный класс для массовой обработки и кэширования
class TextEmbedderQwenWithCache(TextEmbedderQwen):
    """Расширенная версия с кэшированием эмбеддингов."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        import hashlib
        self.hash_func = hashlib.md5

    def _get_text_hash(self, text: str, instruction: Optional[str] = None) -> str:
        """Генерирует хэш для текста и инструкции."""
        full_text = f"{instruction or ''}|{text}"
        return self.hash_func(full_text.encode()).hexdigest()

    def embed(
            self,
            texts: List[str],
            instruction: Optional[str] = None,
            use_cache: bool = True
    ) -> np.ndarray:
        """Получает эмбеддинги с использованием кэша."""
        if not use_cache:
            return self.embed(texts, instruction)

        results = []
        texts_to_process = []
        indices_to_process = []

        # Проверяем кэш для каждого текста
        for i, text in enumerate(texts):
            text_hash = self._get_text_hash(text, instruction)

            if text_hash in self.cache:
                results.append(self.cache[text_hash])
            else:
                texts_to_process.append(text)
                indices_to_process.append(i)
                # Добавляем placeholder в results
                results.append(None)

        # Обрабатываем только тексты, которых нет в кэше
        if texts_to_process:
            new_embeddings = self.embed(texts_to_process, instruction)

            # Сохраняем в кэш и заполняем результаты
            for idx, (text, embedding) in enumerate(zip(texts_to_process, new_embeddings)):
                text_hash = self._get_text_hash(text, instruction)
                self.cache[text_hash] = embedding
                results[indices_to_process[idx]] = embedding

        # Преобразуем в numpy array
        return np.stack(results)

    def clear_cache(self):
        """Очищает кэш эмбеддингов."""
        self.cache.clear()