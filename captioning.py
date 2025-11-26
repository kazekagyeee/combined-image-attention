from abc import ABC, abstractmethod

from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, Blip2Processor, Blip2ForConditionalGeneration, AutoProcessor, AutoTokenizer, AutoModel
import torch
import numpy as np

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

class CaptionerBLIP(CaptionerBase):
    """Генератор текстовых описаний для изображений (BLIP)."""

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device='cuda'):
        print("Loading captioning model:", model_name)
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

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

    def __init__(self, model_name = "Qwen/Qwen2-VL-7B-Instruct", device='cuda'):
        super().__init__(device)
        print("Loading Qwen2‑VL model:", model_name)
        # Загружаем процессор (tokenizer + визуальную часть)
        self.processor = AutoProcessor.from_pretrained(model_name)
        # Загружаем модель
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
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
        output = self.model.generate(**inputs, max_new_tokens=max_length)
        # Так как в входе был префикс (prompt + текст-системные токены), отделим его
        # (Qwen выдаёт весь sequence, поэтому нужно взять только сгенерированную часть)
        # inputs.input_ids — это input ids, а output — полные generated ids
        generated_ids = [
            out_ids[len(input_ids):]  # отрезаем префикс
            for input_ids, out_ids in zip(inputs.input_ids, output)
        ]

        # Декодируем в текст
        caption = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]

        return caption


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
