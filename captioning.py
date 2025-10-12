from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer


class Captioner:
    """Генератор текстовых описаний для изображений"""

    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device='cpu'):
        print("Loading captioning model:", model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def describe(self, pil_image: Image.Image, max_length=32) -> str:
        """
        Генерирует текстовое описание для изображения

        Args:
            pil_image: PIL Image для обработки
            max_length: Максимальная длина генерируемого описания

        Returns:
            str: Текстовое описание изображения
        """
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


class TextEmbedder:
    """Генератор эмбеддингов для текстовых описаний"""

    def __init__(self, model_name="all-MiniLM-L6-v2", device='cpu'):
        print("Loading text embedder:", model_name)
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list) -> list:
        """
        Генерирует эмбеддинги для списка текстов

        Args:
            texts: Список текстов для обработки

        Returns:
            list: Эмбеддинги текстов
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings