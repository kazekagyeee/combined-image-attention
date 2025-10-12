import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from abc import ABC, abstractmethod

# Попытка импортировать GroundingDINO
HAS_GROUNDING_DINO = False
try:
    from groundingdino.models import build_model as gd_build_model  # type: ignore

    HAS_GROUNDING_DINO = True
except Exception:
    HAS_GROUNDING_DINO = False


class DetectorBase(ABC):
    """Базовый класс для детекторов объектов"""

    def __init__(self, device='cpu'):
        self.device = device

    @abstractmethod
    def detect(self, image: Image.Image, text_queries: list, box_threshold=0.3):
        """
        Детектирует объекты на изображении по текстовым запросам.

        Args:
            image: PIL Image для обработки
            text_queries: Список текстовых запросов для поиска объектов
            box_threshold: Порог уверенности для детектирования

        Returns:
            Список обнаруженных объектов с координатами и метками
        """
        pass


class OwlViTDetector(DetectorBase):
    """Детектор объектов на основе OWL-ViT"""

    def __init__(self, model_name='google/owlvit-base-patch32', device='gpu'):
        super().__init__(device)
        print(f"🔍 Loading OWL-ViT model: {model_name} ...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)

    def detect(self, image: Image.Image, text_queries: list, box_threshold=0.3, visualize=True):
        """
        Детектирует объекты на изображении по текстовым запросам.

        Args:
            image: PIL Image для обработки
            text_queries: Список текстовых запросов для поиска объектов
            box_threshold: Порог уверенности для детектирования
            visualize: Флаг для визуализации результатов

        Returns:
            detections: [
                {"bbox": (x0, y0, x1, y1), "score": float, "label": "cat"}
            ],
            annotated_image: PIL.Image (если visualize=True)
        """
        # 1️⃣ Подготовка данных
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.model.device)

        # 2️⃣ Прогон через модель
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3️⃣ Постобработка
        target_sizes = torch.tensor([image.size[::-1]]).to(self.model.device)  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "bbox": tuple(map(float, box.tolist())),
                "score": float(score.item()),
                "label": text_queries[label] if label < len(text_queries) else str(label)
            })

        # 4️⃣ Визуализация (опционально)
        annotated_image = image.copy()
        if visualize:
            draw = ImageDraw.Draw(annotated_image)
            try:
                font = ImageFont.truetype("arial.ttf", size=16)
            except:
                font = ImageFont.load_default()

            for det in detections:
                x0, y0, x1, y1 = det["bbox"]
                label = det["label"]
                score = det["score"]
                color = (255, 0, 0)  # красный bbox
                draw.rectangle([x0, y0, x1, y1], outline=color, width=3)
                text = f"{label} {score:.2f}"
                bbox = draw.textbbox((x0, y0 - 16), text, font=font)
                draw.rectangle(bbox, fill=color)
                draw.text((x0, y0 - 16), text, fill="white", font=font)

        return detections, annotated_image


class GroundingDINOPlaceholder(DetectorBase):
    """
    Заглушка/скелет для Grounding DINO.
    Чтобы использовать реальную Grounding DINO, установи её репозиторий и импорты,
    затем замени логику в detect() на вызовы из реализации GroundingDINO.predictor.
    """

    def __init__(self, device='cpu'):
        super().__init__(device)
        if not HAS_GROUNDING_DINO:
            raise RuntimeError("Grounding DINO не установлен. Установи repo IDEA-Research/GroundingDINO и повтори.")
        # TODO: инициализация реальной модели

    def detect(self, image: Image.Image, text_queries: list, box_threshold=0.3):
        # TODO: заменить реальным вызовом Grounding DINO
        raise NotImplementedError(
            "Требуется реализация Grounding DINO. Сейчас используй OWL-ViT или установи Grounding DINO.")