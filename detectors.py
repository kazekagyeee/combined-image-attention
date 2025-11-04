import torch
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from abc import ABC, abstractmethod
from ultralytics import YOLO

# Попытка импортировать GroundingDINO
HAS_GROUNDING_DINO = False
try:
    from groundingdino.models import build_model as gd_build_model  # type: ignore

    HAS_GROUNDING_DINO = True
except Exception:
    HAS_GROUNDING_DINO = False


class DetectorBase(ABC):
    """Базовый класс для детекторов объектов"""

    def __init__(self, device='gpu'):
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

    def __init__(self, model_name='google/owlv2-base-patch16-ensemble', device='gpu'):
        super().__init__(device)
        print(f"🔍 Loading OWL-ViT model: {model_name} ...")
        self.processor = Owlv2Processor.from_pretrained(model_name)
        self.model = Owlv2ForObjectDetection.from_pretrained(model_name).to(device)

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

        return detections, image


class YOLOv8Detector(DetectorBase):
    """YOLOv8 детектор с поддержкой сегментации"""

    def __init__(self, model_name='yolov8x-seg.pt', device='gpu'):
        super().__init__(device)
        print(f"🧠 Loading YOLOv8 model: {model_name} ...")
        self.model = YOLO(model_name)
        self.model.to(device)

    def detect(self, image: Image.Image, box_threshold=0.3, visualize=True):
        """
        Выполняет сегментацию и детекцию объектов YOLOv8.

        Args:
            image: PIL.Image
            box_threshold: порог уверенности
            visualize: рисовать ли результат

        Returns:
            detections: [
                {"bbox": (x0, y0, x1, y1), "score": float, "label": str, "mask": np.ndarray | None}
            ],
            annotated_image: PIL.Image (если visualize=True)
        """
        results = self.model.predict(image, conf=box_threshold, device=self.device, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            masks = getattr(r, 'masks', None)
            names = self.model.names

            for i, box in enumerate(boxes):
                xyxy = box.xyxy[0].tolist()
                score = float(box.conf.item())
                cls = int(box.cls.item())
                label = names.get(cls, str(cls))

                mask = None
                if masks is not None and len(masks.data) > i:
                    mask = masks.data[i].cpu().numpy()

                detections.append({
                    "bbox": tuple(map(float, xyxy)),
                    "score": score,
                    "label": label,
                    "mask": mask  # np.ndarray (H, W) или None
                })

        return detections, image

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