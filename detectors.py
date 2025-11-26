from PIL import Image
from abc import ABC, abstractmethod
from ultralytics import YOLO


class DetectorBase(ABC):
    """Базовый класс для детекторов объектов"""

    def __init__(self, device='cuda'):
        self.device = device

    @abstractmethod
    def detect(self, image: Image.Image, box_threshold=0.3):
        """
        Детектирует объекты на изображении по текстовым запросам.

        Args:
            image: PIL Image для обработки
            box_threshold: Порог уверенности для детектирования

        Returns:
            Список обнаруженных объектов с координатами и метками
        """
        pass


class YOLOv8Detector(DetectorBase):
    """YOLOv8 детектор с поддержкой сегментации"""

    def __init__(self, model_name='yolov8x-seg.pt', device='cuda'):
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