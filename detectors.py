import os
import tempfile

from PIL import Image
from abc import ABC, abstractmethod
from ultralytics import YOLO
from os.path import join as pjoin
from uied_cv import ip_region_proposal as ip


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

    def __init__(self, model_name='yolov8n.pt', device='cuda'):
        super().__init__(device)
        print(f"🧠 Loading YOLOv8 model: {model_name} ...")
        self.model = YOLO(model_name)
        self.model.to(device)

    def detect(self, image: Image.Image, box_threshold=0.3):
        """
        Выполняет сегментацию и детекцию объектов YOLOv8.

        Args:
            image: PIL.Image
            box_threshold: порог уверенности

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

class UIEDDetector(DetectorBase):
    """
    Детектор UI-компонентов на основе UIED (CV метод)
    """

    def __init__(self, device='cpu', resized_height=800, key_params=None):
        super().__init__(device)
        self.resized_height = resized_height
        self.key_params = key_params or {
            'min-grad': 10,
            'ffl-block': 5,
            'min-ele-area': 50,
            'merge-contained-ele': True,
            'merge-line-to-paragraph': True,
            'remove-bar': True
        }

    def detect(self, image: Image.Image, box_threshold=0.3):
        """
        Запускает UIED и возвращает список компонент в виде:
        [
          { "bbox": [x1, y1, x2, y2], "label": "component" }
        ]
        """

        # Определяет находятся ли сегменты рядом
        def boxes_close(a, b, max_dist=15):
            # вертикальное перекрытие
            vert_overlap = min(a[3], b[3]) - max(a[1], b[1])
            if vert_overlap <= 0:
                return False

            # горизонтальное расстояние
            dist = min(abs(a[0] - b[2]), abs(b[0] - a[2]))

            return dist < max_dist

        # Объединяет рядом стоящие сегменты
        def merge_uied_boxes(boxes):
            merged = True
            while merged:
                merged = False
                new = []
                while boxes:
                    a = boxes.pop(0)
                    merged_with_a = False

                    for i, b in enumerate(boxes):
                        if boxes_close(a, b, max_dist=20):
                            nx1 = min(a[0], b[0])
                            ny1 = min(a[1], b[1])
                            nx2 = max(a[2], b[2])
                            ny2 = max(a[3], b[3])
                            new.append([nx1, ny1, nx2, ny2])
                            boxes.pop(i)
                            merged_with_a = True
                            merged = True
                            break

                    if not merged_with_a:
                        new.append(a)

                boxes = new

            return boxes

        # --- 1) сохраняем изображение во временную папку ---
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = pjoin(tmpdir, "input.png")
            output_dir = pjoin(tmpdir, "out")
            os.makedirs(output_dir, exist_ok=True)

            image.save(input_path)

            # --- 3) Запуск UIED ---
            ip.compo_detection(
                input_img_path=input_path,
                output_root=output_dir,
                uied_params=self.key_params,
                classifier=None,
                resize_by_height=None,
                show=False
            )

            # --- 4) UIED сохраняет промежуточные файлы ---
            # основной результат лежит в out/ip/compo.json
            input_name = os.path.splitext(os.path.basename(input_path))[0]
            compo_json_path = pjoin(output_dir, "ip", f"{input_name}.json")

            if not os.path.exists(compo_json_path):
                print("⚠️ UIED did not generate compo.json")
                return [], image

            # --- 5) Читаем JSON и приводим к формату DetectorBase ---
            import json
            with open(compo_json_path, "r") as f:
                compo_info = json.load(f)

            raw_boxes = []
            for comp in compo_info.get("compos", []):
                x1, y1, x2, y2 = comp["column_min"], comp["row_min"], comp["column_max"], comp["row_max"]
                raw_boxes.append([x1, y1, x2, y2])

            merged_boxes = merge_uied_boxes(raw_boxes)

            results = [
                {"bbox": b, "label": "component", "score": 1.0}
                for b in merged_boxes
            ]

            return results, image