import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from helpers import ensure_dir, save_crop
from detectors import OwlViTDetector, GroundingDINOPlaceholder, HAS_GROUNDING_DINO, YOLOv8Detector
from captioning import Captioner, TextEmbedder
from config import PipelineConfig


class VLMPipeline:
    """Основной pipeline для обработки изображений с помощью VLM"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = self._init_detector()
        self.captioner = Captioner(device=config.device)
        self.embedder = TextEmbedder(device=config.device)

    def _init_detector(self):
        """Инициализирует детектор на основе конфигурации"""
        if self.config.model.lower() == "owlvit":
            return OwlViTDetector(device=self.config.device)
        elif self.config.model.lower() == "groundingdino":
            if not HAS_GROUNDING_DINO:
                raise RuntimeError("Grounding DINO не доступен. Установи его или используй owlvit.")
            return GroundingDINOPlaceholder(device=self.config.device)
        elif self.config.model.lower() == "yolov8":
            return YOLOv8Detector(device=self.config.device)
        else:
            raise ValueError("Unknown detector backend")

    def process_image(self, image_path: str) -> list:
        """
        Обрабатывает одно изображение

        Args:
            image_path: Путь к изображению для обработки

        Returns:
            list: Список обработанных объектов с метаданными
        """
        img = Image.open(image_path).convert("RGB")
        prompt_queries = [q.strip() for q in self.config.prompt.split(",") if q.strip()]
        detections, _ = self.detector.detect(img, prompt_queries, box_threshold=self.config.box_threshold,
                                             visualize=self.config.visualise)

        # if no detections, return empty
        if not detections:
            return []

        # sort by score descending
        detections = sorted(detections, key=lambda x: x["score"], reverse=True)

        # compute areas and coefficients
        areas = []
        for d in detections:
            x0, y0, x1, y1 = d["bbox"]
            w = max(0, x1 - x0)
            h = max(0, y1 - y0)
            areas.append(w * h)
        total_area = sum(areas) if sum(areas) > 0 else 1.0
        coeffs = [a / total_area for a in areas]

        items = []
        base_name = Path(image_path).stem
        for i, (d, coeff) in enumerate(zip(detections, coeffs)):
            bbox = d["bbox"]
            score = d["score"]
            label = d.get("label", "")
            crop_name = f"{base_name}_crop_{i}.jpg"
            crop_path = os.path.join(self.config.out_dir, crop_name)
            _, (w, h) = save_crop(img, bbox, crop_path)

            # caption the crop
            caption = self.captioner.describe(Image.open(crop_path).convert("RGB"),
                                              max_length=self.config.caption_max_length)

            items.append({
                "crop_path": os.path.abspath(crop_path),
                "orig_path": os.path.abspath(image_path),
                "bbox": [float(x) for x in bbox],
                "score": float(score),
                "label": label,
                "caption": caption,
                "area": float(areas[i]),
                "rel_size_coeff": float(coeff),
                "crop_wh": [w, h]
            })

        # compute embeddings for all captions
        captions = [it["caption"] for it in items]
        embeddings = self.embedder.embed(captions)
        for it, emb in zip(items, embeddings):
            it["text_embedding"] = emb.tolist()  # numpy -> list to save to json

        return items

    def run(self) -> list:
        """
        Запускает полный цикл обработки изображений

        Returns:
            list: Список всех обработанных объектов с метаданными
        """
        ensure_dir(self.config.out_dir)

        all_metadata = []
        image_files = list(Path(self.config.input_dir).glob("*"))
        image_files = [str(p) for p in image_files if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

        for img_path in tqdm(image_files, desc="Processing images"):
            items = self.process_image(img_path)
            # append items
            all_metadata.extend(items)

        # normalize rel_size_coeff to sum=1 across all found crops
        if all_metadata:
            total_coeff = sum(item["rel_size_coeff"] for item in all_metadata)
            if total_coeff > 0:
                for item in all_metadata:
                    item["rel_size_coeff"] = float(item["rel_size_coeff"] / total_coeff)

        # save metadata
        out_json = os.path.join(self.config.out_dir, self.config.json_filename)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        print(f"Saved metadata to {out_json} — {len(all_metadata)} crops total.")
        return all_metadata