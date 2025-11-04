import os
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from helpers import ensure_dir, save_crop
from detectors import YOLOv8Detector
from captioning import Captioner, TextEmbedderBERT as TextEmbedder
from config import PipelineConfig


class VLMPipeline:
    """Основной pipeline для обработки изображений с помощью VLM (YOLOv8 + BLIP + BERT)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        # always use YOLOv8Detector as requested
        self.detector = YOLOv8Detector(device=self.config.device)
        self.captioner = Captioner(device=config.device)
        # use BERT-based contextual embedder
        self.embedder = TextEmbedder(device=config.device)

    def process_image(self, image_path: str) -> list:
        """Обрабатывает одно изображение: детект, кропы, BLIP captions (с учетом prompt), BERT-эмбеддинги."""
        img = Image.open(image_path).convert("RGB")
        # YOLOv8 detects without text queries
        detections, _ = self.detector.detect(img, box_threshold=self.config.box_threshold,
                                             visualize=self.config.visualise)

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

            # caption the crop using BLIP and include the global prompt from config
            caption = self.captioner.describe(Image.open(crop_path).convert("RGB"),
                                              prompt=self.config.prompt,
                                              label=label,
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

        # compute contextual embeddings (BERT) for all captions
        captions = [it["caption"] for it in items]
        embeddings = self.embedder.embed(captions)
        for it, emb in zip(items, embeddings):
            it["text_embedding"] = emb.tolist()

        return items

    def run(self) -> list:
        """Запускает полный цикл обработки изображений"""
        ensure_dir(self.config.out_dir)

        all_metadata = []
        image_files = list(Path(self.config.input_dir).glob("*"))
        image_files = [str(p) for p in image_files if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

        for img_path in tqdm(image_files, desc="Processing images"):
            items = self.process_image(img_path)
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
