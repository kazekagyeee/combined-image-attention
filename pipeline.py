import os
import re
import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from helpers import ensure_dir, save_crop
from detectors import YOLOv8Detector, YOLOv13Detector
from captioning import CaptionerQwen, CaptionerBLIP, TextEmbedderBERT as TextEmbedder
from config import PipelineConfig


def clean_text_from_file(file_path):
    # Читаем файл
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    # Оставляем только буквы, цифры, пробелы и знаки препинания
    cleaned_text = re.sub(r'[^\w\s\.,!?;:()\-—\"\']', '', text)

    # Убираем лишние пробелы
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

    return cleaned_text


class VLMPipeline:
    """Основной pipeline для обработки изображений с помощью VLM (YOLOv8 + BLIP + BERT)."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.detector = YOLOv8Detector(device=self.config.device)
        self.captioner = CaptionerQwen(device=config.device) if config.captioner_model == 'qwen' else CaptionerBLIP(device=config.device)
        # use BERT-based contextual embedder
        self.embedder = TextEmbedder(device=config.device)

    def process_image(self, image_path: str, prompt: str = None) -> list:
        """Обрабатывает одно изображение с возможностью указать индивидуальный промт."""
        img = Image.open(image_path).convert("RGB")

        # Используем переданный промт или глобальный из конфига
        current_prompt = self.config.system_prompt + prompt if prompt is not None else self.config.system_prompt

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
            crop_name = f"{base_name}_crop_{i}.jpg"
            crop_path = os.path.join(self.config.out_dir, crop_name)
            _, (w, h) = save_crop(img, bbox, crop_path)

            # caption the crop using BLIP with current prompt
            caption = self.captioner.describe(
                Image.open(crop_path).convert("RGB"),
                prompt=current_prompt,  # Используем текущий промт
                max_length=self.config.caption_max_length
            )

            items.append({
                "crop_path": os.path.abspath(crop_path),
                "orig_path": os.path.abspath(image_path),
                "bbox": [float(x) for x in bbox],
                "score": float(score),
                "caption": caption,
                "prompt_used": current_prompt,  # Сохраняем использованный промт
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

        for img_path in tqdm(image_files, desc="Processing images with individual prompts"):
            # Получаем промт для текущего изображения
            img_name = Path(img_path).name
            # По контракту название промта такое же как у изображения
            prompt = clean_text_from_file(
                self.config.input_dir + '/' + img_name.replace('.png', '.txt')
            )

            items = self.process_image(img_path, prompt=prompt)
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
