import os
import re
import json
import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from helpers import ensure_dir, save_crop
from detectors import YOLOv8Detector, UIEDDetector
from captioning import CaptionerQwen, CaptionerBLIP, CaptionerGLM, CaptionerGemma3
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

        if config.detector_model == 'yolov8':
            self.detector = YOLOv8Detector(device=self.config.device)
        elif config.detector_model == 'uied_cv':
            self.detector = UIEDDetector(device=self.config.device)
        else:
            self.detector = None
            print("Invalid detector model")

        if config.captioner_model == 'qwen':
            self.captioner = CaptionerQwen(device=config.device)
        elif config.captioner_model == 'blip':
            self.captioner = CaptionerBLIP(device=config.device)
        elif config.captioner_model == 'glm':
            self.captioner = CaptionerGLM(device=config.device)
        elif config.captioner_model == 'gemma':
            self.captioner = CaptionerGemma3(device=config.device)
        else:
            self.captioner = None
            print("Invalid captioner model")

    def process_image(self, image_path: str, prompt: str = None) -> list:
        """Обрабатывает одно изображение с возможностью указать индивидуальный промт."""
        img = Image.open(image_path).convert("RGB")
        
        # Пропуск слишком больших изображений (могут вызвать ошибку токенов в vLLM)
        max_dim = getattr(self.config, 'max_image_size', 2500)
        if max(img.size) > max_dim:
            print(f"⚠️ Пропуск {image_path}: слишком большое разрешение ({img.width}x{img.height})")
            return []

        # Используем переданный промт или глобальный из конфига
        current_prompt = self.config.system_prompt + prompt if prompt is not None else self.config.system_prompt

        det_result = self.detector.detect(img, box_threshold=self.config.box_threshold)
        if not det_result or not isinstance(det_result, tuple) or len(det_result) < 2:
            return []
        
        detections, _ = det_result

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

            caption = self.captioner.describe_two_images(
                img,
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

        return items

    def _save_results(self, all_metadata: list, triplet_data: list):
        """Вспомогательный метод для сохранения результатов в JSON."""
        # save metadata
        out_json = os.path.join(self.config.out_dir, self.config.json_filename)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(all_metadata, f, ensure_ascii=False, indent=2)

        # save triplet data
        triplet_json = os.path.join(self.config.out_dir, "triplet_dataset.json")
        with open(triplet_json, "w", encoding="utf-8") as f:
            json.dump(triplet_data, f, ensure_ascii=False, indent=2)

    def run(self) -> list:
        """Запускает полный цикл обработки изображений с поддержкой чекпоинтов"""
        ensure_dir(self.config.out_dir)

        # 1. Загружаем существующие данные для возобновления (resumption)
        out_json = os.path.join(self.config.out_dir, self.config.json_filename)
        triplet_json = os.path.join(self.config.out_dir, "triplet_dataset.json")
        
        all_metadata = []
        triplet_data = []
        processed_images = set()

        if os.path.exists(out_json):
            try:
                with open(out_json, "r", encoding="utf-8") as f:
                    all_metadata = json.load(f)
                # Определяем, какие изображения уже обработаны
                processed_images = {item["orig_path"] for item in all_metadata}
                print(f"--- Возобновление: загружено {len(processed_images)} уже обработанных изображений. ---")
            except Exception as e:
                print(f"Предупреждение: не удалось загрузить существующий JSON ({e}), начинаем с нуля.")

        if os.path.exists(triplet_json):
            try:
                with open(triplet_json, "r", encoding="utf-8") as f:
                    triplet_data = json.load(f)
            except Exception:
                pass

        image_files = list(Path(self.config.input_dir).glob("*"))
        image_files = [str(p) for p in image_files if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

        import random
        
        newly_processed_count = 0
        for img_path in tqdm(image_files, desc="Processing images"):
            abs_path = os.path.abspath(img_path)
            if abs_path in processed_images:
                continue

            # Получаем промт для текущего изображения
            img_name = Path(img_path).name
            txt_name = Path(img_name).stem + '.txt'

            # Копируем оригинальное (необрезанное) изображение в out_dir
            shutil.copy(img_path, os.path.join(self.config.out_dir, img_name))

            # По контракту текстовый файл имеет то же имя (без расширения) что и изображение
            txt_file = os.path.join(self.config.input_dir, txt_name)
            prompt = ""
            if os.path.exists(txt_file):
                prompt = clean_text_from_file(txt_file)

            items = self.process_image(img_path, prompt=prompt)
            all_metadata.extend(items)

            # Формируем выборку триплетов для обучения
            if len(items) > 1:
                img = Image.open(img_path)
                w_img, h_img = img.size
                
                for item in items:
                    neg_candidates = [other for other in items if other != item]
                    if neg_candidates:
                        neg_item = random.choice(neg_candidates)
                        
                        def normalize_bbox(b):
                            return [
                                max(0.0, min(1.0, b[0]/w_img)), 
                                max(0.0, min(1.0, b[1]/h_img)), 
                                max(0.0, min(1.0, b[2]/w_img)), 
                                max(0.0, min(1.0, b[3]/h_img))
                            ]

                        # Используем относительный путь, чтобы датасет был переносимым
                        rel_img = os.path.join(".", img_name)
                        triplet_data.append({
                            "image_path": rel_img,
                            "text": item["caption"],
                            "pos_bbox": normalize_bbox(item["bbox"]),
                            "neg_bbox": normalize_bbox(neg_item["bbox"])
                        })
            
            processed_images.add(abs_path)
            newly_processed_count += 1

            # Сохраняем чекпоинт каждые 10 новых изображений
            if newly_processed_count % 10 == 0:
                self._save_results(all_metadata, triplet_data)

        # 3. Финальная постобработка и сохранение
        if all_metadata:
            # normalize rel_size_coeff to sum=1 across all found crops
            total_coeff = sum(item["rel_size_coeff"] for item in all_metadata)
            if total_coeff > 0:
                for item in all_metadata:
                    item["rel_size_coeff"] = float(item["rel_size_coeff"] / total_coeff)

        self._save_results(all_metadata, triplet_data)

        print(f"Saved metadata files to {self.config.out_dir}")
        print(f"Total entries in metadata: {len(all_metadata)}")
        print(f"Total entries in triplet dataset: {len(triplet_data)}")
        
        return all_metadata
