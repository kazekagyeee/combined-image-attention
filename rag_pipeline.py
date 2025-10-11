"""
vlm_detector_pipeline.py

Функции:
1) детектит с помощью VLM (OWL-ViT или Grounding DINO, опция в config)
2) сохраняет выделенные bbox как подкартинки
3) делает текстовые описания к подкартинкам через выбранную VLM (captioning)
4) получает эмбеддинги подкартинок ПО ТЕКСТОВОМУ ОПИСАНИЮ и сохраняет метаданные в JSON

Запуск:
python vlm_detector_pipeline.py --model owlvit --input_dir ./images --out_dir ./out --prompt "a dog, a cat"
"""

import os
import json
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torchvision.transforms as T

# transformers for OWL-ViT (object detection) and BLIP (captioning)
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import BlipProcessor, BlipForConditionalGeneration

# sentence-transformers for textual embeddings
from sentence_transformers import SentenceTransformer

# try optional GroundingDINO import (user may need to install separately)
HAS_GROUNDING_DINO = False
try:
    # If groundingdino package or local module is installed, import predictor
    # from groundingdino.predictor import Predictor  # common in many forks
    # but names vary by repo; we'll attempt a couple
    from groundingdino.models import build_model as gd_build_model  # type: ignore
    HAS_GROUNDING_DINO = True
except Exception:
    # Not installed — it's optional. We'll fallback to OWL-ViT.
    HAS_GROUNDING_DINO = False


# ---------------------------
# Helpers
# ---------------------------
def ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def save_crop(img: Image.Image, bbox, dest_path: str):
    # bbox = (x_min, y_min, x_max, y_max) in pixels
    x0, y0, x1, y1 = map(int, bbox)
    crop = img.crop((x0, y0, x1, y1))
    crop.save(dest_path)
    return dest_path, crop.size  # return path and (w,h)


# ---------------------------
# Detectors (abstract + implementations)
# ---------------------------

class DetectorBase:
    def __init__(self, device='cpu'):
        self.device = device


class OwlViTDetector(DetectorBase):
    def __init__(self, model_name='google/owlvit-base-patch32', device='cpu'):
        super().__init__(device)
        print(f"🔍 Loading OWL-ViT model: {model_name} ...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)

    def detect(self, image: Image.Image, text_queries: list, box_threshold=0.3):
        """
        Детектирует объекты на изображении по текстовым запросам.
        Возвращает список:
        [
            {"bbox": (x0, y0, x1, y1), "score": float, "label": "cat" (or text query matched)}
        ]
        """

        # 1️⃣ Подготовка данных для модели
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.model.device)

        # 2️⃣ Прогон через модель
        with torch.no_grad():
            outputs = self.model(**inputs)

        # 3️⃣ Постобработка (конвертация предсказаний в реальные координаты)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.model.device)  # (H, W)
        results = self.processor.post_process_object_detection(
            outputs=outputs,
            threshold=box_threshold,
            target_sizes=target_sizes
        )[0]

        # 4️⃣ Формирование списка детекций
        detections = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections.append({
                "bbox": tuple(box.tolist()),           # координаты (x0, y0, x1, y1)
                "score": float(score.item()),           # уверенность
                "label": text_queries[label] if label < len(text_queries) else str(label)  # текст запроса
            })

        return detections


class OwlViTDetector(DetectorBase):
    def __init__(self, model_name='google/owlvit-base-patch32', device='cpu'):
        super().__init__(device)
        print("Loading OWL-ViT:", model_name)
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(device)

    def detect(self, image: Image.Image, text_queries: list, box_threshold=0.3):
        # prepare queries prompt as list of strings
        inputs = self.processor(text=text_queries, images=image, return_tensors="pt").to(self.model.device)
        outputs = self.model(**inputs)

        # convert outputs (following HF OwlViT example)
        target_sizes = torch.tensor([image.size[::-1]]).to(self.model.device)  # (H,W)
        results = self.processor.post_process_object_detection(outputs=outputs, threshold=box_threshold, target_sizes=target_sizes)[0]
        detections = []
        for score, label, box in zip(results["scores"].tolist(), results["labels"].tolist(), results["boxes"].tolist()):
            label_str = self.model.config.id2label[label] if hasattr(self.model.config, "id2label") else str(label)
            # boxes are [x0, y0, x1, y1] in pixels
            detections.append({"bbox": tuple(box), "score": float(score), "label": label_str})
        return detections


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
        raise NotImplementedError("Требуется реализация Grounding DINO. Сейчас используй OWL-ViT или установи Grounding DINO.")


# ---------------------------
# Captioning and Embeddings
# ---------------------------
class Captioner:
    def __init__(self, model_name="Salesforce/blip-image-captioning-base", device='cpu'):
        print("Loading captioning model:", model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def describe(self, pil_image: Image.Image, max_length=32):
        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


class TextEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2", device='cpu'):
        print("Loading text embedder:", model_name)
        self.model = SentenceTransformer(model_name, device=device)

    def embed(self, texts: list):
        return self.model.encode(texts, convert_to_numpy=True)


# ---------------------------
# Pipeline
# ---------------------------
def process_image(path, detector, captioner, embedder, out_dir, prompt_queries):
    img = Image.open(path).convert("RGB")
    detections = detector.detect(img, prompt_queries)

    # if no detections, return empty
    if not detections:
        return []

    # sort by score descending
    detections = sorted(detections, key=lambda x: x["score"], reverse=True)

    # compute areas and coefficients
    areas = []
    for d in detections:
        x0, y0, x1, y1 = d["bbox"]
        w = max(0, x1 - x0); h = max(0, y1 - y0)
        areas.append(w * h)
    total_area = sum(areas) if sum(areas) > 0 else 1.0
    coeffs = [a / total_area for a in areas]

    items = []
    base_name = Path(path).stem
    for i, (d, coeff) in enumerate(zip(detections, coeffs)):
        bbox = d["bbox"]
        score = d["score"]
        label = d.get("label", "")
        crop_name = f"{base_name}_crop_{i}.jpg"
        crop_path = os.path.join(out_dir, crop_name)
        _, (w,h) = save_crop(img, bbox, crop_path)

        # caption the crop
        caption = captioner.describe(Image.open(crop_path).convert("RGB"))

        items.append({
            "crop_path": os.path.abspath(crop_path),
            "orig_path": os.path.abspath(path),
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
    embeddings = embedder.embed(captions)
    for it, emb in zip(items, embeddings):
        it["text_embedding"] = emb.tolist()  # numpy -> list to save to json
    return items


def run_folder(input_dir, out_dir, detector_backend="owlvit", prompt="person, dog, cat", device="cpu", json_out="metadata.json"):
    ensure_dir(out_dir)
    # init detector
    if detector_backend.lower() == "owlvit":
        detector = OwlViTDetector(device=device)
    elif detector_backend.lower() == "groundingdino":
        if not HAS_GROUNDING_DINO:
            raise RuntimeError("Grounding DINO не доступен. Установи его или используй owlvit.")
        detector = GroundingDINOPlaceholder(device=device)
    else:
        raise ValueError("Unknown detector backend: choose 'owlvit' or 'groundingdino'")

    captioner = Captioner(device=device)
    embedder = TextEmbedder(device=device)

    all_metadata = []
    image_files = list(Path(input_dir).glob("*"))
    image_files = [str(p) for p in image_files if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]

    prompt_queries = [q.strip() for q in prompt.split(",") if q.strip()]

    for img_path in tqdm(image_files, desc="Processing images"):
        items = process_image(img_path, detector, captioner, embedder, out_dir, prompt_queries)
        # append items
        all_metadata.extend(items)

    # normalize rel_size_coeff to sum=1 across all found crops (required by user)
    # current pipeline computed per-image coeffs; user asked sum of coefficients = 1 across obtained subimages.
    # We'll normalize across ALL found items.
    if all_metadata:
        total_coeff = sum(item["rel_size_coeff"] for item in all_metadata)
        if total_coeff > 0:
            for item in all_metadata:
                item["rel_size_coeff"] = float(item["rel_size_coeff"] / total_coeff)

    # save metadata
    out_json = os.path.join(out_dir, json_out)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    print(f"Saved metadata to {out_json} — {len(all_metadata)} crops total.")
    return all_metadata


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["owlvit", "groundingdino"], default="owlvit", help="detector backend")
    p.add_argument("--input_dir", required=True, help="folder with images")
    p.add_argument("--out_dir", required=True, help="output folder for crops and metadata")
    p.add_argument("--prompt", default="person, dog, cat", help="comma-separated object queries")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    p.add_argument("--json", default="metadata.json", help="json filename for metadata")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_folder(args.input_dir, args.out_dir, detector_backend=args.model, prompt=args.prompt, device=args.device, json_out=args.json)
