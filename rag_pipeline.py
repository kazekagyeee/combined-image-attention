
"""
rag_pipeline.py
Prototype RAG pipeline for images:
- Semantic encoder: CLIP (preferred) or ViT/ResNet fallback
- Morphological / visual-quality encoder: DISTS or LPIPS or VGG low-level fallback (style features)
- Optional object segmentation: SAM (optional; used to extract object "sub-images")
- Stores embeddings (semantic + morph) per image in JSON (configurable)
- Query pipeline: compute query embeddings and perform retrieval using cosine similarity
- Evaluation: precision@k for retrieval on a small labeled test set (if available)

How to run (example):
    python3 rag_pipeline.py --input_dir /path/to/images --out_json /path/to/embeddings.json --index_json /path/to/index.json

Notes:
- This script tries to import modern libraries (open_clip, transformers CLIP, lpips, dists, segment_anything).
  If they are not available it will fall back to torchvision.vgg features + HOG (skimage) where possible.
- For reproducible research-quality runs you should install:
    pip install torch torchvision ftfy regex tqdm transformers open-clip-torch lpips faiss-cpu segment-anything dists
  (faiss optional if you want fast nearest neighbor search)
- The script is a prototype and includes TODOs for paper-level improvements (fine-tuning, balanced datasets, SVD ideas described in the diagram).
"""

import os
import sys
import json
import argparse
import math
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Try to import preferred encoders and fall back sensibly

import torch
from torchvision import transforms, models
import clip  # openai/clip (if installed)
import open_clip  # open_clip_torch
import lpips  # perceptual metric; we might use its backbones
import dists  # if you installed DISTS implementation as package (optional)

# segment anything placeholder
# SAM import path varies; user must install segment-anything

from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# scikit-image for HOG fallback
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import io as skio

# Normalization helpers
def normalize_vec(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm

# ----------------------------- Feature extractors -----------------------------

class SemanticEncoder:
    """
    Semantic encoder interface. Preferred: CLIP (openai/open_clip). Fallback: pretrained torchvision ResNet or ViT.
    Provides get_embedding(image_pil) -> np.ndarray (L2-normalized)
    """
    def __init__(self, device='cpu', model_name=None):
        self.device = device
        self.model = None
        self.processor = None
        self.model_type = None

        # Try openai clip
        if _has_clip:
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=device)
                self.model_type = 'openai_clip'
                print("[SemanticEncoder] using openai/clip ViT-B/32")
                return
            except Exception as e:
                print("[SemanticEncoder] failed to load openai clip:", e)

        # Try open_clip
        if _has_open_clip:
            try:
                self.model, _, self.preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k')
                self.model_type = 'open_clip'
                self.model.to(device)
                print("[SemanticEncoder] using open_clip ViT-B-32")
                return
            except Exception as e:
                print("[SemanticEncoder] failed to load open_clip:", e)

        # Fallback to torchvision ResNet50 pooled features (global average pool)
        print("[SemanticEncoder] falling back to torchvision ResNet50")
        resnet = models.resnet50(pretrained=True)
        # Remove final classifier, keep avgpool output (2048)
        modules = list(resnet.children())[:-1]
        self.model = torch.nn.Sequential(*modules).to(device)
        self.model.eval()
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.model_type = 'resnet50'

    def get_embedding(self, pil_img) -> np.ndarray:
        img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if self.model_type in ('openai_clip', 'open_clip'):
                # CLIP returns already pooled features from model.encode_image
                emb = self.model.encode_image(img_t)
                emb = emb.cpu().numpy()[0]
            else:
                out = self.model(img_t)
                out = out.squeeze().cpu().numpy()
                emb = out.reshape(-1)
        return normalize_vec(emb.astype(np.float32))

class MorphEncoder:
    """
    Morphological/visual-quality encoder. Preferred: DISTS (if available) or LPIPS backbones.
    Fallback: VGG low-level aggregated features or classical HOG descriptor.
    Provides get_embedding(image_pil) -> np.ndarray
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.model = None
        self.mode = None

        if _has_dists:
            try:
                # user must supply a dists model import; placeholder
                print("[MorphEncoder] DISTS available (user must provide model wrapper). Using DISTS is recommended.")
                self.mode = 'dists'
            except Exception:
                self.mode = None

        if _has_lpips and self.mode is None:
            try:
                # Use LPIPS with VGG backbone to extract intermediate features; we will use the internal network
                loss_fn = lpips.LPIPS(net='vgg').to(device)
                self.model = loss_fn
                self.mode = 'lpips_vgg'
                print("[MorphEncoder] using LPIPS (vgg)")
            except Exception as e:
                print("[MorphEncoder] failed to init LPIPS:", e)

        if self.mode is None:
            # Fallback to VGG low-level conv feature aggregation
            vgg = models.vgg19(pretrained=True).features.to(device).eval()
            self.model = vgg
            self.mode = 'vgg_lowlevel'
            self.vgg_layers = [1, 6, 11]  # conv1_1, conv2_1, conv3_1-like indices in features
            self.preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            print("[MorphEncoder] fallback to VGG low-level features")

    def get_embedding(self, pil_img) -> np.ndarray:
        # If LPIPS available, we can feed image and extract intermediate features via the lpips model
        if self.mode == 'lpips_vgg' and _has_lpips:
            # LPIPS API primarily returns distance; to get features we'd need to hack internal nets.
            # For prototype, we compute LPIPS distance to a blurred version (to produce a scalar feature)
            import torchvision.transforms.functional as TF
            img_t = TF.to_tensor(pil_img).unsqueeze(0).to(self.device)
            # create a blurred variant
            blur = pil_img.filter(getattr(__import__('PIL').ImageFilter, 'GaussianBlur')(radius=2))
            blur_t = TF.to_tensor(blur).unsqueeze(0).to(self.device)
            dist = self.model.forward(img_t, blur_t).cpu().numpy().reshape(-1)
            return normalize_vec(dist.astype(np.float32))

        if self.mode == 'vgg_lowlevel':
            img_t = self.preprocess(pil_img).unsqueeze(0).to(self.device)
            h = img_t
            feats = []
            with torch.no_grad():
                for i, layer in enumerate(self.model):
                    h = layer(h)
                    if i in self.vgg_layers:
                        # global average pool on this feature map
                        f = h.mean(dim=[2,3]).squeeze().cpu().numpy()
                        feats.append(f)
            emb = np.concatenate(feats).astype(np.float32)
            return normalize_vec(emb)

        if self.mode == 'dists':
            # Placeholder: user to implement dists extraction
            raise NotImplementedError("DISTS mode selected but extraction function is not implemented in prototype.")

        # As final fallback, HOG descriptor
        if _has_skimage:
            img_np = np.array(pil_img.convert('RGB'))
            gray = rgb2gray(img_np)
            hog_feat = hog(gray, pixels_per_cell=(16,16), cells_per_block=(2,2))
            return normalize_vec(hog_feat.astype(np.float32))

        # If nothing, error
        raise RuntimeError("No morphological encoder available. Install lpips, skimage or use VGG fallback.")

# ----------------------------- Storage & Retrieval -----------------------------

def store_embeddings_json(out_path: str, entries: List[Dict]):
    """
    entries: list of dicts: { 'image_path': str, 'semantic': [..], 'morph': [..], 'meta': {...} }
    """
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[store] saved {len(entries)} entries to {out_path}")

def load_embeddings_json(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None: return -1.0
    a = a / (np.linalg.norm(a)+1e-12)
    b = b / (np.linalg.norm(b)+1e-12)
    return float(np.dot(a, b))

def retrieve_top_k(query_sem: np.ndarray, query_morph: np.ndarray, db_entries: List[Dict], k=5, alpha=0.5):
    """
    Simple weighted retrieval: combined score = alpha * sim(sem) + (1-alpha) * sim(morph)
    Returns list of (score, entry)
    """
    scores = []
    for e in db_entries:
        sem = np.array(e['semantic'], dtype=np.float32)
        morph = np.array(e['morph'], dtype=np.float32)
        s_sim = cosine_sim(query_sem, sem)
        m_sim = cosine_sim(query_morph, morph)
        score = alpha * s_sim + (1.0 - alpha) * m_sim
        scores.append((score, e))
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[:k]

# ----------------------------- Evaluation -----------------------------

def precision_at_k(retrieved: List[Dict], ground_truth: List[str], k=5):
    """
    retrieved: list of entries with 'image_path'
    ground_truth: list of true similar image paths
    """
    retrieved_paths = [r['image_path'] for _, r in retrieved[:k]]
    hits = sum(1 for p in retrieved_paths if p in ground_truth)
    return hits / k

# ----------------------------- Main pipeline -----------------------------

def build_index(input_dir: str, out_json: str, device='cpu', max_images=None):
    from PIL import Image
    input_dir = Path(input_dir)
    files = sorted([p for p in input_dir.glob('*') if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if max_images:
        files = files[:max_images]
    sem_enc = SemanticEncoder(device=device)
    morph_enc = MorphEncoder(device=device)
    entries = []
    for i, p in enumerate(files):
        try:
            img = Image.open(p).convert('RGB')
            sem = sem_enc.get_embedding(img)
            morph = morph_enc.get_embedding(img)
            entries.append({
                'image_path': str(p.resolve()),
                'semantic': sem.tolist(),
                'morph': morph.tolist(),
                'meta': {}
            })
            if (i+1) % 10 == 0:
                print(f"[build_index] processed {i+1}/{len(files)} images")
        except Exception as e:
            print(f"[build_index] error processing {p}: {e}")
    store_embeddings_json(out_json, entries)
    return out_json

def query_and_eval(index_json: str, query_image_path: str, k=5, alpha=0.5, device='cpu'):
    from PIL import Image
    data = load_embeddings_json(index_json)
    sem_enc = SemanticEncoder(device=device)
    morph_enc = MorphEncoder(device=device)
    img = Image.open(query_image_path).convert('RGB')
    qsem = sem_enc.get_embedding(img)
    qmorph = morph_enc.get_embedding(img)
    topk = retrieve_top_k(qsem, qmorph, data, k=k, alpha=alpha)
    print(f"[query] top-{k} results for query {query_image_path}:")
    for score, e in topk:
        print(f"  score={score:.4f} path={e['image_path']}")
    return topk

# ----------------------------- CLI -----------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Build RAG-style image embeddings (semantic + morph) and run queries")
    p.add_argument('--input_dir', type=str, default='images', help='directory with images to index')
    p.add_argument('--out_json', type=str, default='embeddings.json', help='output JSON file with embeddings')
    p.add_argument('--max_images', type=int, default=200, help='max images to process')
    p.add_argument('--query', type=str, default=None, help='path to query image to run example retrieval')
    p.add_argument('--k', type=int, default=5, help='top-k retrieval')
    p.add_argument('--alpha', type=float, default=0.5, help='weight for semantic vs morph similarity (0..1)')
    p.add_argument('--device', type=str, default='cpu', help='torch device to use')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    # Ensure input dir exists
    if not Path(args.input_dir).exists():
        print("Input dir not found. Please provide --input_dir with images.")
        sys.exit(1)
    print(f"[main] building index for images in {args.input_dir} -> {args.out_json}")
    build_index(args.input_dir, args.out_json, device=args.device, max_images=args.max_images)
    if args.query:
        print("[main] running query")
        topk = query_and_eval(args.out_json, args.query, k=args.k, alpha=args.alpha, device=args.device)
        # no automatic ground-truth evaluation here; user can supply appropriate labels and implement precision_at_k externally
