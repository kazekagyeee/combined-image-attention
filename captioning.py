from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration, AutoTokenizer, AutoModel
import torch
import numpy as np

class Captioner:
    """Генератор текстовых описаний для изображений (BLIP)."""

    def __init__(self, model_name="Salesforce/blip2-flan-t5-xl", device='cuda'):
        print("Loading captioning model:", model_name)
        self.processor = Blip2Processor.from_pretrained(model_name, use_fast=True)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(device)
        self.device = device

    def describe(self, image: Image.Image, prompt: str = None, label: str = None, max_length=128) -> str:
        """Генерирует описание для изображения. При наличии `prompt` — учитывает его."""
        if prompt is None:
            inputs = self.processor(images=image, return_tensors="pt").to(self.model.device)
        else:
            # BLIP supports conditional generation with a prompt
            prompt = prompt.replace("{LABEL}", label)
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(**inputs, max_new_tokens=max_length)
        caption = self.processor.decode(out[0], skip_special_tokens=True)
        return caption


class TextEmbedderBERT:
    """Контекстно-зависимые эмбеддинги текста с помощью BERT (CLS pooling или mean pooling)."""

    def __init__(self, model_name="bert-base-uncased", device='cpu', pooling='cls'):
        print("Loading text embedder (BERT):", model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.device = device if isinstance(device, str) else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.pooling = pooling

    def embed(self, texts: list) -> list:
        """Возвращает numpy массив эмбеддингов для списка текстов."""
        self.model.eval()
        embeddings = []
        with torch.no_grad():
            for t in texts:
                enc = self.tokenizer(t, return_tensors='pt', truncation=True, padding=True).to(self.model.device)
                out = self.model(**enc)
                last_hidden = out.last_hidden_state  # (1, seq_len, hidden)
                attention_mask = enc['attention_mask'].unsqueeze(-1)  # (1, seq_len, 1)
                if self.pooling == 'cls' and hasattr(last_hidden, 'size'):
                    vec = last_hidden[:,0,:]  # CLS token
                    vec = vec.squeeze(0).cpu().numpy()
                else:
                    # mean pooling with attention mask
                    masked = last_hidden * attention_mask
                    summed = masked.sum(dim=1)
                    denom = attention_mask.sum(dim=1).clamp(min=1e-9)
                    vec = (summed / denom).squeeze(0).cpu().numpy()
                embeddings.append(vec)
        return np.stack(embeddings, axis=0)
