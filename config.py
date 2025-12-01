from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Конфигурация для VLM pipeline"""
    detector_model: str = "uied_cv" # yolov8/uied_cv
    captioner_model: str = "qwen" # qwen/blip
    input_dir: str = "./images"
    out_dir: str = "./out"
    # {LABEL} - an object label from image
    system_prompt: str = "Опиши изображение части интерфейса отталкиваясь от текста из мануала: " # if captioner = blip, prompt needs to be in english
    device: str = "cuda"
    json_filename: str = "metadata.json"
    box_threshold: float = 0.5
    caption_max_length: int = 32