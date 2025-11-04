from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Конфигурация для VLM pipeline"""
    model: str = "yolov8"
    input_dir: str = "./images"
    out_dir: str = "./out"
    # {LABEL} - an object label from image
    prompt: str = "Describe this {LABEL} on a picture what is supposed to be a photo of product for sale"
    device: str = "cuda"
    json_filename: str = "metadata.json"
    box_threshold: float = 0.7
    caption_max_length: int = 32
    visualise: bool = True