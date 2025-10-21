from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Конфигурация для VLM pipeline"""
    model: str = "yolov8"
    input_dir: str = "./images"
    out_dir: str = "./out"
    prompt: str = "a clothes, a human, a text"
    device: str = "gpu"
    json_filename: str = "metadata.json"
    box_threshold: float = 0.7
    caption_max_length: int = 32
    visualise: bool = True