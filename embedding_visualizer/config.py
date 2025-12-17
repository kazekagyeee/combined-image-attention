from dataclasses import dataclass


@dataclass
class BenchmarkConfig:
    """Конфигурация для бенчмарка"""
    detector_model: str = "uied_cv" # yolov8/uied_cv
    core_embeddings_path: str = "./embeddings/core_embeddings.json"
    embeddings_under_research: str = "./embeddings/uied-qwen-2.5-2-images.json"
    base_embeddings_path: str = "./embeddings/qwen2.5_embeddings.json"
    threshold_T: float = 0.024895