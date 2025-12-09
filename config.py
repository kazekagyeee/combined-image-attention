from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Конфигурация для VLM pipeline"""
    detector_model: str = "uied_cv" # yolov8/uied_cv
    captioner_model: str = "glm" # qwen/blip/glm/paligemma
    input_dir: str = "./images"
    out_dir: str = "./out"
    # Prompt for two images needs to be like: "The first image is the full picture, the second is a cropped part of it. Describe what you see in the cropped part and its relation to the full image."
    system_prompt: str = ("Первое изображение - полное, второе - вырезанная часть полного. "
                          "Опиши то что ты видишь на втором изображении, отталкиваясь от первого, "
                          "учитывая текст из мануала, описывающего контекст первого изображения: ") # if captioner = blip, prompt needs to be in english
    device: str = "cuda"
    json_filename: str = "metadata.json"
    box_threshold: float = 0.5
    caption_max_length: int = 256