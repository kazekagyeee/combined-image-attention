import argparse
import torch
from config import PipelineConfig
from pipeline import VLMPipeline


def parse_args():
    """Парсит аргументы командной строки"""
    default_cfg = PipelineConfig()
    parser = argparse.ArgumentParser(description="VLM semantic image analyzer pipeline")
    parser.add_argument("--detector_model", choices=["owlvit", "groundingdino"], default=default_cfg.detector_model, help="detector backend")
    parser.add_argument("--input_dir", default=default_cfg.input_dir, help="folder with images")
    parser.add_argument("--out_dir", default=default_cfg.out_dir, help="output folder for crops and metadata")
    parser.add_argument("--system_prompt", default=default_cfg.system_prompt, help="comma-separated object queries")
    parser.add_argument("--device", default=default_cfg.device if torch.cuda.is_available() else "cpu", help="cpu or cuda")
    parser.add_argument("--json", default=default_cfg.json_filename, help="json filename for metadata")
    return parser.parse_args()


def main():
    """Главная функция для запуска pipeline"""
    args = parse_args()

    # Создаем конфигурацию из аргументов командной строки
    config = PipelineConfig(
        detector_model=args.detector_model,
        input_dir=args.input_dir,
        out_dir=args.out_dir,
        system_prompt=args.system_prompt,
        device=args.device,
        json_filename=args.json,
    )

    # Создаем и запускаем pipeline
    pipeline = VLMPipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()