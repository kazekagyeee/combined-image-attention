"""
vlm_detector_pipeline.py

Функции:
1) детектит с помощью VLM (OWL-ViT или Grounding DINO, опция в config)
2) сохраняет выделенные bbox как подкартинки
3) делает текстовые описания к подкартинкам через выбранную VLM (captioning)
4) получает эмбеддинги подкартинок ПО ТЕКСТОВОМУ ОПИСАНИЮ и сохраняет метаданные в JSON

Запуск:
python vlm_pipeline.py --model owlvit --input_dir ./images --out_dir ./out --prompt "a dog, a cat"
"""

from cli import main

if __name__ == "__main__":
    main()