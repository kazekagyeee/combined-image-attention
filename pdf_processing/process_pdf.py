import os
import fitz  # PyMuPDF


def extract_images_and_text(pdf_path, output_dir="../images", skip_first=0, skip_last=0):
    """
    Извлекает изображения и первый абзац текста перед каждым изображением из PDF.

    Args:
        pdf_path (str): Путь к PDF-файлу
        output_dir (str): Директория для сохранения результатов
        skip_first (int): Количество пропускаемых первых страниц
        skip_last (int): Количество пропускаемых последних страниц
    """
    os.makedirs(output_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    start_page = skip_first
    end_page = total_pages - skip_last

    for page_num in range(start_page, end_page):
        page = doc[page_num]
        text = page.get_text()
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        image_list = page.get_images()

        for img_index, img in enumerate(image_list):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)

            if pix.n - pix.alpha < 4:
                img_filename = f"image_{page_num + 1}_{img_index + 1}.png"
            else:
                img_filename = f"image_{page_num + 1}_{img_index + 1}.png"
                pix = fitz.Pixmap(fitz.csRGB, pix)

            img_path = os.path.join(output_dir, img_filename)
            pix.save(img_path)
            pix = None

            # Поиск первого абзаца перед изображением
            first_paragraph = ""
            if paragraphs:
                first_paragraph = paragraphs[0]

            # Сохранение текста
            txt_filename = os.path.splitext(img_filename)[0] + ".txt"
            txt_path = os.path.join(output_dir, txt_filename)

            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(first_paragraph)

    doc.close()

extract_images_and_text("file.pdf", skip_first=2, skip_last=2)
