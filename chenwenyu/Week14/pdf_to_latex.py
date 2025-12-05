import os
import numpy as np
from pdf2image import convert_from_path
from paddleocr import PaddleOCR
from pathlib import Path

def pdf_to_images(pdf_path, dpi=200):
    return convert_from_path(pdf_path, dpi=dpi)

def find_formula_start_y(ocr_result, keyword="建模公式"):
    """
    找到包含关键字的行的 y_min 值，用于裁剪
    """
    for line in ocr_result:
        for box, text_info in line:
            text = text_info[0]
            if keyword in text:
                ys = [p[1] for p in box]
                return min(ys)
    return None


def crop_image_from_y(image, start_y, margin=20):
    """
    从 start_y 开始向下裁剪整张图片
    """
    w, h = image.size
    y1 = max(0, start_y - margin)
    return image.crop((0, y1, w, h))


def extract_formula_section(pdf_path, out_dir="output"):
    ocr = PaddleOCR(lang="ch", use_angle_cls=False, use_gpu=False)

    # 生成输出文件夹
    os.makedirs(out_dir, exist_ok=True)

    # 将 PDF 转为图片
    images = pdf_to_images(pdf_path)

    for page_idx, img in enumerate(images):
        img_np = np.array(img)

        # OCR 整页
        ocr_result = ocr.ocr(img_np)

        # 寻找 “建模公式” 行
        start_y = find_formula_start_y(ocr_result)

        if start_y is None:
            print(f"[跳过] 第 {page_idx+1} 页没有找到“建模公式”")
            continue

        # 从该行起裁剪到底
        cropped = crop_image_from_y(img, start_y)

        # 保存最终只有公式部分的单张图片
        out_path = os.path.join(out_dir, f"{Path(pdf_path).stem}_formula.png")
        cropped.save(out_path)
        print(f"已生成公式图片：{out_path}")
        return  # 每个 PDF 只生成一张图片

    print(f"[警告] PDF 中没有找到“建模公式” → 无法生成公式图片")
