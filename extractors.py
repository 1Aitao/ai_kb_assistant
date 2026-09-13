"""文档内容提取器：按扩展名分发到对应提取函数，新增格式只需在 EXTRACTORS 加一行。"""
import cv2
import fitz
import numpy as np
from pypdf import PdfReader


def extract_txt_md(file_path: str) -> str:
    """提取 txt/md 文本文件内容。"""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def ocr_pdf(file_path: str) -> str:
    """OCR 识别扫描版 PDF：每页渲染成图像后在内存中直接识别，不写临时文件。"""
    from rapidocr_onnxruntime import RapidOCR
    ocr = RapidOCR()
    doc = fitz.open(file_path)
    full_text = ""
    for page in doc:
        # 放大 2 倍渲染，提高小字识别率
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
        # 直接在内存中传递图像字节，避免 Windows 上临时文件被占用导致删除失败
        img = cv2.imdecode(np.frombuffer(pix.tobytes("png"), np.uint8), cv2.IMREAD_COLOR)
        result, _ = ocr(img)
        if result:
            full_text += "\n".join(line[1] for line in result) + "\n"
    return full_text.strip()


def extract_pdf(file_path: str) -> str:
    """提取 PDF 文字：优先读文字层，为空时降级 OCR。"""
    reader = PdfReader(file_path)
    text = "".join((p.extract_text() or "") for p in reader.pages)
    if text.strip():
        return text.strip()
    return ocr_pdf(file_path)


def extract_image(file_path: str) -> str:
    """提取图片（png/jpg）中的文字。"""
    from rapidocr_onnxruntime import RapidOCR
    result, _ = RapidOCR()(file_path)
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()


def extract_docx(file_path: str) -> str:
    """提取 Word 文档段落文字（类名重命名避免与业务模型冲突）。"""
    from docx import Document as DocxDocument
    d = DocxDocument(file_path)
    return "\n".join(p.text for p in d.paragraphs if p.text.strip())


def extract_xlsx(file_path: str) -> str:
    """提取 Excel：read_only+data_only 取单元格值，每行用 | 拼接。"""
    from openpyxl import load_workbook
    wb = load_workbook(file_path, read_only=True, data_only=True)
    lines = []
    for ws in wb.worksheets:
        for row in ws.iter_rows(values_only=True):
            cells = [str(c) for c in row if c is not None]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines)


# 扩展名 -> 提取函数 的路由表
EXTRACTORS = {
    ".txt": extract_txt_md, ".md": extract_txt_md,
    ".pdf": extract_pdf,
    ".png": extract_image, ".jpg": extract_image, ".jpeg": extract_image,
    ".docx": extract_docx,
    ".xlsx": extract_xlsx,
}
