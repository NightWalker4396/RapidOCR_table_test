# -*- encoding: utf-8 -*-
"""表格识别示例脚本

使用本地安装的 RapidOCR 对输入图片进行识别，并根据
指定的表头文字自动整理出表格，同时返回表格前后的
普通文本，便于快速验证表格解析功能。

示例::

    python table_demo.py path/to/img.png --headers 项目 数量 单价 金额
"""
import argparse
from typing import List

from PIL import Image

from rapidocr import RapidOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="表格识别示例")
    parser.add_argument("image", help="待识别的图片路径")
    parser.add_argument(
        "--headers", nargs="+", required=True, help="表头文字列表，需与图片中一致"
    )
    return parser.parse_args()


def main(image_path: str, headers: List[str]) -> None:
    """执行表格识别并打印结果"""
    img = Image.open(image_path)

    ocr = RapidOCR()
    output = ocr(img)

    doc = output.to_document(headers)

    print("表格前文字:\n", doc.get("before", ""))
    print("\n表格内容:")
    for row in doc.get("table", []):
        print(row)
    print("\n表格后文字:\n", doc.get("after", ""))


if __name__ == "__main__":
    args = parse_args()
    main(args.image, args.headers)
