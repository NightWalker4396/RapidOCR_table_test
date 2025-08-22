# -*- encoding: utf-8 -*-
import types
import importlib.util
from pathlib import Path

import numpy as np

# 手动构建 rapidocr.utils 包的路径，方便在测试中直接导入
utils_pkg = types.ModuleType("rapidocr.utils")
utils_pkg.__path__ = [str(Path(__file__).resolve().parents[1] / "rapidocr" / "utils")]
import sys
sys.modules.setdefault("rapidocr.utils", utils_pkg)

# 动态加载 output.py 模块，以获取 RapidOCROutput 类
spec = importlib.util.spec_from_file_location(
    "rapidocr.utils.output", Path(__file__).resolve().parents[1] / "rapidocr" / "utils" / "output.py"
)
output_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(output_module)
RapidOCROutput = output_module.RapidOCROutput


def _create_sample_output():
    boxes = np.array([
        # 表格前的普通文本
        [[0, -30], [40, -30], [40, -20], [0, -20]],
        [[0, -15], [80, -15], [80, -5], [0, -5]],
        # 表头: 项目 数量 单价 金额
        [[0, 0], [10, 0], [10, 10], [0, 10]],
        [[20, 0], [30, 0], [30, 10], [20, 10]],
        [[40, 0], [50, 0], [50, 10], [40, 10]],
        [[60, 0], [70, 0], [70, 10], [60, 10]],
        # 第一行
        [[0, 15], [10, 15], [10, 25], [0, 25]],
        [[20, 15], [30, 15], [30, 25], [20, 25]],
        [[40, 15], [50, 15], [50, 25], [40, 25]],
        [[60, 15], [70, 15], [70, 25], [60, 25]],
        # 第二行，首列包含两段文字模拟换行
        [[0, 30], [10, 30], [10, 40], [0, 40]],
        [[0, 40], [10, 40], [10, 50], [0, 50]],
        [[20, 35], [30, 35], [30, 45], [20, 45]],
        [[40, 35], [50, 35], [50, 45], [40, 45]],
        [[60, 35], [70, 35], [70, 45], [60, 45]],
        # 表格后的普通文本
        [[0, 55], [60, 55], [60, 65], [0, 65]],
    ])
    txts = (
        "序号:1",
        "日期:2025-01-01",
        "项目",
        "数量",
        "单价",
        "金额",
        "A",
        "1",
        "100",
        "100",
        "多",
        "行",
        "2",
        "50",
        "100",
        "总计:200",
    )
    return RapidOCROutput(boxes=boxes, txts=txts)


def test_to_table():
    result = _create_sample_output()
    table = result.to_table(["项目", "数量", "单价", "金额"])
    assert table == [
        {"项目": "A", "数量": "1", "单价": "100", "金额": "100"},
        {"项目": "多行", "数量": "2", "单价": "50", "金额": "100"},
    ]


def test_to_document():
    result = _create_sample_output()
    doc = result.to_document(["项目", "数量", "单价", "金额"])
    assert doc["table"] == [
        {"项目": "A", "数量": "1", "单价": "100", "金额": "100"},
        {"项目": "多行", "数量": "2", "单价": "50", "金额": "100"},
    ]
    assert doc["before"] == "序号:1\n日期:2025-01-01"
    assert doc["after"] == "总计:200"
