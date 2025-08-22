# -*- encoding: utf-8 -*-
# @Author: SWHL
# @Contact: liekkaskono@163.com
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .logger import Logger
from .to_json import ToJSON
from .to_markdown import ToMarkdown
from .to_table import ToTable
from .utils import save_img

try:  # 可选依赖：可视化需要 OpenCV 等图形库
    from .vis_res import VisRes
except Exception:  # pragma: no cover - 当缺少 cv2 或 libGL 时使用降级策略
    VisRes = None  # type: ignore

logger = Logger(logger_name=__name__).get_log()


@dataclass
class RapidOCROutput:
    img: Optional[np.ndarray] = None
    boxes: Optional[np.ndarray] = None
    txts: Optional[Tuple[str]] = None
    scores: Optional[Tuple[float]] = None
    word_results: Tuple[Tuple[str, float, Optional[List[List[int]]]]] = (
        ("", 1.0, None),
    )
    elapse_list: List[Union[float, None]] = field(default_factory=list)
    elapse: float = field(init=False)
    viser: Optional[VisRes] = None

    def __post_init__(self):
        self.elapse = sum(v for v in self.elapse_list if isinstance(v, float))

    def __len__(self):
        if self.txts is None:
            return 0
        return len(self.txts)

    def to_json(self) -> Optional[List[Dict[Any, Any]]]:
        """转为 JSON 结构"""
        if any(v is None for v in (self.boxes, self.txts, self.scores)):
            logger.warning("The identified content is empty.")
            return None
        return ToJSON.to(self.boxes, self.txts, self.scores)

    def to_markdown(self) -> str:
        """转为 Markdown 形式，便于展示"""
        return ToMarkdown.to(self.boxes, self.txts)

    def to_table(self, headers: List[str]) -> Optional[List[Dict[str, str]]]:
        """根据给定表头将识别结果整理成表格"""
        if any(v is None for v in (self.boxes, self.txts)):
            logger.warning("The identified content is empty.")
            return None
        return ToTable.to(self.boxes, self.txts, headers)

    def to_document(self, headers: List[str]) -> Optional[Dict[str, Any]]:
        """返回表格前后的文本以及表格本身，实现文档级提取"""
        if any(v is None for v in (self.boxes, self.txts)):
            logger.warning("The identified content is empty.")
            return None
        return ToTable.with_text(self.boxes, self.txts, headers)

    def vis(self, save_path: Optional[str] = None) -> Optional[np.ndarray]:
        if self.img is None or self.boxes is None:
            logger.warning("No image or boxes to visualize.")
            return None

        if self.viser is None:
            logger.error("vis instance is None")
            return None

        if all(v is None for v in self.word_results):
            vis_img = self.viser(self.img, self.boxes, self.txts, self.scores)

            if save_path is not None:
                save_img(save_path, vis_img)
                logger.info("Visualization saved as %s", save_path)
            return vis_img

        # single word vis
        words_results = sum(self.word_results, ())
        words, words_scores, words_boxes = list(zip(*words_results))
        vis_img = self.viser(self.img, words_boxes, words, words_scores)

        if save_path is not None:
            save_img(save_path, vis_img)
            logger.info("Single word visualization saved as %s", save_path)
        return vis_img
