# -*- encoding: utf-8 -*-
from typing import Dict, List, Tuple, Optional

import numpy as np

from .to_markdown import ToMarkdown


class ToTable:
    """将 OCR 识别出的文本和坐标整理成结构化表格"""

    @staticmethod
    def _box_props(box: np.ndarray) -> Dict[str, float]:
        """计算单个框的几何属性，便于后续排序和归类"""

        ys = box[:, 1]
        xs = box[:, 0]
        top = np.min(ys)
        bottom = np.max(ys)
        left = np.min(xs)
        right = np.max(xs)
        return {
            "top": float(top),
            "bottom": float(bottom),
            "left": float(left),
            "right": float(right),
            "height": float(bottom - top),  # 高度
            "center_x": float((left + right) / 2),  # 中心点 x 坐标
            "center_y": float((top + bottom) / 2),  # 中心点 y 坐标
        }

    @classmethod
    def to(
        cls, boxes: Optional[np.ndarray], txts: Optional[Tuple[str]], headers: List[str]
    ) -> List[Dict[str, str]]:
        if boxes is None or txts is None or not headers:
            return []

        # 1. 根据表头文字找到对应的检测框
        header_boxes: List[np.ndarray] = []
        header_indices: List[int] = []
        for h in headers:
            found = False
            for i, t in enumerate(txts):
                if t == h:
                    header_boxes.append(boxes[i])
                    header_indices.append(i)
                    found = True
                    break
            if not found:
                # 只要有一个表头没找到就返回空表
                return []

        # 计算表头的几何信息，便于后续定位正文
        header_props = [cls._box_props(b) for b in header_boxes]
        header_bottom = max(p["bottom"] for p in header_props)
        header_height = np.median([p["height"] for p in header_props])
        centers_x = [p["center_x"] for p in header_props]

        # 2. 根据表头中心点计算各列的左右边界
        col_bounds: List[Tuple[float, float]] = []
        for i, cx in enumerate(centers_x):
            left = (centers_x[i - 1] + cx) / 2 if i > 0 else -np.inf
            right = (cx + centers_x[i + 1]) / 2 if i < len(centers_x) - 1 else np.inf
            col_bounds.append((left, right))

        # 3. 收集正文中的文本框
        body: List[Tuple[Dict[str, float], str]] = []
        for i, (box, txt) in enumerate(zip(boxes, txts)):
            if i in header_indices:
                continue
            prop = cls._box_props(box)
            if prop["top"] <= header_bottom:  # 位于表头之上或重叠，跳过
                continue
            body.append((prop, txt))

        if not body:
            return []

        # 4. 按 y 中心坐标排序，并聚合成行
        body.sort(key=lambda item: item[0]["center_y"])

        rows: List[List[Tuple[Dict[str, float], str]]] = []
        current_row: List[Tuple[Dict[str, float], str]] = []
        row_thresh = header_height * 1.2  # 行间距阈值
        for prop, txt in body:
            if not current_row:
                current_row.append((prop, txt))
                continue
            prev_center = current_row[-1][0]["center_y"]
            if abs(prop["center_y"] - prev_center) <= row_thresh:
                current_row.append((prop, txt))
            else:
                rows.append(current_row)
                current_row = [(prop, txt)]
        if current_row:
            rows.append(current_row)

        # 5. 将同一行的文本归入对应列并拼接多行内容
        table: List[Dict[str, str]] = []
        for row in rows:
            row_dict: Dict[str, str] = {h: "" for h in headers}
            for prop, txt in row:
                cx = prop["center_x"]
                for idx, (left, right) in enumerate(col_bounds):
                    if left < cx <= right:
                        key = headers[idx]
                        row_dict[key] += txt
                        break
            if row_dict[headers[0]]:  # 过滤掉空行
                table.append(row_dict)
        return table

    @classmethod
    def with_text(
        cls, boxes: Optional[np.ndarray], txts: Optional[Tuple[str]], headers: List[str]
    ) -> Dict[str, object]:
        if boxes is None or txts is None or not headers:
            return {"before": "", "table": [], "after": ""}

        # 1. 定位表头
        header_boxes: List[np.ndarray] = []
        header_indices: List[int] = []
        for h in headers:
            for i, t in enumerate(txts):
                if t == h:
                    header_boxes.append(boxes[i])
                    header_indices.append(i)
                    break
            else:
                return {"before": "", "table": [], "after": ""}

        header_props = [cls._box_props(b) for b in header_boxes]
        header_top = min(p["top"] for p in header_props)
        header_bottom = max(p["bottom"] for p in header_props)
        header_height = np.median([p["height"] for p in header_props])
        centers_x = [p["center_x"] for p in header_props]

        # 2. 计算列边界
        col_bounds: List[Tuple[float, float]] = []
        for i, cx in enumerate(centers_x):
            left = (centers_x[i - 1] + cx) / 2 if i > 0 else -np.inf
            right = (cx + centers_x[i + 1]) / 2 if i < len(centers_x) - 1 else np.inf
            col_bounds.append((left, right))

        # 3. 收集表体并记录索引，方便分割前后文本
        body: List[Tuple[Dict[str, float], str, int]] = []
        table_bottom = header_bottom
        for i, (box, txt) in enumerate(zip(boxes, txts)):
            if i in header_indices:
                continue
            prop = cls._box_props(box)
            if prop["top"] <= header_bottom:
                continue
            body.append((prop, txt, i))
            table_bottom = max(table_bottom, prop["bottom"])

        # 4. 按行聚合
        body.sort(key=lambda item: item[0]["center_y"])
        rows: List[List[Tuple[Dict[str, float], str, int]]] = []
        current_row: List[Tuple[Dict[str, float], str, int]] = []
        row_thresh = header_height * 1.2
        for prop, txt, idx in body:
            if not current_row:
                current_row.append((prop, txt, idx))
                continue
            prev_center = current_row[-1][0]["center_y"]
            if abs(prop["center_y"] - prev_center) <= row_thresh:
                current_row.append((prop, txt, idx))
            else:
                rows.append(current_row)
                current_row = [(prop, txt, idx)]
        if current_row:
            rows.append(current_row)

        # 5. 归并列并记录表格外的文本
        table: List[Dict[str, str]] = []
        used_indices = set(header_indices)
        after_boxes_row: List[np.ndarray] = []
        after_txts_row: List[str] = []
        for row in rows:
            row_dict: Dict[str, str] = {h: "" for h in headers}
            row_indices: List[int] = []
            row_boxes: List[np.ndarray] = []
            row_txts: List[str] = []
            row_bottom = header_bottom
            for prop, txt, idx in row:
                cx = prop["center_x"]
                for col_idx, (left, right) in enumerate(col_bounds):
                    if left < cx <= right:
                        key = headers[col_idx]
                        row_dict[key] += txt
                        break
                row_indices.append(idx)
                row_boxes.append(boxes[idx])
                row_txts.append(txt)
                row_bottom = max(row_bottom, prop["bottom"])
            if row_dict[headers[0]]:
                table.append(row_dict)
                used_indices.update(row_indices)
                table_bottom = max(table_bottom, row_bottom)
            else:
                after_boxes_row.extend(row_boxes)
                after_txts_row.extend(row_txts)
                used_indices.update(row_indices)

        # 6. 划分表格前后的文本并转为 Markdown 输出
        before_boxes: List[np.ndarray] = []
        before_txts: List[str] = []
        for i, (box, txt) in enumerate(zip(boxes, txts)):
            if i in used_indices:
                continue
            prop = cls._box_props(box)
            if prop["bottom"] <= header_top:
                before_boxes.append(box)
                before_txts.append(txt)

        before = (
            ToMarkdown.to(np.array(before_boxes), tuple(before_txts))
            if before_boxes
            else ""
        )
        after = (
            ToMarkdown.to(np.array(after_boxes_row), tuple(after_txts_row))
            if after_boxes_row
            else ""
        )

        return {"before": before, "table": table, "after": after}
