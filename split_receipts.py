#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""将单页包含多张票据的 PDF 自动裁剪为多份独立 PDF 的脚本。

检测逻辑：
1. 使用 PyMuPDF (fitz) 将每页渲染为图像
2. 计算行/列内容密度，找到大片空白区域
3. 根据空白区域拆分成多个票据边界框
4. 使用 PyPDF2 根据边界框裁剪页面并写出新 PDF

裁剪后的 PDF 保持与被裁剪区域一致的尺寸，不做缩放变形
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple

try:
    import fitz
except ImportError:
    print("错误: 需要安装 PyMuPDF 库")
    print("请运行: pip install PyMuPDF")
    exit(1)

from PyPDF2 import PdfReader, PdfWriter, Transformation
from PyPDF2.generic import RectangleObject


@dataclass
class ImageBox:
    """表示图像空间内的矩形区域，坐标以左上角为原点。"""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_tuple(self) -> Tuple[int, int, int, int]:
        return (self.left, self.top, self.right, self.bottom)


@dataclass
class PdfBox:
    left: float
    bottom: float
    right: float
    top: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.top - self.bottom


class ReceiptSplitter:
    def __init__(
        self,
        pdf_path: Path,
        output_dir: Path,
        dpi: int = 300,
        row_blank_threshold: float = 0.012,
        col_blank_threshold: float = 0.01,
        min_row_gap_ratio: float = 0.03,
        min_col_gap_ratio: float = 0.03,
        min_row_height_ratio: float = 0.18,
        min_col_width_ratio: float = 0.35,
        min_region_fill: float = 0.005,
    ) -> None:
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.dpi = dpi
        self.row_blank_threshold = row_blank_threshold
        self.col_blank_threshold = col_blank_threshold
        self.min_row_gap_ratio = min_row_gap_ratio
        self.min_col_gap_ratio = min_col_gap_ratio
        self.min_row_height_ratio = min_row_height_ratio
        self.min_col_width_ratio = min_col_width_ratio
        self.min_region_fill = min_region_fill

    def run(self) -> None:
        reader = PdfReader(self.pdf_path)
        total_pages = len(reader.pages)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        print(f"开始处理 {self.pdf_path}，共 {total_pages} 页")

        export_count = 0
        doc = fitz.open(str(self.pdf_path))

        for idx in range(total_pages):
            page_no = idx + 1
            page = reader.pages[idx]
            rotation = int(page.get("/Rotate") or 0) % 360
            base_width = float(page.mediabox.width)
            base_height = float(page.mediabox.height)
            if rotation in (90, 270):
                page_width, page_height = base_height, base_width
            else:
                page_width, page_height = base_width, base_height

            pixmap = self._render_page(doc, idx)
            boxes = self._detect_image_boxes(pixmap)

            print(f"第 {page_no:02d} 页检测到 {len(boxes)} 个候选区域")

            if len(boxes) <= 1:
                export_count += 1
                output_path = self.output_dir / f"page-{page_no:02d}-receipt-{export_count:03d}.pdf"
                self._save_full_page(reader, idx, output_path)
                continue

            img_width, img_height = pixmap.width, pixmap.height
            page_boxes = [
                self._image_box_to_pdf_box(
                    box,
                    rotation,
                    base_width,
                    base_height,
                    page_width,
                    page_height,
                    img_width,
                    img_height,
                )
                for box in boxes
            ]

            for part_index, pdf_box in enumerate(page_boxes, start=1):
                export_count += 1
                output_path = self.output_dir / (
                    f"page-{page_no:02d}-part-{part_index:02d}-receipt-{export_count:03d}.pdf"
                )
                print(
                    "  -> 导出分块",
                    f"{part_index}/{len(page_boxes)}",
                    f"尺寸 {pdf_box.width:.2f} x {pdf_box.height:.2f}",
                    f"保存到 {output_path.name}",
                )
                self._save_cropped_page(reader, idx, pdf_box, output_path)

        doc.close()
        print(f"完成！输出目录：{self.output_dir}")

    def _render_page(self, doc: fitz.Document, page_index: int) -> fitz.Pixmap:
        page = doc[page_index]
        zoom = self.dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return pix

    def _detect_image_boxes(self, pixmap: fitz.Pixmap) -> List[ImageBox]:
        width, height = pixmap.width, pixmap.height
        samples = pixmap.samples
        stride = pixmap.stride
        
        gray_data = []
        for y in range(height):
            row = []
            for x in range(width):
                offset = y * stride + x * 3
                r, g, b = samples[offset], samples[offset + 1], samples[offset + 2]
                gray_val = int(0.299 * r + 0.587 * g + 0.114 * b)
                row.append(gray_val)
            gray_data.append(row)
        
        threshold = self._otsu_threshold(gray_data, width, height)
        
        content_mask = [[gray_data[y][x] < threshold for x in range(width)] for y in range(height)]

        row_segments = self._split_axis(
            mask=content_mask,
            axis=0,
            blank_threshold=self.row_blank_threshold,
            min_gap_ratio=self.min_row_gap_ratio,
            min_segment_ratio=self.min_row_height_ratio,
        )

        if not row_segments:
            row_segments = [(0, height)]

        boxes: List[ImageBox] = []
        for row_start, row_end in row_segments:
            row_slice = [content_mask[y] for y in range(row_start, row_end)]
            col_segments = self._split_axis(
                mask=row_slice,
                axis=1,
                blank_threshold=self.col_blank_threshold,
                min_gap_ratio=self.min_col_gap_ratio,
                min_segment_ratio=self.min_col_width_ratio,
            )

            if not col_segments:
                col_segments = [(0, width)]

            for col_start, col_end in col_segments:
                total_pixels = 0
                content_pixels = 0
                for y in range(len(row_slice)):
                    for x in range(col_start, col_end):
                        total_pixels += 1
                        if row_slice[y][x]:
                            content_pixels += 1
                
                fill_ratio = content_pixels / total_pixels if total_pixels > 0 else 0
                if fill_ratio < self.min_region_fill:
                    continue
                boxes.append(ImageBox(col_start, row_start, col_end, row_end))

        if not boxes:
            boxes.append(ImageBox(0, 0, width, height))

        boxes = self._merge_overlapping_boxes(boxes)
        boxes.sort(key=lambda b: (b.top, b.left))
        return boxes

    def _split_axis(
        self,
        mask: List[List[bool]],
        axis: int,
        blank_threshold: float,
        min_gap_ratio: float,
        min_segment_ratio: float,
    ) -> List[Tuple[int, int]]:
        if axis == 0:
            length = len(mask)
            densities = [sum(row) / len(row) if len(row) > 0 else 0 for row in mask]
        else:
            length = len(mask[0]) if mask else 0
            densities = []
            for x in range(length):
                col_sum = sum(mask[y][x] for y in range(len(mask)))
                densities.append(col_sum / len(mask) if len(mask) > 0 else 0)
        
        min_gap = max(30, int(length * min_gap_ratio))
        min_segment = max(60, int(length * min_segment_ratio))

        blank_flags = [d < blank_threshold for d in densities]

        segments: List[Tuple[int, int]] = []
        start = 0
        idx = 0
        while idx < length:
            if blank_flags[idx]:
                gap_start = idx
                while idx < length and blank_flags[idx]:
                    idx += 1
                gap_len = idx - gap_start
                if gap_len >= min_gap and gap_start - start >= min_segment:
                    segments.append((start, gap_start))
                    start = idx
            else:
                idx += 1

        if length - start >= min_segment:
            segments.append((start, length))

        return segments

    def _merge_overlapping_boxes(self, boxes: Sequence[ImageBox]) -> List[ImageBox]:
        if not boxes:
            return []

        merged: List[ImageBox] = []
        for box in boxes:
            placed = False
            for i, existing in enumerate(merged):
                if self._boxes_overlap(existing, box):
                    merged[i] = self._merge_two_boxes(existing, box)
                    placed = True
                    break
            if not placed:
                merged.append(box)
        return merged

    @staticmethod
    def _boxes_overlap(a: ImageBox, b: ImageBox) -> bool:
        horizontal_overlap = not (a.right < b.left or b.right < a.left)
        vertical_overlap = not (a.bottom < b.top or b.bottom < a.top)
        return horizontal_overlap and vertical_overlap

    @staticmethod
    def _merge_two_boxes(a: ImageBox, b: ImageBox) -> ImageBox:
        return ImageBox(
            left=min(a.left, b.left),
            top=min(a.top, b.top),
            right=max(a.right, b.right),
            bottom=max(a.bottom, b.bottom),
        )

    @staticmethod
    def _otsu_threshold(gray_data: List[List[int]], width: int, height: int) -> int:
        hist = [0] * 256
        for row in gray_data:
            for val in row:
                hist[val] += 1
        
        total = width * height
        sum_total = sum(i * hist[i] for i in range(256))

        sum_b = 0.0
        weight_b = 0.0
        max_var = -1.0
        threshold = 127

        for i in range(256):
            weight_b += hist[i]
            if weight_b == 0:
                continue
            weight_f = total - weight_b
            if weight_f == 0:
                break
            sum_b += i * hist[i]
            mean_b = sum_b / weight_b
            mean_f = (sum_total - sum_b) / weight_f
            var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
            if var_between > max_var:
                max_var = var_between
                threshold = i

        threshold = max(30, min(240, int(threshold * 0.9)))
        return threshold

    def _image_box_to_pdf_box(
        self,
        box: ImageBox,
        rotation: int,
        base_width: float,
        base_height: float,
        page_width: float,
        page_height: float,
        img_width: int,
        img_height: int,
    ) -> PdfBox:
        rot_left = (box.left / img_width) * page_width
        rot_right = (box.right / img_width) * page_width
        rot_top = page_height - (box.top / img_height) * page_height
        rot_bottom = page_height - (box.bottom / img_height) * page_height

        corners_rot = [
            (rot_left, rot_bottom),
            (rot_left, rot_top),
            (rot_right, rot_bottom),
            (rot_right, rot_top),
        ]
        corners_pdf = [
            self._map_rotated_point_to_pdf(x, y, rotation, base_width, base_height)
            for x, y in corners_rot
        ]
        xs = [pt[0] for pt in corners_pdf]
        ys = [pt[1] for pt in corners_pdf]
        left, right = min(xs), max(xs)
        bottom, top = min(ys), max(ys)
        return PdfBox(left, bottom, right, top)

    @staticmethod
    def _map_rotated_point_to_pdf(
        x_rot: float,
        y_rot: float,
        rotation: int,
        base_width: float,
        base_height: float,
    ) -> Tuple[float, float]:
        if rotation == 0:
            return x_rot, y_rot
        if rotation == 90:
            return y_rot, base_height - x_rot
        if rotation == 180:
            return base_width - x_rot, base_height - y_rot
        if rotation == 270:
            return base_width - y_rot, x_rot
        raise ValueError(f"不支持的页面旋转角度: {rotation}")

    def _save_full_page(self, reader: PdfReader, page_index: int, output_path: Path) -> None:
        writer = PdfWriter()
        writer.add_page(copy.deepcopy(reader.pages[page_index]))
        with output_path.open("wb") as fh:
            writer.write(fh)

    def _save_cropped_page(
        self,
        reader: PdfReader,
        page_index: int,
        pdf_box: PdfBox,
        output_path: Path,
    ) -> None:
        page = copy.deepcopy(reader.pages[page_index])
        transformation = Transformation().translate(-pdf_box.left, -pdf_box.bottom)
        page.add_transformation(transformation)

        box_rect = RectangleObject([0, 0, pdf_box.width, pdf_box.height])
        page.mediabox = box_rect
        page.cropbox = box_rect
        page.trimbox = box_rect
        page.artbox = box_rect

        writer = PdfWriter()
        writer.add_page(page)
        with output_path.open("wb") as fh:
            writer.write(fh)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="自动拆分 PDF 中的多张票据")
    parser.add_argument("pdf", type=Path, help="需要处理的 PDF 文件路径")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="输出目录（默认为 ./output）",
    )
    parser.add_argument("--dpi", type=int, default=300, help="渲染 PDF 时使用的 DPI")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.pdf.exists():
        raise FileNotFoundError(f"未找到 PDF 文件：{args.pdf}")

    splitter = ReceiptSplitter(pdf_path=args.pdf, output_dir=args.output, dpi=args.dpi)
    splitter.run()


if __name__ == "__main__":
    main()
