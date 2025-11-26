#!/usr/bin/env python3
# 指示操作系统使用 Python 3 解释器来执行当前脚本
# -*- coding: utf-8 -*-
# 声明源代码文件采用 UTF-8 编码，确保中文字符正常显示
"""将单页包含多张票据的 PDF 自动裁剪为多份独立 PDF 的脚本。

检测逻辑：
1. 使用 PyMuPDF (fitz) 将每页渲染为图像
2. 计算行/列内容密度，找到大片空白区域
3. 根据空白区域拆分成多个票据边界框
4. 使用 PyPDF2 根据边界框裁剪页面并写出新 PDF

裁剪后的 PDF 保持与被裁剪区域一致的尺寸，不做缩放变形
"""

from __future__ import annotations  # 允许在类型注解中引用尚未定义的类

import argparse  # 解析命令行参数
import copy  # 提供深拷贝功能，避免直接修改原页面对象
from dataclasses import dataclass  # 帮助定义数据类以保存矩形信息
from pathlib import Path  # 处理文件系统路径
from typing import List, Sequence, Tuple  # 提供类型注解所需的泛型容器

try:
    # 尝试导入 PyMuPDF 库（别名 fitz），用于渲染 PDF 为图像
    import fitz
except ImportError:
    # 如果未安装 PyMuPDF 库，打印错误信息并退出
    print("错误: 需要安装 PyMuPDF 库")
    print("请运行: pip install PyMuPDF")
    exit(1)

# 导入 PyPDF2 相关模块，用于读取、写入和变换 PDF 页面
from PyPDF2 import PdfReader, PdfWriter, Transformation
# 导入 PDF 对象类型，用于操作 PDF 内部结构
from PyPDF2.generic import NameObject, NumberObject, RectangleObject


# 使用数据类自动生成初始化和比较方法，方便存放图像坐标
@dataclass
class ImageBox:
    """表示图像空间内的矩形区域，坐标以左上角为原点。"""

    # 矩形的左边界 X 坐标（像素）
    left: int
    # 矩形的上边界 Y 坐标（像素）
    top: int
    # 矩形的右边界 X 坐标（像素）
    right: int
    # 矩形的下边界 Y 坐标（像素）
    bottom: int

    @property
    def width(self) -> int:
        """计算并返回矩形的宽度（像素）"""
        return self.right - self.left

    @property
    def height(self) -> int:
        """计算并返回矩形的高度（像素）"""
        return self.bottom - self.top

    def to_tuple(self) -> Tuple[int, int, int, int]:
        """将矩形边界转换为元组格式 (left, top, right, bottom)"""
        return (self.left, self.top, self.right, self.bottom)


# 定义 PDF 坐标空间内的矩形盒子，方便记录裁剪区域
@dataclass
class PdfBox:
    # 裁剪区域的左边界（单位：pt）
    left: float
    # 裁剪区域的下边界（单位：pt）
    bottom: float
    # 裁剪区域的右边界（单位：pt）
    right: float
    # 裁剪区域的上边界（单位：pt）
    top: float

    @property
    def width(self) -> float:
        """返回盒子的宽度（单位：pt）"""
        return self.right - self.left

    @property
    def height(self) -> float:
        """返回盒子的高度（单位：pt）"""
        return self.top - self.bottom


class ReceiptSplitter:
    """票据拆分处理类，负责检测并裁剪 PDF 中的多个票据区域。"""
    
    def __init__(
        self,
        pdf_path: Path,  # 要处理的 PDF 文件路径
        output_dir: Path,  # 输出文件夹路径
        dpi: int = 300,  # 渲染 PDF 为图像时的分辨率（点/英寸）
        row_blank_threshold: float = 0.012,  # 判断某一行是否为空白行的密度阈值
        col_blank_threshold: float = 0.01,  # 判断某一列是否为空白列的密度阈值
        min_row_gap_ratio: float = 0.03,  # 行方向上空白区域最小占页面高度的比例
        min_col_gap_ratio: float = 0.03,  # 列方向上空白区域最小占页面宽度的比例
        min_row_height_ratio: float = 0.18,  # 票据区域在行方向最小高度占页面高度的比例
        min_col_width_ratio: float = 0.35,  # 票据区域在列方向最小宽度占页面宽度的比例
        min_region_fill: float = 0.005,  # 一个区域被判定为有效票据的最小内容填充率
        padding: float = 12.0,  # 裁剪时向外扩展的边距（单位：pt）
    ) -> None:
        # 保存输入的 PDF 文件路径
        self.pdf_path = pdf_path
        # 保存输出目录路径
        self.output_dir = output_dir
        # 保存渲染 DPI 设置
        self.dpi = dpi
        # 保存行空白判定阈值
        self.row_blank_threshold = row_blank_threshold
        # 保存列空白判定阈值
        self.col_blank_threshold = col_blank_threshold
        # 保存行方向空白间隙最小比例
        self.min_row_gap_ratio = min_row_gap_ratio
        # 保存列方向空白间隙最小比例
        self.min_col_gap_ratio = min_col_gap_ratio
        # 保存票据在行方向的最小高度比例
        self.min_row_height_ratio = min_row_height_ratio
        # 保存票据在列方向的最小宽度比例
        self.min_col_width_ratio = min_col_width_ratio
        # 保存区域有效性的最小填充率
        self.min_region_fill = min_region_fill
        # 保存裁剪边距参数
        self.padding = padding

    def run(self) -> None:
        """主执行函数，负责读取 PDF、检测区域、裁剪并写出结果。"""
        # 使用 PyPDF2 的 PdfReader 打开输入的 PDF 文件
        reader = PdfReader(self.pdf_path)
        # 获取 PDF 的总页数
        total_pages = len(reader.pages)
        # 创建输出目录（如果不存在），parents=True 会递归创建父目录，exist_ok=True 表示已存在不报错
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 打印处理开始信息
        print(f"开始处理 {self.pdf_path}，共 {total_pages} 页")

        # 初始化票据导出计数器，用于跟踪已导出的票据数量
        export_count = 0
        # 使用 PyMuPDF 打开 PDF 文档，用于渲染图像
        doc = fitz.open(str(self.pdf_path))
        # 创建 PyPDF2 的 PdfWriter 对象，用于合并所有裁剪后的页面
        merged_writer = PdfWriter()

        # 遍历 PDF 的每一页
        for idx in range(total_pages):
            # 计算显示用的页码（从 1 开始）
            page_no = idx + 1
            # 获取当前页面对象
            page = reader.pages[idx]
            # 读取页面的旋转角度，如果没有则默认为 0，并取模 360
            rotation = int(page.get("/Rotate") or 0) % 360
            # 获取页面媒体框的原始宽度和高度（未应用旋转）
            base_width = float(page.mediabox.width)
            base_height = float(page.mediabox.height)
            # 如果页面旋转了 90 或 270 度，宽高需要互换
            if rotation in (90, 270):
                page_width, page_height = base_height, base_width
            else:
                # 否则保持原样
                page_width, page_height = base_width, base_height

            # 将当前页面渲染为图像（Pixmap 对象）
            pixmap = self._render_page(doc, idx)
            # 在渲染的图像中检测票据边界框
            boxes = self._detect_image_boxes(pixmap)

            # 打印检测到的候选区域数量
            print(f"第 {page_no:02d} 页检测到 {len(boxes)} 个候选区域")

            # 如果检测到的区域少于等于 1 个，说明整页是一个票据或无需拆分
            if len(boxes) <= 1:
                # 票据计数加 1
                export_count += 1
                # 打印信息
                print(f"  -> 添加完整页面到合并文件（票据 #{export_count}）")
                # 直接将完整页面添加到输出
                self._add_full_page_to_writer(reader, idx, merged_writer)
                # 跳过后续的裁剪处理，继续下一页
                continue

            # 获取渲染图像的宽度和高度（像素）
            img_width, img_height = pixmap.width, pixmap.height
            # 将图像空间的边界框转换为 PDF 坐标空间，并扩展边距
            page_boxes = [
                self._expand_pdf_box(  # 扩展边距
                    self._image_box_to_pdf_box(  # 坐标转换
                        box,  # 当前检测到的图像边界框
                        rotation,  # 页面整体旋转角度
                        base_width,  # 页面原始宽度（未旋转）
                        base_height,  # 页面原始高度（未旋转）
                        page_width,  # 应用旋转后页面宽度
                        page_height,  # 应用旋转后页面高度
                        img_width,  # 渲染图像宽度（像素）
                        img_height,  # 渲染图像高度（像素）
                    ),
                    page_width,  # 当前页面宽度（用于限制扩展）
                    page_height,  # 当前页面高度（用于限制扩展）
                )
                for box in boxes  # 遍历所有检测到的边界框
            ]

            # 遍历每一个票据分块
            for part_index, pdf_box in enumerate(page_boxes, start=1):
                # 票据计数加 1
                export_count += 1
                # 打印分块信息
                print(
                    f"  -> 添加分块 {part_index}/{len(page_boxes)} "
                    f"到合并文件（票据 #{export_count}），"
                    f"尺寸 {pdf_box.width:.2f} x {pdf_box.height:.2f}"
                )
                # 将裁剪后的页面添加到输出
                self._add_cropped_page_to_writer(reader, idx, pdf_box, merged_writer)

        # 生成输出文件名：原文件名 + "_receipts.pdf"
        output_filename = self.pdf_path.stem + "_receipts.pdf"
        # 构建完整输出路径
        output_path = self.output_dir / output_filename
        # 以二进制写模式打开输出文件
        with output_path.open("wb") as fh:
            # 将合并的 PDF 写入文件
            merged_writer.write(fh)

        # 关闭 PyMuPDF 文档对象
        doc.close()
        # 打印完成信息
        print(f"完成！共导出 {export_count} 个票据到：{output_path}")

    def _render_page(self, doc: fitz.Document, page_index: int) -> fitz.Pixmap:
        """按照指定 DPI 渲染给定页，并返回对应的位图。"""
        # 通过下标访问 PyMuPDF 文档中的目标页面
        page = doc[page_index]
        # 计算缩放因子：72 是 PDF 默认分辨率，将其转换为设定的 DPI
        zoom = self.dpi / 72
        # 构建渲染矩阵，设置水平和垂直缩放比例
        mat = fitz.Matrix(zoom, zoom)
        # 使用渲染矩阵生成位图，alpha=False 表示不需要透明通道
        pix = page.get_pixmap(matrix=mat, alpha=False)
        # 返回渲染后的位图对象，供后续检测使用
        return pix

    def _detect_image_boxes(self, pixmap: fitz.Pixmap) -> List[ImageBox]:
        """在渲染的位图中检测所有票据区域的边界框。"""
        # 获取位图的宽度和高度（单位：像素）
        width, height = pixmap.width, pixmap.height
        # 获取位图的原始像素数据（字节数组）
        samples = pixmap.samples
        # 获取每一行像素数据的字节跨度
        stride = pixmap.stride
        
        # 构建二维灰度图数组，将 RGB 转为灰度值
        gray_data = []
        # 遍历每一行
        for y in range(height):
            # 当前行的灰度值列表
            row = []
            # 遍历每一列
            for x in range(width):
                # 计算当前像素在字节数组中的偏移量（RGB 三通道，每像素 3 字节）
                offset = y * stride + x * 3
                # 提取红、绿、蓝三个通道的值
                r, g, b = samples[offset], samples[offset + 1], samples[offset + 2]
                # 根据加权公式将 RGB 转换为灰度值
                gray_val = int(0.299 * r + 0.587 * g + 0.114 * b)
                # 将灰度值加入当前行
                row.append(gray_val)
            # 将当前行加入灰度数据
            gray_data.append(row)
        
        # 使用 Otsu 算法自动计算二值化阈值
        threshold = self._otsu_threshold(gray_data, width, height)
        
        # 生成内容掩码：灰度值低于阈值的像素为 True（内容），否则为 False（空白）
        content_mask = [[gray_data[y][x] < threshold for x in range(width)] for y in range(height)]

        # 在行方向上根据空白区域切分，得到行段列表（每段代表一个水平条带）
        row_segments = self._split_axis(
            mask=content_mask,  # 内容掩码
            axis=0,  # 行方向
            blank_threshold=self.row_blank_threshold,  # 判断空白行的阈值
            min_gap_ratio=self.min_row_gap_ratio,  # 最小空白间隙比例
            min_segment_ratio=self.min_row_height_ratio,  # 最小段高度比例
        )

        # 如果没有检测到行段，则使用整个图像高度作为单段
        if not row_segments:
            row_segments = [(0, height)]

        # 初始化边界框列表
        boxes: List[ImageBox] = []
        # 遍历每一个行段
        for row_start, row_end in row_segments:
            # 提取该行段对应的内容掩码切片
            row_slice = [content_mask[y] for y in range(row_start, row_end)]
            # 在列方向上根据空白区域切分，得到列段列表
            col_segments = self._split_axis(
                mask=row_slice,  # 当前行段的掩码
                axis=1,  # 列方向
                blank_threshold=self.col_blank_threshold,  # 判断空白列的阈值
                min_gap_ratio=self.min_col_gap_ratio,  # 最小空白间隙比例
                min_segment_ratio=self.min_col_width_ratio,  # 最小段宽度比例
            )

            # 如果没有检测到列段，则使用整个图像宽度作为单段
            if not col_segments:
                col_segments = [(0, width)]

            # 遍历每一个列段，生成矩形区域
            for col_start, col_end in col_segments:
                # 初始化总像素数和内容像素数计数器
                total_pixels = 0
                content_pixels = 0
                # 统计当前矩形区域内的内容填充率
                for y in range(len(row_slice)):
                    for x in range(col_start, col_end):
                        # 总像素加 1
                        total_pixels += 1
                        # 如果当前像素是内容（True），内容像素加 1
                        if row_slice[y][x]:
                            content_pixels += 1
                
                # 计算填充率
                fill_ratio = content_pixels / total_pixels if total_pixels > 0 else 0
                # 如果填充率过低，认为该区域不是有效票据，跳过
                if fill_ratio < self.min_region_fill:
                    continue
                # 添加边界框到列表
                boxes.append(ImageBox(col_start, row_start, col_end, row_end))

        # 如果未检测到任何边界框，则将整个图像作为单个边界框
        if not boxes:
            boxes.append(ImageBox(0, 0, width, height))

        # 合并相互重叠的边界框
        boxes = self._merge_overlapping_boxes(boxes)
        # 按照从上到下、从左到右的顺序排序
        boxes.sort(key=lambda b: (b.top, b.left))
        # 返回最终的边界框列表
        return boxes

    def _split_axis(
        self,
        mask: List[List[bool]],
        axis: int,
        blank_threshold: float,
        min_gap_ratio: float,
        min_segment_ratio: float,
    ) -> List[Tuple[int, int]]:
        """沿给定轴根据空白分隔符拆分掩码，返回 (start, end) 段列表。"""
        if axis == 0:
            # 沿行方向拆分：长度为行数
            length = len(mask)
            # 计算每一行的内容密度（True 数量占总列数的比例）
            densities = [sum(row) / len(row) if len(row) > 0 else 0 for row in mask]
        else:
            # 沿列方向拆分：长度为列数
            length = len(mask[0]) if mask else 0
            densities = []
            for x in range(length):
                # 统计这一列的内容像素数量
                col_sum = sum(mask[y][x] for y in range(len(mask)))
                # 计算内容密度（占总行数的比例）
                densities.append(col_sum / len(mask) if len(mask) > 0 else 0)
        
        # 计算最小空白间隙长度，至少为 30 像素
        min_gap = max(30, int(length * min_gap_ratio))
        # 计算最小有效片段长度，至少为 60 像素
        min_segment = max(60, int(length * min_segment_ratio))

        # 根据密度阈值生成空白标记列表，True 表示该位置为空白
        blank_flags = [d < blank_threshold for d in densities]

        # 初始化结果列表
        segments: List[Tuple[int, int]] = []
        # 当前片段的起始位置
        start = 0
        # 扫描指针
        idx = 0
        # 遍历整个长度
        while idx < length:
            if blank_flags[idx]:
                # 遇到空白区域，记录空白开始位置
                gap_start = idx
                # 向后移动直到空白结束
                while idx < length and blank_flags[idx]:
                    idx += 1
                # 计算空白长度
                gap_len = idx - gap_start
                # 如果空白够宽，且之前的片段也足够长，则截断为一个有效段
                if gap_len >= min_gap and gap_start - start >= min_segment:
                    segments.append((start, gap_start))
                    # 更新下一个片段的起点为当前索引
                    start = idx
            else:
                # 非空白位置直接向后移动
                idx += 1

        # 处理最后一个片段，如果其长度满足要求则加入
        if length - start >= min_segment:
            segments.append((start, length))

        # 返回所有有效片段
        return segments

    def _merge_overlapping_boxes(self, boxes: Sequence[ImageBox]) -> List[ImageBox]:
        """合并相互重叠的矩形框，避免同一票据被多次识别。"""
        # 如果输入为空，直接返回空列表
        if not boxes:
            return []

        # 存放合并后的结果
        merged: List[ImageBox] = []
        # 遍历每一个待处理的边界框
        for box in boxes:
            # 标记该边界框是否已被合并
            placed = False
            # 遍历已经存入的边界框，看是否与当前边界框重叠
            for i, existing in enumerate(merged):
                if self._boxes_overlap(existing, box):
                    # 如果重叠，合并两个边界框，更新列表中的项
                    merged[i] = self._merge_two_boxes(existing, box)
                    placed = True
                    break
            # 如果没有与任何已存在框重叠，则直接加入结果列表
            if not placed:
                merged.append(box)
        # 返回合并后的边界框列表
        return merged

    @staticmethod
    def _boxes_overlap(a: ImageBox, b: ImageBox) -> bool:
        """判断两个矩形框是否存在重叠。"""
        # 判断水平方向是否有重叠
        horizontal_overlap = not (a.right < b.left or b.right < a.left)
        # 判断垂直方向是否有重叠
        vertical_overlap = not (a.bottom < b.top or b.bottom < a.top)
        # 只有水平和垂直都重叠才返回 True
        return horizontal_overlap and vertical_overlap

    @staticmethod
    def _merge_two_boxes(a: ImageBox, b: ImageBox) -> ImageBox:
        """合并两个矩形框为一个更大的矩形框，取两者的最小和最大边界。"""
        return ImageBox(
            left=min(a.left, b.left),  # 取左边界最小值
            top=min(a.top, b.top),  # 取上边界最小值
            right=max(a.right, b.right),  # 取右边界最大值
            bottom=max(a.bottom, b.bottom),  # 取下边界最大值
        )

    @staticmethod
    def _otsu_threshold(gray_data: List[List[int]], width: int, height: int) -> int:
        """使用 Otsu 算法根据灰度直方图计算自适应阈值。"""
        # 初始化 256 级灰度的直方图
        hist = [0] * 256
        # 遍历所有像素，统计对应灰度值的出现次数
        for row in gray_data:
            for val in row:
                hist[val] += 1
        
        # 像素总数
        total = width * height
        # 灰度值与频次的加权和，用于计算均值
        sum_total = sum(i * hist[i] for i in range(256))

        # 前景累计灰度和
        sum_b = 0.0
        # 前景累计权重（像素数）
        weight_b = 0.0
        # 最大类间方差初始值
        max_var = -1.0
        # 默认阈值初始值
        threshold = 127

        # 遍历所有可能的阈值
        for i in range(256):
            # 更新前景权重
            weight_b += hist[i]
            if weight_b == 0:
                continue
            # 背景权重 = 总像素 - 前景权重
            weight_f = total - weight_b
            if weight_f == 0:
                break
            # 更新前景灰度和
            sum_b += i * hist[i]
            # 计算前景平均灰度
            mean_b = sum_b / weight_b
            # 计算背景平均灰度
            mean_f = (sum_total - sum_b) / weight_f
            # 计算类间方差
            var_between = weight_b * weight_f * (mean_b - mean_f) ** 2
            # 如果类间方差更大，则更新阈值
            if var_between > max_var:
                max_var = var_between
                threshold = i

        # 稍微下调阈值，并限制在 [30, 240] 范围内，避免过亮/过暗干扰
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
        """将图像坐标系中的矩形转换为 PDF 坐标系中的矩形。"""
        # 先按比例将图像坐标映射到旋转后的页面宽度方向
        rot_left = (box.left / img_width) * page_width
        rot_right = (box.right / img_width) * page_width
        # 再按比例将图像坐标映射到旋转后的页面高度方向（PDF 坐标原点在左下）
        rot_top = page_height - (box.top / img_height) * page_height
        rot_bottom = page_height - (box.bottom / img_height) * page_height

        # 归一化左、右、上、下边界，防止数值颠倒
        left = min(rot_left, rot_right)
        right = max(rot_left, rot_right)
        bottom = min(rot_bottom, rot_top)
        top = max(rot_bottom, rot_top)
        # 返回转换后的 PDF 坐标矩形
        return PdfBox(left, bottom, right, top)

    def _expand_pdf_box(
        self, pdf_box: PdfBox, page_width: float, page_height: float
    ) -> PdfBox:
        """向外扩展矩形边界以增加裁剪边距，但不超出页面边界。"""
        # 如果边距设置为 0 或负数，则不扩展，直接返回
        if self.padding <= 0:
            return pdf_box

        # 向左扩展，但不能小于 0
        left = max(0, pdf_box.left - self.padding)
        # 向下扩展，但不能小于 0
        bottom = max(0, pdf_box.bottom - self.padding)
        # 向右扩展，但不能超出页面宽度
        right = min(page_width, pdf_box.right + self.padding)
        # 向上扩展，但不能超出页面高度
        top = min(page_height, pdf_box.top + self.padding)
        # 返回扩展后的边界框
        return PdfBox(left, bottom, right, top)

    def _add_full_page_to_writer(
        self, reader: PdfReader, page_index: int, writer: PdfWriter
    ) -> None:
        """将完整页面（未拆分）添加到 PDF 写出器。"""
        # 深拷贝页面对象，避免修改原数据
        # 将拷贝后的页面添加到输出的 PDF
        writer.add_page(copy.deepcopy(reader.pages[page_index]))

    def _add_cropped_page_to_writer(
        self,
        reader: PdfReader,
        page_index: int,
        pdf_box: PdfBox,
        writer: PdfWriter,
    ) -> None:
        """按照给定的 PDF 边界框裁剪页面并写入输出文档。"""
        # 深拷贝原页面，避免直接修改 reader 中的数据
        page = copy.deepcopy(reader.pages[page_index])
        # 获取页面的旋转角度
        rotation = int(page.get("/Rotate") or 0) % 360
        
        # 如果页面存在旋转，需要先矫正旋转，使页面回到未旋转状态
        if rotation != 0:
            # 记录原始 MediaBox 的宽度和高度
            base_width = float(page.mediabox.width)
            base_height = float(page.mediabox.height)
            
            # 将页面的 /Rotate 属性重置为 0，后续使用变换矩阵完成矫正
            page[NameObject("/Rotate")] = NumberObject(0)
            
            if rotation == 90:
                # 旋转 90° 时，宽高互换，并通过矩阵反向旋转
                page.mediabox = RectangleObject([0, 0, base_height, base_width])
                transformation = Transformation().rotate(-90).translate(0, base_width)
                page.add_transformation(transformation)
            elif rotation == 180:
                # 旋转 180° 时，需要旋转回 -180° 并平移回原位置
                page.mediabox = RectangleObject([0, 0, base_width, base_height])
                transformation = Transformation().rotate(-180).translate(base_width, base_height)
                page.add_transformation(transformation)
            elif rotation == 270:
                # 旋转 270° 时，同样需要宽高互换并反向旋转
                page.mediabox = RectangleObject([0, 0, base_height, base_width])
                transformation = Transformation().rotate(-270).translate(base_height, 0)
                page.add_transformation(transformation)
        
        # 将页面平移，使得目标区域的左下角对齐到原点
        transformation = Transformation().translate(-pdf_box.left, -pdf_box.bottom)
        page.add_transformation(transformation)

        # 构建裁剪矩形，宽高与目标区域一致
        box_rect = RectangleObject([0, 0, pdf_box.width, pdf_box.height])
        # 同步更新页面的 MediaBox / CropBox / TrimBox / ArtBox，确保裁剪生效
        page.mediabox = box_rect
        page.cropbox = box_rect
        page.trimbox = box_rect
        page.artbox = box_rect

        # 将裁剪后的页面加入写出器
        writer.add_page(page)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，返回命名空间对象。"""
    # 创建参数解析器，并添加脚本描述
    parser = argparse.ArgumentParser(description="自动拆分 PDF 中的多张票据")
    # 添加必选参数：PDF 文件路径
    parser.add_argument("pdf", type=Path, help="需要处理的 PDF 文件路径")
    # 添加可选参数：输出目录
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output"),
        help="输出目录（默认为 ./output）",
    )
    # 添加可选参数：渲染 DPI
    parser.add_argument("--dpi", type=int, default=300, help="渲染 PDF 时使用的 DPI")
    # 添加可选参数：裁剪边距
    parser.add_argument(
        "--padding",
        type=float,
        default=12.0,
        help="裁剪区域向外扩展的边距（单位 pt，默认 12）",
    )
    # 执行解析并返回结果
    return parser.parse_args()


def main() -> None:
    """主函数，协调参数解析和票据拆分流程。"""
    # 解析命令行参数
    args = parse_args()
    # 检查 PDF 文件是否存在
    if not args.pdf.exists():
        raise FileNotFoundError(f"未找到 PDF 文件：{args.pdf}")

    # 创建票据拆分器实例，传入文件路径和参数
    splitter = ReceiptSplitter(
        pdf_path=args.pdf, output_dir=args.output, dpi=args.dpi, padding=args.padding
    )
    # 执行拆分任务
    splitter.run()


# 程序入口：当脚本被直接执行时运行 main 函数
if __name__ == "__main__":
    main()
