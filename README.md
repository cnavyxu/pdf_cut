# PDF 票据自动分割工具

## 目标
遍历每一页 PDF，对于单页中存在多张票据时，需要将单页 PDF 裁剪成多张 PDF，裁剪后的 PDF 页面尺寸和原始尺寸保持一致。

## 数据样例
可参考【农行影印版.pdf】文件

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基本用法

```bash
python split_receipts.py 农行影印版.pdf
```

这会将处理后的 PDF 文件输出到 `output/` 目录。

### 指定输出目录

```bash
python split_receipts.py 农行影印版.pdf -o my_output
```

### 调整 DPI（默认 300）

```bash
python split_receipts.py 农行影印版.pdf --dpi 200
```

## 工作原理

1. **页面渲染**：使用 PyMuPDF 将每页 PDF 渲染为高清图像
2. **内容检测**：通过 Otsu 二值化算法和密度分析检测页面上的内容区域
3. **区域分割**：根据行列的空白间隙自动分割出多个票据区域
4. **PDF 裁剪**：使用 PyPDF2 将检测到的每个区域裁剪成独立的 PDF 文件

## 代码结构与变量说明

为了避免在源码中插入大量行内注释，本节通过「逐段讲解」的方式解释 `split_receipts.py` 中的类、函数以及关键变量，可结合行号对照阅读。

### 核心参数速查
- `dpi`：渲染页面时使用的分辨率，数值越高检测越精细、耗时越长。
- `row_blank_threshold` / `col_blank_threshold`：判断某一行/列是否为空白的内容密度阈值，范围 0~1，越小越严格。
- `min_row_gap_ratio` / `min_col_gap_ratio`：行/列方向判定「空白间隙」时要求的最小比例，用于避免被噪点打断。
- `min_row_height_ratio` / `min_col_width_ratio`：过滤掉过小内容块的阈值，防止把噪声当成票据。
- `min_region_fill`：一个候选区域内实际有内容的像素占比下限。
- `padding`：最终裁剪框向四周扩展的边距（pt），避免裁掉边缘信息。

### 函数与方法说明

| 代码位置 | 说明 |
| --- | --- |
| `ImageBox`（33-52 行） | 表示渲染图像上的矩形区域，提供宽高计算、元组转换等辅助方法。 |
| `PdfBox`（54-68 行） | 表示 PDF 坐标空间内的矩形区域，对应 `ImageBox` 换算后的结果。 |
| `ReceiptSplitter.__init__`（70-95 行） | 保存输入路径、输出目录和一系列检测阈值，供后续流程使用。 |
| `ReceiptSplitter.run`（97-165 行） | 主流程：逐页渲染、检测候选框、按需裁剪并写入合并输出。 |
| `_render_page`（166-171 行） | 使用 PyMuPDF 按指定 DPI 渲染单页，返回 `Pixmap`。 |
| `_detect_image_boxes`（173-235 行） | 将页面图像二值化后，通过行/列密度分析找出票据区域，并过滤空白区。 |
| `_split_axis`（238-279 行） | 在行或列方向根据密度曲线寻找足够大的空白间隙，拆分内容块。 |
| `_merge_overlapping_boxes` / `_boxes_overlap` / `_merge_two_boxes`（281-310 行） | 处理检测结果中的重叠区域，保证每个票据区域互不覆盖。 |
| `_otsu_threshold`（313-343 行） | 实现 Otsu 算法，用于计算全局二值化阈值。 |
| `_image_box_to_pdf_box`（345-365 行） | 将图像坐标转换成 PDF 坐标，同时处理纵轴方向的差异。 |
| `_expand_pdf_box`（367-378 行） | 根据 `padding` 对裁剪框加入安全边距，并限制在页面范围内。 |
| `_add_full_page_to_writer`（379-383 行） | 当一页只检测到一张票据时，直接把整页复制到输出。 |
| `_add_cropped_page_to_writer`（384-423 行） | 负责去除页面旋转、平移内容并设置新的裁剪尺寸，是裁剪的关键步骤。 |
| `parse_args`（425-442 行） | 定义命令行参数，允许用户调整输出目录、DPI、padding 等配置。 |
| `main`（445-453 行） | 程序入口：解析参数并执行 `ReceiptSplitter.run()`。 |

## 输出说明

输出的 PDF 文件命名规则：
- 单票据页面：`page-XX-receipt-XXX.pdf`
- 多票据页面：`page-XX-part-XX-receipt-XXX.pdf`

其中：
- `page-XX`：原始页码
- `part-XX`：在该页中的票据序号
- `receipt-XXX`：全局票据序号

## 依赖库

- **PyPDF2**：用于 PDF 文件的读取和裁剪
- **PyMuPDF (fitz)**：用于高质量 PDF 页面渲染

## 注意事项

- 裁剪后的 PDF 保持实际内容区域的尺寸，不是原始整页尺寸
- 对于无法检测到明确分割区域的页面，会保留整页输出
- 检测算法针对扫描件票据优化，对其他类型文档可能需要调整参数
