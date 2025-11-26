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
