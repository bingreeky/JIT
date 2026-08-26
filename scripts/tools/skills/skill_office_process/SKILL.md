# Office Document Processing Skill

When creating or manipulating office documents (PDF, PPTX, XLSX) or generating charts, use the `execute_code` tool with the following Python libraries and patterns.

## PDF Creation

### Simple PDFs — use `fpdf2`
```python
from fpdf import FPDF

pdf = FPDF()
pdf.add_page()
pdf.set_font("Helvetica", size=16)
pdf.cell(text="Title", new_x="LMARGIN", new_y="NEXT", align="C")
pdf.set_font("Helvetica", size=12)
pdf.multi_cell(w=0, text="Body text here...")
pdf.output("output.pdf")
```

### Complex PDFs with tables/images — use `reportlab`
```python
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

doc = SimpleDocTemplate("report.pdf", pagesize=letter)
styles = getSampleStyleSheet()
elements = []
elements.append(Paragraph("Report Title", styles['Title']))
elements.append(Paragraph("Body text...", styles['Normal']))

# Table
data = [["Header1", "Header2"], ["Row1Col1", "Row1Col2"]]
table = Table(data)
table.setStyle(TableStyle([
    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
    ('GRID', (0, 0), (-1, -1), 1, colors.black),
]))
elements.append(table)

# Include an image
elements.append(Image("chart.png", width=400, height=300))
doc.build(elements)
```

## PowerPoint (PPTX) Creation — use `python-pptx`

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN

prs = Presentation()

# Title slide
slide = prs.slides.add_slide(prs.slide_layouts[0])
slide.shapes.title.text = "Presentation Title"
slide.placeholders[1].text = "Subtitle"

# Content slide with bullet points
slide = prs.slides.add_slide(prs.slide_layouts[1])
slide.shapes.title.text = "Key Points"
body = slide.placeholders[1]
body.text = "First point"
for point in ["Second point", "Third point"]:
    p = body.text_frame.add_paragraph()
    p.text = point
    p.level = 0

# Slide with image
slide = prs.slides.add_slide(prs.slide_layouts[5])  # blank layout
slide.shapes.add_picture("chart.png", Inches(1), Inches(1), width=Inches(8))

# Slide with table
slide = prs.slides.add_slide(prs.slide_layouts[5])
table = slide.shapes.add_table(rows=3, cols=3, left=Inches(1), top=Inches(1.5), width=Inches(8), height=Inches(2)).table
table.cell(0, 0).text = "Header"

prs.save("output.pptx")
```

## Excel (XLSX) Creation — use `openpyxl`

```python
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.utils import get_column_letter

wb = Workbook()
ws = wb.active
ws.title = "Data"

# Headers with styling
headers = ["Name", "Value", "Category"]
header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
header_font = Font(color="FFFFFF", bold=True)
for col, header in enumerate(headers, 1):
    cell = ws.cell(row=1, column=col, value=header)
    cell.fill = header_fill
    cell.font = header_font

# Data rows
data = [("Item A", 100, "Type 1"), ("Item B", 200, "Type 2")]
for row_idx, row_data in enumerate(data, 2):
    for col_idx, value in enumerate(row_data, 1):
        ws.cell(row=row_idx, column=col_idx, value=value)

# Formulas — IMPORTANT: always use Excel formulas, not hardcoded values
ws.cell(row=4, column=2, value="=SUM(B2:B3)")
ws.cell(row=4, column=1, value="Total")

# Column widths
for col in range(1, 4):
    ws.column_dimensions[get_column_letter(col)].width = 15

# Add chart
chart = BarChart()
chart.title = "Values by Category"
data_ref = Reference(ws, min_col=2, min_row=1, max_row=3)
cats_ref = Reference(ws, min_col=1, min_row=2, max_row=3)
chart.add_data(data_ref, titles_from_data=True)
chart.set_categories(cats_ref)
ws.add_chart(chart, "E2")

wb.save("output.xlsx")
```

## Charts and Visualizations — use `matplotlib`

```python
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend — ALWAYS set this
import matplotlib.pyplot as plt
import numpy as np

# Line chart
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(2020, 2026)
y = [100, 120, 115, 140, 160, 180]
ax.plot(x, y, marker='o', linewidth=2, label='Revenue')
ax.set_xlabel('Year')
ax.set_ylabel('Revenue ($M)')
ax.set_title('Annual Revenue Trend')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Bar chart
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['A', 'B', 'C', 'D']
values = [25, 40, 30, 55]
ax.bar(categories, values, color=['#4472C4', '#ED7D31', '#A5A5A5', '#FFC000'])
ax.set_title('Category Comparison')
plt.savefig('bar_chart.png', dpi=150, bbox_inches='tight')
plt.close()

# Pie chart
fig, ax = plt.subplots(figsize=(8, 8))
sizes = [35, 25, 20, 20]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
ax.set_title('Distribution')
plt.savefig('pie_chart.png', dpi=150, bbox_inches='tight')
plt.close()
```

## Reading Existing Documents

### Read PDF
```python
import pdfminer.high_level
text = pdfminer.high_level.extract_text("input.pdf")
print(text)
```

### Read Excel
```python
import openpyxl
wb = openpyxl.load_workbook("input.xlsx", data_only=True)
ws = wb.active
for row in ws.iter_rows(values_only=True):
    print(row)
```

### Read PowerPoint
```python
from pptx import Presentation
prs = Presentation("input.pptx")
for slide in prs.slides:
    for shape in slide.shapes:
        if shape.has_text_frame:
            print(shape.text_frame.text)
```

## Key Reminders
- Always use `matplotlib.use('Agg')` before importing pyplot (no display server available)
- For Excel: use formulas (`=SUM(...)`) not hardcoded calculated values
- For charts embedded in reports: save as PNG first, then embed via reportlab or python-pptx
- For Chinese text in matplotlib: set `plt.rcParams['font.sans-serif'] = ['SimHei']` or use a CJK font
- Print file paths of created files so the observation shows what was produced
