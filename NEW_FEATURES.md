# DeepSeek-OCR Studio Pro - New Features Guide

## 🎉 What's New

This enhanced version of DeepSeek-OCR Studio includes 7 major feature additions:

1. **Batch Folder Processing** with job queue and progress persistence
2. **Additional Output Formats** (JSON, HTML, DOCX, CSV/Excel)
3. **OCR Comparison Tool** for testing different modes and prompts
4. **Interactive Editor** with markdown editing
5. **Multi-Language Support** (5 languages)
6. **Microsoft Office Format Support** (DOCX, PPTX, XLSX input)
7. **Intelligent Post-Processing** (spell-check, grammar, validation)

---

## 📁 Feature 1: Batch Folder Processing

### Overview
Process hundreds of documents with persistent job tracking. Jobs survive application restarts.

### How to Use

1. Navigate to the **"Batch Processing"** tab
2. Enter a job name
3. Upload multiple files (PDF, images, Office documents)
4. Click **"Create Batch Job"**
5. Monitor progress in real-time
6. Jobs are saved to SQLite database (`jobs.db`)

### Features

- **Progress Persistence**: Resume interrupted jobs
- **Status Tracking**: View pending, processing, completed, failed files
- **Job Management**: Cancel, delete, or view results
- **Concurrent Processing**: Process multiple files in parallel
- **Result Storage**: All outputs saved with job metadata

### Database Schema

```sql
jobs (job_id, name, status, total_files, processed_files, config, timestamps)
job_files (file_id, job_id, filename, status, result_path, error_message)
job_results (result_id, job_id, file_id, output_format, output_path)
```

### API Usage

```python
from utils.job_queue import JobQueue, JobStatus

# Create job queue
queue = JobQueue("jobs.db")

# Create a new job
job_id = queue.create_job(
    name="My Batch Job",
    files=["doc1.pdf", "doc2.pdf"],
    config={"resolution_mode": "Base", "prompt": "..."}
)

# Check progress
progress = queue.get_job_progress(job_id)
print(f"Progress: {progress['progress']}%")

# Get results
results = queue.get_job_results(job_id)
```

---

## 📊 Feature 2: Additional Output Formats

### Supported Formats

| Format | Extension | Use Case |
|--------|-----------|----------|
| **JSON** | `.json` | Structured data with metadata and coordinates |
| **HTML** | `.html` | Styled webpage with CSS |
| **DOCX** | `.docx` | Editable Microsoft Word document |
| **CSV** | `.csv` | Simple table export |
| **Excel** | `.xlsx` | Formatted spreadsheet with styling |

### JSON Output Structure

```json
{
  "text": "Cleaned OCR text...",
  "elements": [
    {
      "type": "title",
      "coordinates": [[x1, y1, x2, y2], ...],
      "raw": "<|ref|>title<|/ref|>..."
    }
  ],
  "tables": [
    {
      "headers": ["Col1", "Col2"],
      "rows": [["val1", "val2"], ...],
      "raw": "| markdown table |"
    }
  ],
  "metadata": {
    "filename": "document.pdf",
    "page": 1
  },
  "dimensions": {
    "width": 2480,
    "height": 3508
  }
}
```

### HTML Output

- **Styled Headers**: H1, H2, H3 with colors
- **Formatted Tables**: Styled with CSS
- **Code Blocks**: Syntax highlighting
- **Responsive Design**: Mobile-friendly
- **Print-Ready**: Optimized for PDF conversion

### DOCX Output

- **Proper Heading Styles**: Heading 1, 2, 3
- **Lists**: Bullet and numbered lists
- **Metadata**: Document properties
- **Editable**: Full Microsoft Word compatibility

### Excel Output

- **Multiple Sheets**: One per table
- **Styled Headers**: Bold, colored background
- **Auto-Width Columns**: Optimized readability
- **Cell Formatting**: Aligned and bordered

### How to Use

1. Process your document in the **"Upload & Process"** tab
2. Go to **"Results"** → **"Downloads"** tab
3. Choose your desired format:
   - Click **"Download JSON"** for structured data
   - Click **"Download HTML"** for styled webpage
   - Click **"Download DOCX"** for Word document
   - Click **"Download CSV/Excel"** for spreadsheet

### Programmatic Usage

```python
from utils.output_formatters import JSONFormatter, HTMLFormatter, DOCXFormatter

# JSON
json_output = JSONFormatter.format(text, matches, width, height, metadata)

# HTML
html_output = HTMLFormatter.format(text, matches, metadata)

# DOCX
DOCXFormatter.format(text, matches, "output.docx", metadata)

# Excel
ExcelFormatter.format(text, "output.xlsx")
```

---

## 🔄 Feature 3: OCR Comparison Tool

### Overview
Compare OCR results across different settings to find optimal configuration.

### What You Can Compare

- **Resolution Modes**: Tiny, Small, Base, Large, Gundam
- **Prompt Templates**: Different prompts for different document types
- **Side-by-Side View**: Visual comparison of results

### How to Use

1. Process a document first
2. Navigate to **"Comparison"** tab
3. Select file to compare
4. Choose 2+ resolution modes
5. Choose 2+ prompt templates
6. Click **"Compare"**
7. View results side-by-side

### Use Cases

- **Quality vs Speed**: Compare Tiny (fast) vs Gundam (quality)
- **Prompt Optimization**: Find best prompt for your document type
- **Performance Testing**: Measure accuracy across modes
- **Document Type**: Test which mode works best for forms, tables, etc.

### Comparison Metrics

- Processing time
- Output quality
- Token count
- Table detection accuracy
- Formula recognition

---

## ✏️ Feature 4: Interactive Editor

### Overview
Edit OCR results with live preview and save changes.

### Features

- **Markdown Editor**: Full-featured text editing
- **Live Preview**: See changes in real-time
- **Save Changes**: Store edited versions
- **Download**: Export edited text
- **Undo/Redo**: Built into text area

### How to Use

1. Process documents
2. Go to **"Editor"** tab
3. Select file and page
4. Edit text in the editor
5. Preview changes below
6. Click **"Save Changes"** or **"Download"**

### Future Enhancements

- Bounding box adjustment (drag-and-drop)
- Re-process specific regions
- Collaborative editing
- Version control

---

## 🌐 Feature 5: Multi-Language Support

### Supported Languages

| Language | Code | Status |
|----------|------|--------|
| **English** | `en` | ✅ Complete |
| **Español** | `es` | ✅ Complete |
| **中文** | `zh` | ✅ Complete |
| **Français** | `fr` | ✅ Partial |
| **Deutsch** | `de` | ✅ Partial |

### How to Use

1. Click **Language selector** in sidebar
2. Choose your preferred language
3. UI updates instantly
4. Settings persist across sessions

### Adding New Languages

1. Create translation file in `locales/`
2. Use ISO 639-1 code (e.g., `ja.json` for Japanese)
3. Follow the structure in `utils/i18n.py`

Example translation file (`locales/ja.json`):

```json
{
  "app": {
    "title": "DeepSeek-OCR スタジオ",
    "subtitle": "プレゼンテーション、PDF、表とグラフィックスを含む文書から情報を抽出"
  },
  "sidebar": {
    "configuration": "設定"
  }
}
```

### Translation Keys

All text uses translation keys:

```python
from utils.i18n import I18n

i18n = I18n('es')  # Spanish
title = i18n.t('app.title')  # "DeepSeek-OCR Studio"

# With parameters
message = i18n.t('upload.files_uploaded', count=5)
# "¡5 archivo(s) subido(s) exitosamente!"
```

---

## 📋 Feature 6: Microsoft Office Format Support

### Supported Formats

- **DOCX**: Microsoft Word documents
- **PPTX**: PowerPoint presentations
- **XLSX**: Excel spreadsheets

### How It Works

Office documents are converted to images automatically:

1. **DOCX**: Extracts paragraphs, renders to images
2. **PPTX**: Converts slides to images, preserves layout
3. **XLSX**: Renders sheets as tables to images

### Quality Settings

- Adjust **PDF DPI** in sidebar (72-300)
- Higher DPI = better quality, slower processing
- Recommended: 150 DPI for Office documents

### How to Use

1. Upload DOCX, PPTX, or XLSX file
2. Process normally
3. OCR extracts text, tables, and structure
4. Export to any format

### Conversion Details

#### DOCX Conversion
- Preserves headers, paragraphs, lists
- Renders with DejaVu fonts
- Multiple pages if needed
- Line wrapping at 80 characters

#### PPTX Conversion
- One image per slide
- Extracts text from shapes
- Embeds images from slides
- 16:9 aspect ratio

#### XLSX Conversion
- One image per sheet
- First 50 rows, 15 columns
- Styled header row
- Auto-adjusted column widths
- Cell borders and grid

### Limitations

- Complex formatting may not render perfectly
- Embedded objects (videos, audio) are ignored
- Macros and VBA code are not executed
- Charts rendered as static images

### Dependencies

```bash
pip install python-docx python-pptx openpyxl
```

---

## 🔍 Feature 7: Intelligent Post-Processing

### Overview
Automatic quality improvements and validation.

### Features

#### 1. **Spell Checking**
- Detects misspelled words
- Suggests corrections
- Applies automatic fixes
- Skips technical terms

#### 2. **Grammar Checking**
- Sentence structure validation
- Punctuation fixes
- Space normalization
- Common error correction

#### 3. **Table Validation**
- Column count consistency
- Row alignment checks
- Empty cell detection
- Structure verification

#### 4. **Formula Validation**
- LaTeX syntax checking
- Bracket/brace matching
- Command verification
- OCR artifact detection

### How to Use

1. Open sidebar **"Post-Processing"** section
2. Enable desired features:
   - ☑️ Enable Spell Check
   - ☑️ Enable Grammar Check
   - ☑️ Validate Tables
   - ☑️ Validate Formulas
3. Process documents
4. View issues and corrections in **Results** tab

### Post-Processing Report

```python
{
  "spelling_errors": [
    {"word": "teh", "correction": "the", "type": "spelling"}
  ],
  "grammar_issues": [
    {"error": "  ", "suggestion": " ", "message": "...", "type": "grammar"}
  ],
  "table_issues": [
    {"table_index": 0, "issue": "Column mismatch", "severity": "warning"}
  ],
  "formula_issues": [
    {"formula_index": 0, "issue": "Unmatched braces: 1", "severity": "error"}
  ],
  "corrections_applied": 15
}
```

### Quality Analysis

View detailed statistics:

- Character/word/line/paragraph counts
- Average word length
- Table count
- Formula count
- Code block count
- Special character ratio

### Programmatic Usage

```python
from utils.post_processing import PostProcessor, TextQualityAnalyzer

# Post-processing
processor = PostProcessor(
    enable_spellcheck=True,
    enable_grammar=True,
    enable_table_validation=True,
    enable_formula_check=True
)

processed_text, issues = processor.process(text)

# Quality analysis
analysis = TextQualityAnalyzer.analyze(text)
print(f"Words: {analysis['word_count']}")
print(f"Tables: {analysis['table_count']}")
```

### Common OCR Errors Fixed

| Error | Correction | Type |
|-------|------------|------|
| `I` in lowercase | `l` | Character confusion |
| `0` in word | `o` or `O` | Zero vs letter O |
| `rn` | `m` | Adjacent character merge |
| `teh` | `the` | Common typo |
| Double spaces | Single space | Formatting |
| Missing space after `.` | Add space | Punctuation |

---

## 🚀 Installation & Setup

### Install All Dependencies

```bash
# Core dependencies
pip install -r requirements.txt

# Optional (for advanced features)
pip install language-tool-python  # Grammar checking
pip install pypandoc  # Advanced DOCX conversion
```

### Database Setup

Batch processing uses SQLite (no setup required):

```bash
# Database created automatically on first use
ls -lh jobs.db
```

### Language Files

Translation files in `locales/`:

```bash
locales/
├── en.json  # English (built-in)
├── es.json  # Spanish (built-in)
├── zh.json  # Chinese (built-in)
├── fr.json  # French (built-in)
└── de.json  # German (built-in)
```

---

## 📖 API Reference

### Job Queue

```python
from utils.job_queue import JobQueue

queue = JobQueue("jobs.db")

# Create job
job_id = queue.create_job(name, files, config)

# Get job
job = queue.get_job(job_id)

# Update status
queue.update_job_status(job_id, JobStatus.COMPLETED)

# Get progress
progress = queue.get_job_progress(job_id)

# Delete job
queue.delete_job(job_id)
```

### Output Formatters

```python
from utils.output_formatters import (
    JSONFormatter, HTMLFormatter, DOCXFormatter,
    CSVFormatter, ExcelFormatter
)

# JSON
json_data = JSONFormatter.format(text, matches, width, height, metadata)

# HTML
html = HTMLFormatter.format(text, matches, metadata)

# DOCX
DOCXFormatter.format(text, matches, "output.docx", metadata)

# CSV
CSVFormatter.format(text, "output.csv")

# Excel
ExcelFormatter.format(text, "output.xlsx")
```

### Office Converters

```python
from utils.office_converters import OfficeConverter

# Convert to images
images = OfficeConverter.convert_to_images(
    file_bytes,
    file_type='docx',  # or 'pptx', 'xlsx'
    dpi=150
)
```

### Post-Processing

```python
from utils.post_processing import (
    PostProcessor, SpellChecker, GrammarChecker,
    TableValidator, FormulaValidator, TextQualityAnalyzer
)

# Full processing
processor = PostProcessor(
    enable_spellcheck=True,
    enable_grammar=True,
    enable_table_validation=True,
    enable_formula_check=True
)
processed_text, issues = processor.process(text)

# Individual components
corrected_text, spelling_issues = SpellChecker.check_and_correct(text)
corrected_text, grammar_issues = GrammarChecker.check_and_correct(text)
table_issues = TableValidator.validate(text)
formula_issues = FormulaValidator.validate(text)
analysis = TextQualityAnalyzer.analyze(text)
```

### Internationalization

```python
from utils.i18n import I18n

# Initialize with locale
i18n = I18n('es')

# Translate
title = i18n.t('app.title')

# With parameters
message = i18n.t('upload.files_uploaded', count=5)

# Change locale
i18n.set_locale('zh')
```

---

## 🎯 Best Practices

### Batch Processing

- **Chunk Large Jobs**: Break into 50-100 files per job
- **Monitor Progress**: Check job status periodically
- **Clean Old Jobs**: Delete completed jobs to save space

### Output Formats

- **JSON**: Best for programmatic processing
- **HTML**: Best for sharing/viewing
- **DOCX**: Best for editing
- **Excel**: Best for tables/data

### Office Files

- **Use Higher DPI**: 200-300 for complex documents
- **Simplify First**: Remove animations, transitions from PPTX
- **Split Large Files**: Process XLSX sheet-by-sheet if >1000 rows

### Post-Processing

- **Enable Selectively**: Spell-check slows processing
- **Validate Tables**: Always enable for forms/reports
- **Check Formulas**: Critical for scientific documents
- **Review Issues**: Don't blindly accept all corrections

### Performance

- **Batch Mode**: Use batch processing for 10+ files
- **Resolution**: Use Small/Base for most documents
- **Workers**: Set to CPU count for preprocessing
- **GPU Memory**: 0.9 for dedicated GPUs, 0.7 for shared

---

## 🐛 Troubleshooting

### Batch Jobs Stuck

```bash
# Check database
sqlite3 jobs.db "SELECT * FROM jobs WHERE status='processing';"

# Cancel stuck jobs
sqlite3 jobs.db "UPDATE jobs SET status='failed' WHERE status='processing';"
```

### Office Conversion Fails

```bash
# Install missing dependencies
pip install python-docx python-pptx openpyxl

# Check font availability
ls /usr/share/fonts/truetype/dejavu/
```

### Post-Processing Slow

```python
# Disable spell-check for speed
processor = PostProcessor(
    enable_spellcheck=False,  # Slow
    enable_grammar=False,     # Very slow
    enable_table_validation=True,
    enable_formula_check=True
)
```

### Language Not Loading

```bash
# Check translation file exists
ls locales/es.json

# Verify JSON syntax
python -m json.tool locales/es.json
```

---

## 🔮 Future Enhancements

Planned features:

- [ ] REST API endpoints
- [ ] Docker deployment
- [ ] Cloud storage integration (S3, GCS)
- [ ] Webhook notifications
- [ ] Advanced bounding box editor
- [ ] Model fine-tuning interface
- [ ] More output formats (EPUB, PDF)
- [ ] Collaborative editing
- [ ] OCR accuracy metrics
- [ ] Custom model support

---

## 📝 License

Same as DeepSeek-OCR. See main repository for details.

---

## 🤝 Contributing

To add features:

1. Create utility module in `utils/`
2. Update `app.py` with UI
3. Add to `requirements.txt`
4. Update documentation
5. Test thoroughly
6. Submit pull request

---

**Built with ❤️ for the DeepSeek-OCR community**
