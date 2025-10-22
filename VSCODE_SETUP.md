# VSCode Setup Guide for DeepSeek-OCR

Complete guide to set up VSCode for DeepSeek-OCR on macOS Apple Silicon.

## Initial Setup

### 1. Install Required VSCode Extensions

The repository includes recommended extensions. When you open the folder, VSCode will prompt you to install them:

- **Python** (ms-python.python) - Required
- **Pylance** (ms-python.vscode-pylance) - Required
- **Python Debugger** (ms-python.debugpy) - Required
- **Jupyter** (ms-toolsai.jupyter) - Optional
- **Markdown All in One** (yzhang.markdown-all-in-one) - Recommended
- **PDF Preview** (tomoki1207.pdf) - Recommended

Or install manually:
1. Press `Cmd+Shift+X`
2. Search for "Python"
3. Install "Python" by Microsoft

### 2. Open the Repository

```bash
# In terminal
cd /path/to/DeepSeek-OCR
code .
```

Or in VSCode: `File > Open Folder` → select `DeepSeek-OCR`

### 3. Select Python Interpreter

1. Press `Cmd+Shift+P`
2. Type: `Python: Select Interpreter`
3. Choose: `./venv/bin/python` (or create venv first - see below)

---

## Creating Virtual Environment in VSCode

### Option 1: Using Integrated Terminal

1. Open terminal: `` Cmd+` ``
2. Run commands:
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements_macos.txt
```

### Option 2: Using Command Palette

1. Press `Cmd+Shift+P`
2. Type: `Python: Create Environment`
3. Select `Venv`
4. Choose Python 3.9+ interpreter
5. Check `requirements_macos.txt`
6. Click OK

---

## VSCode Configuration Files

The `.vscode/` folder contains:

### settings.json
- Sets default Python interpreter to `./venv/bin/python`
- Configures file associations (`.mmd` files as markdown)
- Excludes unnecessary folders from search/watcher
- Optimizes performance

### launch.json
Debug configurations:
- **Python: Current File** - Run any Python file
- **macOS: Parse PDF** - Run the example script
- **Transformers: Run OCR** - Run transformers implementation

### extensions.json
Recommended extensions list

---

## Running Scripts in VSCode

### Method 1: Run Button (Easiest)

1. Open `example_macos_pdf_parse.py`
2. Click ▶️ button in top-right corner
3. Output appears in integrated terminal

### Method 2: Right-Click

1. Right-click on `.py` file
2. Select "Run Python File in Terminal"

### Method 3: Integrated Terminal

1. Open terminal: `` Cmd+` ``
2. Make sure `(venv)` appears in prompt
3. Run: `python example_macos_pdf_parse.py`

### Method 4: Debug Mode

1. Open file you want to debug
2. Press `F5` or click "Run and Debug" (left sidebar)
3. Select debug configuration
4. Set breakpoints by clicking left of line numbers

---

## Useful Keyboard Shortcuts

| Action | Shortcut |
|--------|----------|
| Open terminal | `` Cmd+` `` |
| Command palette | `Cmd+Shift+P` |
| Run file | `Ctrl+Option+N` |
| Debug | `F5` |
| Stop debugging | `Shift+F5` |
| Toggle breakpoint | `F9` |
| New file | `Cmd+N` |
| Save | `Cmd+S` |
| Find in files | `Cmd+Shift+F` |
| Explorer | `Cmd+Shift+E` |

---

## Editing Python Files

### IntelliSense (Auto-completion)

- Start typing → suggestions appear automatically
- Press `Ctrl+Space` to manually trigger
- Press `Tab` or `Enter` to accept suggestion

### Quick Documentation

- Hover over function/class → see documentation
- Press `Cmd+K Cmd+I` for inline documentation

### Go to Definition

- `Cmd+Click` on function name
- Or: `F12`

### Find All References

- Right-click → "Find All References"
- Or: `Shift+F12`

### Format Document

- Press `Shift+Option+F`
- Or: Right-click → "Format Document"

---

## Working with PDFs

### Preview PDFs

1. Install "PDF Preview" extension
2. Click on `.pdf` file
3. Preview opens in editor

### Compare Before/After

Use Split Editor:
1. Open original PDF
2. Press `Cmd+\` to split editor
3. Open `output/filename_layouts.pdf` in other pane

---

## Terminal Tips

### Activate Virtual Environment

If `(venv)` is not showing:
```bash
source venv/bin/activate
```

### Check Environment

```bash
which python     # Should show: .../venv/bin/python
python --version # Should be 3.9+
pip list         # Show installed packages
```

### Multiple Terminals

- Click `+` in terminal panel for new terminal
- Click split icon for side-by-side terminals
- Rename terminals: Right-click → "Rename"

---

## File Management

### Workspace Layout

```
DeepSeek-OCR/
├── .vscode/              # VSCode configuration (auto-created)
├── venv/                 # Virtual environment
├── output/               # Generated output files
├── example_macos_pdf_parse.py  # Main script
├── requirements_macos.txt      # Dependencies
└── QUICKSTART_MACOS.md        # Quick start guide
```

### Recommended .gitignore

Already configured to ignore:
- `venv/`
- `output/`
- `__pycache__/`
- `*.pyc`
- `.DS_Store`

---

## Debugging

### Setting Breakpoints

1. Click left of line number (red dot appears)
2. Run in debug mode (`F5`)
3. Execution pauses at breakpoint

### Debug Controls

- **Continue** (`F5`): Resume execution
- **Step Over** (`F10`): Execute current line
- **Step Into** (`F11`): Enter function
- **Step Out** (`Shift+F11`): Exit function
- **Restart** (`Cmd+Shift+F5`): Restart debugging

### Variables Panel

- Shows all variables in current scope
- Expand objects to see properties
- Right-click → "Copy Value"

### Debug Console

- Evaluate expressions during debugging
- Type variable name to see value
- Run functions to test

---

## Common Issues

### "No Python interpreter selected"

1. Press `Cmd+Shift+P`
2. Type "Python: Select Interpreter"
3. Choose `./venv/bin/python`

### "Module not found"

Virtual environment not activated:
```bash
source venv/bin/activate
pip install -r requirements_macos.txt
```

### Terminal shows wrong Python

1. Close all terminals
2. Open new terminal (`` Cmd+` ``)
3. Should auto-activate venv

### Slow IntelliSense

Large folders in workspace. Update `.vscode/settings.json`:
```json
"files.watcherExclude": {
  "**/venv/**": true,
  "**/output/**": true
}
```

---

## Tips for Working with DeepSeek-OCR

### Editing Configuration

1. Open `example_macos_pdf_parse.py`
2. Find configuration section at top
3. Edit paths:
```python
INPUT_FILE = '/Users/yourname/Documents/file.pdf'
OUTPUT_DIR = './output'
```

### Testing with Small Files

Start with 1-2 page PDFs to test setup before processing large documents.

### Monitoring Progress

Watch the integrated terminal for progress:
- Model download progress
- Page-by-page processing
- Completion status

### Viewing Results

1. Output appears in `./output/`
2. Click on `.md` files to preview in VSCode
3. Use Markdown Preview: `Cmd+Shift+V`

---

## Jupyter Notebooks (Optional)

If you prefer notebooks:

### Setup

```bash
pip install jupyter ipykernel
python -m ipykernel install --user --name=deepseek-ocr
```

### Create Notebook

1. Press `Cmd+Shift+P`
2. Type "Create: New Jupyter Notebook"
3. Select kernel: `deepseek-ocr`
4. Start coding in cells

### Run Cells

- Click ▶️ next to cell
- Or: `Shift+Enter` to run and move to next

---

## Performance Optimization

### Close Unused Files

- Too many open files slow down VSCode
- Close tabs: `Cmd+W`
- Close all: `Cmd+K W`

### Disable Unused Extensions

1. Click Extensions icon
2. Right-click extension → "Disable (Workspace)"

### Increase Memory (if needed)

Add to settings:
```json
"files.watcherExclude": {
  "**/venv/**": true,
  "**/.git/objects/**": true,
  "**/output/**": true
}
```

---

## Getting Help

### VSCode Documentation

- Press `Cmd+Shift+P`
- Type "Help: Getting Started"

### Python in VSCode

- [Official Guide](https://code.visualstudio.com/docs/python/python-tutorial)

### DeepSeek-OCR Help

- See `QUICKSTART_MACOS.md` for usage
- See `SETUP_MACOS.md` for troubleshooting

---

**Ready to start?** Open `example_macos_pdf_parse.py` and click the ▶️ button!
