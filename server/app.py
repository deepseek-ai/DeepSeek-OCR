import os
import shutil
import uuid
import asyncio
import zipfile
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from typing import Dict, Optional
import subprocess
import io

# Reuse the vLLM PDF pipeline by spawning the script with configured paths
from subprocess import run, CalledProcessError, Popen, PIPE

ROOT = Path(__file__).resolve().parents[1]
VLLM_DIR = ROOT / 'DeepSeek-OCR-master' / 'DeepSeek-OCR-vllm'

app = FastAPI(title="DeepSeek-OCR Service")

static_dir = ROOT / 'server' / 'static'
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Store active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


def write_config(input_path: Path, output_dir: Path, prompt: Optional[str] = None):
    cfg_path = VLLM_DIR / 'config.py'
    text = cfg_path.read_text(encoding='utf-8')
    
    def repl(key: str, value_literal: str, src: str) -> str:
        import re
        # Match the key and everything after = on that line
        pattern = rf"^{key}\s*=.*$"
        # Use a function to avoid any escaping issues with backslashes
        def replacer(match):
            return f"{key} = {value_literal}"
        return re.sub(pattern, replacer, src, flags=re.MULTILINE)

    # Use repr() to produce valid Python string literals with proper escape sequences
    updated = repl('INPUT_PATH', repr(str(input_path)), text)
    updated = repl('OUTPUT_PATH', repr(str(output_dir)), updated)
    if prompt:
        prompt_literal = repr(prompt)
        updated = repl('PROMPT', prompt_literal, updated)
    cfg_path.write_text(updated, encoding='utf-8')


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!doctype html>
<html lang="zh-CN">
<head>
    <meta charset='utf-8'>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DeepSeek-OCR - AI 文档识别服务</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 600px;
            width: 100%;
            padding: 40px;
            animation: fadeIn 0.5s ease;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        h1 {
            color: #333;
            margin-bottom: 10px;
            font-size: 28px;
            text-align: center;
        }
        .subtitle {
            color: #666;
            text-align: center;
            margin-bottom: 30px;
            font-size: 14px;
        }
        .upload-area {
            border: 3px dashed #ddd;
            border-radius: 15px;
            padding: 40px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-bottom: 20px;
        }
        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }
        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f3ff;
        }
        .upload-icon {
            font-size: 48px;
            margin-bottom: 10px;
        }
        input[type="file"] {
            display: none;
        }
        .file-name {
            margin-top: 15px;
            color: #667eea;
            font-weight: 500;
        }
        .prompt-group {
            margin-bottom: 20px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            color: #555;
            font-weight: 500;
        }
        input[type="text"] {
            width: 100%;
            padding: 12px;
            border: 2px solid #e0e0e0;
            border-radius: 10px;
            font-size: 14px;
            transition: border-color 0.3s;
        }
        input[type="text"]:focus {
            outline: none;
            border-color: #667eea;
        }
        .btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
        }
        .btn:disabled {
            background: #ccc;
            cursor: not-allowed;
            transform: none;
        }
        .progress-container {
            display: none;
            margin-top: 30px;
        }
        .progress-bar {
            width: 100%;
            height: 8px;
            background: #f0f0f0;
            border-radius: 10px;
            overflow: hidden;
            margin-bottom: 15px;
        }
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            width: 0%;
            transition: width 0.3s ease;
        }
        .status-text {
            color: #666;
            text-align: center;
            margin-bottom: 10px;
        }
        .log-container {
            background: #f8f9fa;
            border-radius: 10px;
            padding: 15px;
            max-height: 200px;
            overflow-y: auto;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            color: #333;
        }
        .log-line {
            margin-bottom: 5px;
            padding: 3px 0;
        }
        .results {
            display: none;
            margin-top: 30px;
            padding: 20px;
            background: #f0f9ff;
            border-radius: 15px;
            border-left: 4px solid #667eea;
        }
        .results h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .download-link {
            display: block;
            padding: 12px 20px;
            background: white;
            border: 2px solid #667eea;
            border-radius: 10px;
            color: #667eea;
            text-decoration: none;
            margin-bottom: 10px;
            transition: all 0.3s;
            text-align: center;
            font-weight: 500;
        }
        .download-link:hover {
            background: #667eea;
            color: white;
        }
        .download-link.primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            font-weight: 600;
        }
        .download-link.primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        }
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            border: 3px solid rgba(255,255,255,.3);
            border-radius: 50%;
            border-top-color: white;
            animation: spin 1s ease-in-out infinite;
        }
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 DeepSeek-OCR</h1>
        <p class="subtitle">AI 驱动的智能文档识别服务</p>
        
        <form id="uploadForm">
            <div class="upload-area" id="uploadArea">
                <div class="upload-icon">📄</div>
                <div>点击或拖拽 PDF 文件到此处</div>
                <div class="file-name" id="fileName"></div>
                <input type="file" id="fileInput" name="file" accept="application/pdf" required />
            </div>
            
            <div class="prompt-group">
                <label for="prompt">自定义 Prompt（可选）</label>
                <input type="text" id="prompt" name="prompt" 
                       placeholder="留空使用默认：<image>\\n<|grounding|>Convert the document to markdown." />
            </div>
            
            <button type="submit" class="btn" id="submitBtn">
                开始识别
            </button>
        </form>
        
        <div class="progress-container" id="progressContainer">
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill"></div>
            </div>
            <div class="status-text" id="statusText">准备中...</div>
            <div class="log-container" id="logContainer"></div>
        </div>
        
        <div class="results" id="results">
            <h3>✅ 识别完成！</h3>
            <a href="#" class="download-link" id="linkMmd" download>📝 下载 Markdown 文件</a>
            <a href="#" class="download-link" id="linkDetMmd" download>📋 下载完整标注文件</a>
            <a href="#" class="download-link" id="linkLayouts" download>🖼️ 下载可视化 PDF</a>
            <a href="#" class="download-link" id="linkImages" download>🎨 下载提取的图片 (ZIP)</a>
            <a href="#" class="download-link primary" id="linkAll" download>📦 下载全部文件 (ZIP)</a>
        </div>
    </div>
    
    <script>
        const uploadArea = document.getElementById('uploadArea');
        const fileInput = document.getElementById('fileInput');
        const fileName = document.getElementById('fileName');
        const uploadForm = document.getElementById('uploadForm');
        const submitBtn = document.getElementById('submitBtn');
        const progressContainer = document.getElementById('progressContainer');
        const progressFill = document.getElementById('progressFill');
        const statusText = document.getElementById('statusText');
        const logContainer = document.getElementById('logContainer');
        const results = document.getElementById('results');
        
        let selectedFile = null;
        let ws = null;
        
        // File upload area interactions
        uploadArea.addEventListener('click', () => fileInput.click());
        
        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0 && files[0].type === 'application/pdf') {
                fileInput.files = files;
                selectedFile = files[0];
                fileName.textContent = '已选择: ' + files[0].name;
            }
        });
        
        fileInput.addEventListener('change', (e) => {
            if (e.target.files.length > 0) {
                selectedFile = e.target.files[0];
                fileName.textContent = '已选择: ' + selectedFile.name;
            }
        });
        
        // Form submission
        uploadForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            if (!selectedFile) {
                alert('请先选择一个 PDF 文件');
                return;
            }
            
            // Show progress, hide results
            progressContainer.style.display = 'block';
            results.style.display = 'none';
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner"></span> 处理中...';
            progressFill.style.width = '10%';
            statusText.textContent = '上传文件中...';
            logContainer.innerHTML = '';
            
            const formData = new FormData();
            formData.append('file', selectedFile);
            formData.append('prompt', document.getElementById('prompt').value);
            
            try {
                const response = await fetch('/ocr/pdf', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    throw new Error('上传失败: ' + response.statusText);
                }
                
                const data = await response.json();
                
                if (data.error) {
                    throw new Error(data.error);
                }
                
                // Connect WebSocket for progress updates
                connectWebSocket(data.job_id, data);
                
            } catch (error) {
                statusText.textContent = '❌ 错误: ' + error.message;
                submitBtn.disabled = false;
                submitBtn.textContent = '开始识别';
            }
        });
        
        function connectWebSocket(jobId, resultData) {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(`${protocol}//${window.location.host}/ws/${jobId}`);
            
            ws.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    progressFill.style.width = data.progress + '%';
                    statusText.textContent = data.message;
                } else if (data.type === 'log') {
                    const logLine = document.createElement('div');
                    logLine.className = 'log-line';
                    logLine.textContent = data.message;
                    logContainer.appendChild(logLine);
                    logContainer.scrollTop = logContainer.scrollHeight;
                } else if (data.type === 'complete') {
                    progressFill.style.width = '100%';
                    statusText.textContent = '✅ 处理完成！';
                    
                    // Show download links
                    document.getElementById('linkMmd').href = resultData.mmd;
                    document.getElementById('linkDetMmd').href = resultData.det_mmd;
                    document.getElementById('linkLayouts').href = resultData.layouts;
                    document.getElementById('linkImages').href = resultData.images;
                    document.getElementById('linkAll').href = resultData.all;
                    results.style.display = 'block';
                    
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始识别';
                } else if (data.type === 'error') {
                    statusText.textContent = '❌ 错误: ' + data.message;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '开始识别';
                }
            };
            
            ws.onerror = () => {
                statusText.textContent = '⚠️ WebSocket 连接失败，但处理可能仍在继续...';
            };
            
            ws.onclose = () => {
                console.log('WebSocket closed');
            };
        }
    </script>
</body>
</html>
"""


@app.post("/ocr/pdf")
async def ocr_pdf(file: UploadFile = File(...), prompt: str = Form('')):
    if file.content_type not in ("application/pdf", "application/x-pdf", "application/acrobat") and not file.filename.lower().endswith('.pdf'):
        return JSONResponse({"error": "Please upload a PDF."}, status_code=400)

    job_id = uuid.uuid4().hex[:8]
    uploads_dir = ROOT / 'server' / 'uploads' / job_id
    output_dir = ROOT / 'server' / 'outputs' / job_id
    uploads_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf_path = uploads_dir / file.filename
    with open(pdf_path, 'wb') as f:
        shutil.copyfileobj(file.file, f)

    # update vLLM config
    prompt_str = prompt.strip() or "<image>\n<|grounding|>Convert the document to markdown."
    write_config(pdf_path, output_dir, prompt_str)

    # Start async task to run OCR
    asyncio.create_task(run_ocr_task(job_id, pdf_path, output_dir))

    # Return immediately with job info
    return {
        "job_id": job_id,
        "mmd": f"/download/{job_id}/mmd",
        "det_mmd": f"/download/{job_id}/det_mmd",
        "layouts": f"/download/{job_id}/layouts",
        "images": f"/download/{job_id}/images",
        "all": f"/download/{job_id}/all",
    }


async def run_ocr_task(job_id: str, pdf_path: Path, output_dir: Path):
    """Run OCR in background and send progress via WebSocket"""
    try:
        if job_id in active_connections:
            await send_progress(job_id, 15, "初始化模型...")
        
        env = os.environ.copy()
        env.setdefault('CUDA_VISIBLE_DEVICES', '0')
        
        # Run subprocess and capture output
        process = await asyncio.create_subprocess_exec(
            "python", str(VLLM_DIR / 'run_dpsk_ocr_pdf.py'),
            cwd=str(VLLM_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env
        )
        
        if job_id in active_connections:
            await send_progress(job_id, 30, "加载模型中...")
        
        # Read output line by line
        progress = 30
        async for line in process.stdout:
            line_text = line.decode('utf-8', errors='ignore').strip()
            if line_text:
                if job_id in active_connections:
                    await send_log(job_id, line_text)
                
                # Update progress based on output
                if 'Loading' in line_text or 'loading' in line_text:
                    progress = min(progress + 2, 50)
                    await send_progress(job_id, progress, "加载模型...")
                elif 'PDF loading' in line_text:
                    progress = 55
                    await send_progress(job_id, progress, "加载 PDF...")
                elif 'Pre-processed' in line_text:
                    progress = 65
                    await send_progress(job_id, progress, "预处理图像...")
                elif 'Processed prompts' in line_text or 'it/s' in line_text:
                    progress = min(progress + 5, 95)
                    await send_progress(job_id, progress, "识别中...")
        
        await process.wait()
        
        if process.returncode == 0:
            if job_id in active_connections:
                await send_complete(job_id)
        else:
            if job_id in active_connections:
                await send_error(job_id, f"OCR 处理失败，退出码: {process.returncode}")
    
    except Exception as e:
        if job_id in active_connections:
            await send_error(job_id, str(e))


async def send_progress(job_id: str, progress: int, message: str):
    """Send progress update via WebSocket"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "progress",
            "progress": progress,
            "message": message
        })


async def send_log(job_id: str, message: str):
    """Send log message via WebSocket"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "log",
            "message": message
        })


async def send_complete(job_id: str):
    """Send completion message via WebSocket"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "complete"
        })


async def send_error(job_id: str, error: str):
    """Send error message via WebSocket"""
    if job_id in active_connections:
        await active_connections[job_id].send_json({
            "type": "error",
            "message": error
        })


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    await websocket.accept()
    active_connections[job_id] = websocket
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        if job_id in active_connections:
            del active_connections[job_id]


@app.get('/download/{job_id}/mmd')
def download_mmd(job_id: str):
    output_dir = ROOT / 'server' / 'outputs' / job_id
    # pick the single .mmd file
    for p in output_dir.glob('*.mmd'):
        if not p.name.endswith('_det.mmd'):
            return FileResponse(str(p), filename=p.name, media_type='text/markdown')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/det_mmd')
def download_det_mmd(job_id: str):
    output_dir = ROOT / 'server' / 'outputs' / job_id
    for p in output_dir.glob('*_det.mmd'):
        return FileResponse(str(p), filename=p.name, media_type='text/markdown')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/layouts')
def download_layouts(job_id: str):
    output_dir = ROOT / 'server' / 'outputs' / job_id
    for p in output_dir.glob('*_layouts.pdf'):
        return FileResponse(str(p), filename=p.name, media_type='application/pdf')
    return JSONResponse({"error": "file not found"}, status_code=404)


@app.get('/download/{job_id}/images')
def download_images(job_id: str):
    """Download all extracted images as a zip file"""
    output_dir = ROOT / 'server' / 'outputs' / job_id
    images_dir = output_dir / 'images'
    
    if not images_dir.exists() or not any(images_dir.iterdir()):
        return JSONResponse({"error": "No images found"}, status_code=404)
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for img_file in images_dir.glob('*'):
            if img_file.is_file():
                zip_file.write(img_file, arcname=f'images/{img_file.name}')
    
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="images_{job_id}.zip"'}
    )


@app.get('/download/{job_id}/all')
def download_all(job_id: str):
    """Download all results (markdown files, images, and layout PDF) as a zip file"""
    output_dir = ROOT / 'server' / 'outputs' / job_id
    
    if not output_dir.exists():
        return JSONResponse({"error": "Job not found"}, status_code=404)
    
    # Create zip file in memory
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        # Add all files from output directory
        for file_path in output_dir.rglob('*'):
            if file_path.is_file():
                # Calculate relative path from output_dir
                rel_path = file_path.relative_to(output_dir)
                zip_file.write(file_path, arcname=str(rel_path))
    
    zip_buffer.seek(0)
    return StreamingResponse(
        zip_buffer,
        media_type='application/zip',
        headers={'Content-Disposition': f'attachment; filename="ocr_results_{job_id}.zip"'}
    )


