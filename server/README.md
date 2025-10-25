# DeepSeek-OCR Web 服务

美观的 Web 界面，支持实时进度显示和文件下载。

## 功能特性

- 🎨 **现代化 UI** - 渐变色背景、流畅动画、响应式设计
- 📤 **拖拽上传** - 支持点击和拖拽两种上传方式
- 📊 **实时进度** - WebSocket 实时推送处理进度和日志
- 📥 **便捷下载** - 处理完成后自动展示下载链接
- 🌐 **跨平台访问** - 可从局域网任何设备访问

## 启动服务

```bash
cd /home/hawk/ext_data/workspace/ocr/DeepSeek-OCR
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

## 访问地址

- **本地访问**: http://localhost:8000
- **局域网访问**: http://<Ubuntu机器IP>:8000

查看 Ubuntu 机器 IP：
```bash
hostname -I
# 或
ip addr show | grep "inet "
```

## 使用流程

1. 打开浏览器访问服务地址
2. 点击或拖拽 PDF 文件到上传区域
3. （可选）自定义 Prompt
4. 点击"开始识别"
5. 实时查看处理进度和日志
6. 完成后点击下载链接获取结果

## 输出文件

- **Markdown 文件** (`.mmd`) - 清洗后的 Markdown 文本
- **完整标注文件** (`_det.mmd`) - 包含检测标记的原始输出
- **可视化 PDF** (`_layouts.pdf`) - 带有检测框的可视化文档
- **提取的图片** (`images/*.jpg`) - 文档中提取的所有图片和图表（ZIP 格式）
- **全部文件** - 包含以上所有文件的完整压缩包（ZIP 格式）

## 技术栈

- **后端**: FastAPI + WebSocket
- **前端**: 原生 HTML/CSS/JavaScript
- **OCR**: DeepSeek-OCR + vLLM

## 注意事项

- 首次运行会加载模型，需要等待约 10-20 秒
- 确保 CUDA 可用且有足够显存
- 默认使用 GPU 0，可通过环境变量 `CUDA_VISIBLE_DEVICES` 修改

