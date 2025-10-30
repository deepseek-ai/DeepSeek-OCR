# DeepSeek OCR - Streamlit Application

Aplicação web completa para OCR (Reconhecimento Óptico de Caracteres) usando DeepSeek OCR com interface Streamlit.

## Funcionalidades

- 📤 **Upload de Imagens**: Suporte para JPG, PNG, JPEG
- 🎯 **Múltiplos Modos de Resolução**:
  - Tiny (512x512) - Rápido
  - Small (640x640) - Balanceado
  - Base (1024x1024) - Qualidade
  - Large (1280x1280) - Alta qualidade
  - Gundam (Dinâmico) - Adaptativo (RECOMENDADO)
  - Custom - Configuração manual
- 💬 **Prompts Predefinidos**:
  - Conversão para Markdown
  - OCR de imagem geral
  - Parse de figuras
  - Descrição detalhada
  - Localização de texto
- 🎨 **Visualização de Resultados**:
  - Texto puro
  - Markdown renderizado
  - Estatísticas e análise
- 💾 **Download de Resultados**: TXT e MD
- 🔧 **Configurações Avançadas**: Compressão de tokens, salvar resultados
- 📊 **Análise de Texto**: Estatísticas e métricas

## Instalação

### 1. Pré-requisitos

```bash
# Python 3.8 ou superior
python --version

# CUDA (opcional, mas recomendado para GPU)
nvidia-smi
```

### 2. Instalar Dependências

```bash
# Instalar dependências do Streamlit
pip install -r requirements_streamlit.txt

# OU instalar todas as dependências do projeto
pip install -r DeepSeek-OCR-master/DeepSeek-OCR-hf/requirements.txt
pip install streamlit
```

### 3. Baixar o Modelo

O modelo será baixado automaticamente do HuggingFace na primeira execução:
- Modelo: `deepseek-ai/DeepSeek-OCR`
- Tamanho: ~10GB
- Requer token HuggingFace se o modelo for privado

## Como Usar

### Executar a Aplicação

```bash
streamlit run streamlit_ocr_app.py
```

A aplicação abrirá automaticamente no navegador em `http://localhost:8501`

### Passo a Passo

1. **Configure o Modo de Resolução** (sidebar):
   - Escolha entre os modos predefinidos
   - Ou use "Custom" para configuração manual

2. **Faça Upload da Imagem**:
   - Clique em "Browse files"
   - Selecione uma imagem (JPG, PNG, JPEG)

3. **Escolha um Prompt**:
   - Selecione um prompt predefinido
   - Ou crie um prompt customizado

4. **Processar**:
   - Clique em "🚀 Processar Imagem com OCR"
   - Aguarde o processamento (pode levar alguns segundos)

5. **Visualizar Resultados**:
   - Veja o texto extraído em diferentes formatos
   - Analise estatísticas
   - Faça download dos resultados

## Modos de Resolução

| Modo | Base Size | Image Size | Crop Mode | Tokens | Uso |
|------|-----------|------------|-----------|--------|-----|
| **Tiny** | 512 | 512 | ❌ | 64 | Testes rápidos |
| **Small** | 640 | 640 | ❌ | 100 | Documentos simples |
| **Base** | 1024 | 1024 | ❌ | 256 | Uso geral |
| **Large** | 1280 | 1280 | ❌ | 400 | Alta qualidade |
| **Gundam** | 1024 | 640 | ✅ | Variável | Adaptativo (melhor) |

## Prompts Disponíveis

### 1. Documento para Markdown
```
<image>\n<|grounding|>Convert the document to markdown.
```
Converte documentos com layout complexo para Markdown, preservando estrutura.

### 2. OCR de Imagem Geral
```
<image>\n<|grounding|>OCR this image.
```
Extrai todo texto da imagem com informações de posicionamento.

### 3. OCR Livre
```
<image>\nFree OCR.
```
Extrai texto sem informações de layout.

### 4. Parse de Figura
```
<image>\nParse the figure.
```
Analisa e descreve figuras, gráficos e diagramas.

### 5. Descrição Detalhada
```
<image>\nDescribe this image in detail.
```
Gera descrição detalhada da imagem.

### 6. Localização de Texto
```
<image>\nLocate <|ref|>texto<|/ref|> in the image.
```
Localiza texto específico na imagem.

## Transforms Utilizados

A aplicação usa todos os transforms do DeepSeek OCR:

### ImageTransform
```python
# Normalização e conversão para tensor
mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)
```

### Dynamic Preprocessing
- Redimensionamento adaptativo
- Divisão em tiles (crop_mode)
- Preservação de aspect ratio

### Count Tiles
- Layout otimizado baseado em aspect ratio
- 2-6 tiles por dimensão (configurável)

### Aspect Ratio Matching
- Encontra melhor correspondência
- Minimiza distorção

## Pipeline de Processamento

```
1. Upload Imagem → PIL Image RGB
2. Dynamic Preprocessing → Resize + Crop
3. ImageTransform → Tensor + Normalize
4. Padding → Quadrado (base_size)
5. Feature Extraction → SAM ViT-B + CLIP-L
6. Projection → Embedding Space (2048→1280)
7. LLM Generation → DeepSeek V2/V3
8. Output → Texto Final
```

## Requisitos de Hardware

### Mínimo
- CPU: Intel i5 / AMD Ryzen 5
- RAM: 16GB
- Espaço: 20GB

### Recomendado
- CPU: Intel i7 / AMD Ryzen 7
- GPU: NVIDIA RTX 3060+ (12GB VRAM)
- RAM: 32GB
- Espaço: 30GB

### Ideal
- CPU: Intel i9 / AMD Ryzen 9
- GPU: NVIDIA RTX 4090 (24GB VRAM)
- RAM: 64GB
- Espaço: 50GB

## Resolução de Problemas

### Erro: "CUDA out of memory"
- Reduza o modo de resolução (use Tiny ou Small)
- Diminua max_crops se usar Gundam
- Ative test_compress
- Feche outros programas que usam GPU

### Erro: "Model not found"
- Verifique conexão com internet
- Configure token HuggingFace se necessário
- Tente baixar modelo manualmente

### Aplicação lenta
- Use GPU se disponível
- Reduza resolução
- Use modo Tiny para testes

### Erro ao carregar modelo
- Verifique versão do transformers: `pip install transformers>=4.46.3`
- Instale flash-attention-2 se usar GPU
- Verifique compatibilidade CUDA

## Exemplos de Uso

### Documentos Escaneados
- Modo: **Gundam** ou **Large**
- Prompt: "Convert the document to markdown"
- Resultado: Markdown estruturado com tabelas, listas, etc.

### Screenshots de Código
- Modo: **Base** ou **Large**
- Prompt: "OCR this image"
- Resultado: Código extraído com formatação

### Notas Manuscritas
- Modo: **Large** ou **Gundam**
- Prompt: "OCR this image"
- Resultado: Texto das notas manuscritas

### Tabelas Complexas
- Modo: **Gundam**
- Prompt: "Convert the document to markdown"
- Resultado: Tabelas em formato Markdown

### Figuras e Diagramas
- Modo: **Gundam**
- Prompt: "Parse the figure"
- Resultado: Descrição detalhada da figura

## Estrutura do Arquivo

```python
streamlit_ocr_app.py
├── Imports e Configuração
├── CSS Customizado
├── load_model() - Cache do modelo
├── RESOLUTION_MODES - Configurações de resolução
├── PREDEFINED_PROMPTS - Prompts predefinidos
├── Sidebar - Configurações
├── Upload de Imagem
├── Configuração de Prompt
├── Processamento
├── Visualização de Resultados
├── Documentação
└── Exemplos
```

## Tecnologias Utilizadas

- **Streamlit**: Framework web
- **DeepSeek OCR**: Modelo de OCR
- **Transformers**: HuggingFace library
- **PyTorch**: Deep learning
- **Pillow**: Processamento de imagem
- **SAM ViT-B**: Vision encoder
- **CLIP-L**: Vision-language model

## Recursos Adicionais

- [DeepSeek OCR HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- [Streamlit Documentation](https://docs.streamlit.io)
- [Transformers Documentation](https://huggingface.co/docs/transformers)

## Licença

Este projeto usa o modelo DeepSeek OCR. Verifique a licença do modelo no HuggingFace.

## Contribuindo

Contribuições são bem-vindas! Por favor, abra uma issue ou pull request.

## Suporte

Para problemas ou dúvidas:
1. Verifique a seção "Resolução de Problemas"
2. Consulte a documentação do DeepSeek OCR
3. Abra uma issue no repositório

---

**Desenvolvido com ❤️ usando DeepSeek OCR e Streamlit**
