"""
DeepSeek OCR - Streamlit Application
Aplicação completa com upload de imagens e todos os recursos do DeepSeek OCR
"""

import streamlit as st
import torch
from transformers import AutoModel, AutoTokenizer
from PIL import Image
import os
import io
import json
from pathlib import Path
import tempfile
import time

# Configuração da página
st.set_page_config(
    page_title="DeepSeek OCR",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #424242;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #E3F2FD;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #F5F5F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 4px solid #1E88E5;
    }
    .stButton > button {
        width: 100%;
        background-color: #1E88E5;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Título principal
st.markdown('<h1 class="main-header">🔍 DeepSeek OCR Application</h1>', unsafe_allow_html=True)
st.markdown("### Upload de imagens e extração de texto com IA")

# Cache do modelo para não recarregar a cada interação
@st.cache_resource
def load_model():
    """Carrega o modelo DeepSeek OCR"""
    with st.spinner("🔄 Carregando modelo DeepSeek OCR... (isso pode levar alguns minutos)"):
        try:
            model_name = 'deepseek-ai/DeepSeek-OCR'

            # Verifica se CUDA está disponível
            device = "cuda" if torch.cuda.is_available() else "cpu"

            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            if device == "cuda":
                # Tenta carregar com flash_attention_2 primeiro
                try:
                    st.info("🔄 Tentando carregar modelo com Flash Attention 2...")
                    model = AutoModel.from_pretrained(
                        model_name,
                        _attn_implementation='flash_attention_2',
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    st.success("✅ Modelo carregado com Flash Attention 2!")
                except Exception as flash_error:
                    # Fallback: carrega sem flash_attention_2
                    st.warning(f"⚠️ Flash Attention 2 não disponível: {str(flash_error)}")
                    st.info("🔄 Carregando modelo com atenção padrão (eager)...")
                    model = AutoModel.from_pretrained(
                        model_name,
                        _attn_implementation='eager',
                        trust_remote_code=True,
                        use_safetensors=True
                    )
                    st.success("✅ Modelo carregado com atenção padrão!")

                model = model.eval().cuda().to(torch.bfloat16)
            else:
                st.info("🔄 Carregando modelo para CPU...")
                model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    use_safetensors=True
                )
                model = model.eval()
                st.success("✅ Modelo carregado para CPU!")

            return model, tokenizer, device

        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {str(e)}")
            st.error("💡 Dica: Instale flash-attn com: pip install flash-attn --no-build-isolation")
            return None, None, None

# Modos de resolução predefinidos
RESOLUTION_MODES = {
    "Tiny (512x512 - 64 tokens)": {
        "base_size": 512,
        "image_size": 512,
        "crop_mode": False,
        "description": "Mais rápido, menor qualidade. Ideal para testes."
    },
    "Small (640x640 - 100 tokens)": {
        "base_size": 640,
        "image_size": 640,
        "crop_mode": False,
        "description": "Rápido com qualidade razoável."
    },
    "Base (1024x1024 - 256 tokens)": {
        "base_size": 1024,
        "image_size": 1024,
        "crop_mode": False,
        "description": "Balanceado entre velocidade e qualidade."
    },
    "Large (1280x1280 - 400 tokens)": {
        "base_size": 1280,
        "image_size": 1280,
        "crop_mode": False,
        "description": "Alta qualidade, mais lento."
    },
    "Gundam (Dinâmico 1024 + 640)": {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "description": "Resolução dinâmica adaptativa (RECOMENDADO)."
    },
    "Custom (Personalizado)": {
        "base_size": 1024,
        "image_size": 640,
        "crop_mode": True,
        "description": "Configure manualmente os parâmetros."
    }
}

# Prompts predefinidos
PREDEFINED_PROMPTS = {
    "Documento para Markdown (com layout)": "<image>\n<|grounding|>Convert the document to markdown.",
    "OCR de Imagem Geral": "<image>\n<|grounding|>OCR this image.",
    "OCR Livre (sem layout)": "<image>\nFree OCR.",
    "Parse de Figura": "<image>\nParse the figure.",
    "Descrição Detalhada": "<image>\nDescribe this image in detail.",
    "Localização de Texto": "<image>\nLocate <|ref|>{text}<|/ref|> in the image.",
    "Custom (Personalizado)": ""
}

# Sidebar - Configurações
st.sidebar.header("⚙️ Configurações")

# Seleção de modo de resolução
st.sidebar.subheader("📐 Modo de Resolução")
resolution_mode = st.sidebar.selectbox(
    "Escolha o modo:",
    list(RESOLUTION_MODES.keys()),
    index=4  # Default: Gundam
)

# Mostra descrição do modo
st.sidebar.info(RESOLUTION_MODES[resolution_mode]["description"])

# Parâmetros configuráveis
if resolution_mode == "Custom (Personalizado)":
    st.sidebar.subheader("🔧 Parâmetros Customizados")

    base_size = st.sidebar.select_slider(
        "Base Size (resolução global):",
        options=[512, 640, 1024, 1280],
        value=1024
    )

    image_size = st.sidebar.select_slider(
        "Image Size (resolução local):",
        options=[512, 640, 1024, 1280],
        value=640
    )

    crop_mode = st.sidebar.checkbox(
        "Crop Mode (divisão dinâmica)",
        value=True
    )

    if crop_mode:
        min_crops = st.sidebar.slider("Min Crops:", 1, 9, 2)
        max_crops = st.sidebar.slider("Max Crops:", min_crops, 9, 6)
    else:
        min_crops, max_crops = 2, 6

else:
    base_size = RESOLUTION_MODES[resolution_mode]["base_size"]
    image_size = RESOLUTION_MODES[resolution_mode]["image_size"]
    crop_mode = RESOLUTION_MODES[resolution_mode]["crop_mode"]
    min_crops, max_crops = 2, 6

# Configurações adicionais
st.sidebar.subheader("🎛️ Configurações Avançadas")

test_compress = st.sidebar.checkbox(
    "Test Compress (compressão de tokens)",
    value=True,
    help="Ativa compressão de tokens de visão para economizar memória"
)

save_results = st.sidebar.checkbox(
    "Salvar Resultados",
    value=False,
    help="Salva os resultados em arquivos"
)

# Mostrar informações do sistema
st.sidebar.subheader("💻 Informações do Sistema")
device_info = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
st.sidebar.info(f"**Dispositivo:** {device_info}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    st.sidebar.info(f"**GPU:** {gpu_name}\n\n**Memória:** {gpu_memory:.1f} GB")

# Área principal
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown('<div class="sub-header">📤 Upload de Imagem</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Escolha uma imagem (JPG, PNG, JPEG):",
        type=["jpg", "jpeg", "png"],
        help="Faça upload de uma imagem para processar com OCR"
    )

    if uploaded_file is not None:
        # Exibe a imagem
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Imagem carregada", width='stretch')

        # Informações da imagem
        width, height = image.size
        st.info(f"**Dimensões:** {width}x{height} pixels | **Formato:** {image.format}")

with col2:
    st.markdown('<div class="sub-header">💬 Configuração do Prompt</div>', unsafe_allow_html=True)

    prompt_choice = st.selectbox(
        "Escolha um prompt predefinido:",
        list(PREDEFINED_PROMPTS.keys())
    )

    if prompt_choice == "Custom (Personalizado)":
        prompt_text = st.text_area(
            "Digite seu prompt personalizado:",
            value="<image>\n<|grounding|>Convert the document to markdown.",
            height=150,
            help="Use <image> para posição da imagem. Use <|grounding|> para incluir informações de layout."
        )
    elif prompt_choice == "Localização de Texto":
        search_text = st.text_input("Digite o texto a ser localizado:", value="texto")
        prompt_text = PREDEFINED_PROMPTS[prompt_choice].replace("{text}", search_text)
    else:
        prompt_text = PREDEFINED_PROMPTS[prompt_choice]
        st.code(prompt_text, language="text")

    # Informações sobre os prompts
    st.markdown("""
    <div class="info-box">
    <strong>💡 Dicas de Prompts:</strong>
    <ul>
        <li><code>&lt;image&gt;</code> - Posição da imagem no prompt</li>
        <li><code>&lt;|grounding|&gt;</code> - Inclui informações de layout e posicionamento</li>
        <li><code>&lt;|ref|&gt;texto&lt;|/ref|&gt;</code> - Localiza texto específico na imagem</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

# Botão de processamento
st.markdown("---")
process_button = st.button("🚀 Processar Imagem com OCR", width='stretch')

if process_button and uploaded_file is not None:

    # Carrega o modelo
    model, tokenizer, device = load_model()

    if model is None or tokenizer is None:
        st.error("❌ Não foi possível carregar o modelo. Verifique a instalação.")
        st.stop()

    # Cria diretório temporário para resultados
    with tempfile.TemporaryDirectory() as temp_dir:

        # Salva imagem temporária
        temp_image_path = os.path.join(temp_dir, "temp_image.jpg")
        image.save(temp_image_path)

        # Mostra configurações sendo usadas
        with st.expander("🔍 Ver Configurações de Processamento", expanded=False):
            config_info = {
                "Modo": resolution_mode,
                "Base Size": base_size,
                "Image Size": image_size,
                "Crop Mode": crop_mode,
                "Min Crops": min_crops if crop_mode else "N/A",
                "Max Crops": max_crops if crop_mode else "N/A",
                "Test Compress": test_compress,
                "Prompt": prompt_text,
                "Device": device_info
            }
            st.json(config_info)

        # Processamento
        try:
            with st.spinner("🔄 Processando imagem com DeepSeek OCR..."):
                start_time = time.time()

                # Chama o modelo
                result = model.infer(
                    tokenizer=tokenizer,
                    prompt=prompt_text,
                    image_file=temp_image_path,
                    output_path=temp_dir,
                    base_size=base_size,
                    image_size=image_size,
                    crop_mode=crop_mode,
                    test_compress=test_compress,
                    save_results=save_results
                )

                end_time = time.time()
                processing_time = end_time - start_time

            # Mostra resultados
            st.success(f"✅ Processamento concluído em {processing_time:.2f} segundos!")

            # Resultado principal
            st.markdown('<div class="sub-header">📝 Resultado do OCR</div>', unsafe_allow_html=True)

            # Cria tabs para diferentes visualizações
            tab1, tab2, tab3 = st.tabs(["📄 Texto", "📋 Markdown", "🔢 Estatísticas"])

            with tab1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.text_area(
                    "Texto extraído:",
                    value=result,
                    height=400,
                    label_visibility="collapsed"
                )
                st.markdown('</div>', unsafe_allow_html=True)

                # Botão de download
                st.download_button(
                    label="💾 Download Texto (.txt)",
                    data=result,
                    file_name="ocr_result.txt",
                    mime="text/plain"
                )

            with tab2:
                st.markdown(result)

                # Botão de download markdown
                st.download_button(
                    label="💾 Download Markdown (.md)",
                    data=result,
                    file_name="ocr_result.md",
                    mime="text/markdown"
                )

            with tab3:
                # Estatísticas do resultado
                stats = {
                    "Total de caracteres": len(result),
                    "Total de palavras": len(result.split()),
                    "Total de linhas": len(result.split('\n')),
                    "Tempo de processamento": f"{processing_time:.2f}s",
                    "Caracteres/segundo": f"{len(result)/processing_time:.0f}"
                }

                # Mostra estatísticas em colunas
                stat_cols = st.columns(3)
                for idx, (key, value) in enumerate(stats.items()):
                    with stat_cols[idx % 3]:
                        st.metric(key, value)

                # Análise do texto
                st.subheader("📊 Análise do Texto")

                # Contagem de tipos de caracteres
                num_digits = sum(c.isdigit() for c in result)
                num_alpha = sum(c.isalpha() for c in result)
                num_spaces = sum(c.isspace() for c in result)
                num_special = len(result) - num_digits - num_alpha - num_spaces

                analysis_data = {
                    "Letras": num_alpha,
                    "Números": num_digits,
                    "Espaços": num_spaces,
                    "Especiais": num_special
                }

                st.bar_chart(analysis_data)

            # Se salvar resultados estiver ativado, mostra informação
            if save_results:
                st.info(f"📁 Resultados salvos em: {temp_dir}")

        except Exception as e:
            st.error(f"❌ Erro durante o processamento: {str(e)}")
            st.exception(e)

elif process_button and uploaded_file is None:
    st.warning("⚠️ Por favor, faça upload de uma imagem primeiro!")

# Rodapé com informações
st.markdown("---")
st.markdown("""
### 📚 Sobre o DeepSeek OCR

DeepSeek OCR é um modelo avançado de reconhecimento óptico de caracteres que combina:
- **SAM ViT-B**: Segment Anything Model para features locais
- **CLIP-L**: Semantic features globais
- **DeepSeek V2/V3**: LLM backbone para compreensão contextual

**Recursos:**
- ✅ Multi-resolução adaptativa
- ✅ Suporte a layouts complexos
- ✅ Extração de tabelas e figuras
- ✅ Conversão para Markdown
- ✅ Grounding e localização de texto

**Transformações de Imagem:**
- `ImageTransform`: Normalização e conversão para tensor
- `dynamic_preprocess`: Redimensionamento e divisão em tiles
- `count_tiles`: Layout otimizado baseado em aspect ratio

---
🔗 **Modelo:** [deepseek-ai/DeepSeek-OCR](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
""")

# Exemplos de uso
with st.expander("💡 Exemplos de Uso"):
    st.markdown("""
    ### Casos de Uso Comuns

    **1. Documentos Escaneados:**
    - Modo: Gundam ou Large
    - Prompt: "Convert the document to markdown"

    **2. Screenshots de Código:**
    - Modo: Base ou Large
    - Prompt: "OCR this image"

    **3. Tabelas Complexas:**
    - Modo: Gundam
    - Prompt: "Convert the document to markdown"

    **4. Imagens com Texto Pequeno:**
    - Modo: Large
    - Prompt: "OCR this image"

    **5. Figuras e Diagramas:**
    - Modo: Gundam
    - Prompt: "Parse the figure"
    """)

# Informações técnicas
with st.expander("🔧 Informações Técnicas sobre Transforms"):
    st.markdown("""
    ### Transforms Utilizados no DeepSeek OCR

    **1. ImageTransform**
    ```python
    class ImageTransform:
        def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), normalize=True):
            # Aplica ToTensor() e Normalize
            # Converte PIL Image para tensor PyTorch
            # Normaliza valores de pixel para [-1, 1]
    ```

    **2. Dynamic Preprocessing**
    ```python
    def dynamic_preprocess(image, base_size, image_size, crop_mode):
        # Redimensiona imagem mantendo aspect ratio
        # Divide em tiles se crop_mode=True
        # Retorna global view + local views
    ```

    **3. Count Tiles**
    ```python
    def count_tiles(h, w, min_num, max_num):
        # Calcula grid layout otimizado
        # Baseado no aspect ratio da imagem
        # Retorna número de tiles em cada dimensão
    ```

    **4. Aspect Ratio Matching**
    ```python
    def find_closest_aspect_ratio(aspect_ratio, target_ratios):
        # Encontra melhor correspondência de aspect ratio
        # Usado para dynamic cropping
    ```

    **Pipeline Completo:**
    1. Load image → PIL Image RGB
    2. Dynamic preprocessing → Resize + Crop
    3. ImageTransform → Tensor + Normalize
    4. Pad → Quadrado com tamanho alvo
    5. Feature extraction → SAM + CLIP
    6. Projection → Embedding space
    7. LLM generation → Texto final
    """)
