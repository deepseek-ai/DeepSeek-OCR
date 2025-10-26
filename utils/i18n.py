"""
Internationalization (i18n) support
Multi-language UI translations
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional


class I18n:
    """Internationalization manager"""

    def __init__(self, locale: str = 'en'):
        self.locale = locale
        self.translations = self._load_translations(locale)
        self.fallback_translations = self._load_translations('en') if locale != 'en' else {}

    def _load_translations(self, locale: str) -> Dict[str, Any]:
        """Load translations for a locale"""
        locale_file = Path(__file__).parent.parent / 'locales' / f'{locale}.json'

        if locale_file.exists():
            with open(locale_file, 'r', encoding='utf-8') as f:
                return json.load(f)

        # Return built-in translations if file doesn't exist
        return self._get_builtin_translations(locale)

    def _get_builtin_translations(self, locale: str) -> Dict[str, Any]:
        """Get built-in translations"""
        translations = {
            'en': TRANSLATIONS_EN,
            'es': TRANSLATIONS_ES,
            'zh': TRANSLATIONS_ZH,
            'fr': TRANSLATIONS_FR,
            'de': TRANSLATIONS_DE,
        }
        return translations.get(locale, TRANSLATIONS_EN)

    def t(self, key: str, **kwargs) -> str:
        """Translate a key with optional parameters"""
        # Navigate nested keys (e.g., 'app.title')
        keys = key.split('.')
        value = self.translations

        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                # Try fallback
                value = self.fallback_translations
                for k in keys:
                    if isinstance(value, dict) and k in value:
                        value = value[k]
                    else:
                        return key  # Return key if translation not found

        # Format with parameters
        if isinstance(value, str) and kwargs:
            try:
                return value.format(**kwargs)
            except KeyError:
                return value

        return value if isinstance(value, str) else key

    def get_locale(self) -> str:
        """Get current locale"""
        return self.locale

    def set_locale(self, locale: str):
        """Change locale"""
        self.locale = locale
        self.translations = self._load_translations(locale)
        self.fallback_translations = self._load_translations('en') if locale != 'en' else {}


# Built-in translations
TRANSLATIONS_EN = {
    "app": {
        "title": "DeepSeek-OCR Studio",
        "subtitle": "Extract information from presentations, PDFs, and documents with tables and graphics"
    },
    "sidebar": {
        "configuration": "Configuration",
        "model_settings": "Model Settings",
        "model_path": "Model Path",
        "resolution_mode": "Resolution Mode",
        "advanced_settings": "Advanced Settings",
        "prompt_template": "Prompt Template",
        "pdf_settings": "PDF Settings"
    },
    "tabs": {
        "upload": "Upload & Process",
        "results": "Results",
        "batch": "Batch Processing",
        "comparison": "Comparison",
        "editor": "Editor",
        "about": "About"
    },
    "upload": {
        "drag_drop": "Drag and drop files here or click to browse",
        "files_uploaded": "{count} file(s) uploaded successfully!",
        "process_button": "Process Files",
        "processing": "Processing...",
        "complete": "Processing complete!"
    },
    "results": {
        "select_file": "Select File",
        "select_page": "Select Page",
        "markdown_output": "Markdown Output",
        "visualized": "Visualized",
        "downloads": "Downloads",
        "raw_output": "Raw Output",
        "original_image": "Original Image",
        "with_bounding_boxes": "With Bounding Boxes",
        "download_markdown": "Download Markdown",
        "download_annotated": "Download Annotated",
        "download_raw": "Download Raw Text",
        "download_json": "Download JSON",
        "download_html": "Download HTML",
        "download_docx": "Download DOCX",
        "download_csv": "Download CSV/Excel"
    },
    "batch": {
        "create_job": "Create Batch Job",
        "job_name": "Job Name",
        "active_jobs": "Active Jobs",
        "completed_jobs": "Completed Jobs",
        "job_progress": "Progress",
        "cancel_job": "Cancel Job",
        "delete_job": "Delete Job",
        "view_results": "View Results"
    },
    "comparison": {
        "title": "OCR Comparison Tool",
        "select_modes": "Select modes to compare",
        "compare_button": "Compare",
        "side_by_side": "Side by Side Comparison"
    },
    "editor": {
        "title": "Interactive Editor",
        "edit_markdown": "Edit Markdown",
        "adjust_boxes": "Adjust Bounding Boxes",
        "save_changes": "Save Changes",
        "reprocess": "Re-process"
    },
    "post_processing": {
        "title": "Post-Processing",
        "enable_spellcheck": "Enable Spell Check",
        "enable_grammar": "Enable Grammar Check",
        "enable_table_validation": "Validate Tables",
        "enable_formula_check": "Validate Formulas",
        "issues_found": "{count} issue(s) found",
        "corrections_applied": "{count} correction(s) applied"
    },
    "messages": {
        "error": "Error",
        "warning": "Warning",
        "info": "Info",
        "success": "Success"
    }
}

TRANSLATIONS_ES = {
    "app": {
        "title": "DeepSeek-OCR Studio",
        "subtitle": "Extraer información de presentaciones, PDFs y documentos con tablas y gráficos"
    },
    "sidebar": {
        "configuration": "Configuración",
        "model_settings": "Configuración del Modelo",
        "model_path": "Ruta del Modelo",
        "resolution_mode": "Modo de Resolución",
        "advanced_settings": "Configuración Avanzada",
        "prompt_template": "Plantilla de Prompt",
        "pdf_settings": "Configuración PDF"
    },
    "tabs": {
        "upload": "Subir y Procesar",
        "results": "Resultados",
        "batch": "Procesamiento por Lotes",
        "comparison": "Comparación",
        "editor": "Editor",
        "about": "Acerca de"
    },
    "upload": {
        "drag_drop": "Arrastra y suelta archivos aquí o haz clic para explorar",
        "files_uploaded": "¡{count} archivo(s) subido(s) exitosamente!",
        "process_button": "Procesar Archivos",
        "processing": "Procesando...",
        "complete": "¡Procesamiento completo!"
    },
    "results": {
        "select_file": "Seleccionar Archivo",
        "select_page": "Seleccionar Página",
        "markdown_output": "Salida Markdown",
        "visualized": "Visualizado",
        "downloads": "Descargas",
        "raw_output": "Salida Sin Procesar",
        "original_image": "Imagen Original",
        "with_bounding_boxes": "Con Cajas Delimitadoras"
    },
    "messages": {
        "error": "Error",
        "warning": "Advertencia",
        "info": "Información",
        "success": "Éxito"
    }
}

TRANSLATIONS_ZH = {
    "app": {
        "title": "DeepSeek-OCR 工作室",
        "subtitle": "从演示文稿、PDF和包含表格和图形的文档中提取信息"
    },
    "sidebar": {
        "configuration": "配置",
        "model_settings": "模型设置",
        "model_path": "模型路径",
        "resolution_mode": "分辨率模式",
        "advanced_settings": "高级设置",
        "prompt_template": "提示模板",
        "pdf_settings": "PDF设置"
    },
    "tabs": {
        "upload": "上传和处理",
        "results": "结果",
        "batch": "批量处理",
        "comparison": "对比",
        "editor": "编辑器",
        "about": "关于"
    },
    "upload": {
        "drag_drop": "拖放文件到此处或点击浏览",
        "files_uploaded": "成功上传 {count} 个文件！",
        "process_button": "处理文件",
        "processing": "处理中...",
        "complete": "处理完成！"
    },
    "results": {
        "select_file": "选择文件",
        "select_page": "选择页面",
        "markdown_output": "Markdown 输出",
        "visualized": "可视化",
        "downloads": "下载",
        "raw_output": "原始输出",
        "original_image": "原始图像",
        "with_bounding_boxes": "带边界框"
    },
    "messages": {
        "error": "错误",
        "warning": "警告",
        "info": "信息",
        "success": "成功"
    }
}

TRANSLATIONS_FR = {
    "app": {
        "title": "DeepSeek-OCR Studio",
        "subtitle": "Extraire des informations de présentations, PDFs et documents avec tableaux et graphiques"
    },
    "sidebar": {
        "configuration": "Configuration",
        "model_settings": "Paramètres du Modèle",
        "model_path": "Chemin du Modèle",
        "resolution_mode": "Mode de Résolution",
        "advanced_settings": "Paramètres Avancés",
        "prompt_template": "Modèle de Prompt",
        "pdf_settings": "Paramètres PDF"
    },
    "tabs": {
        "upload": "Télécharger et Traiter",
        "results": "Résultats",
        "batch": "Traitement par Lots",
        "comparison": "Comparaison",
        "editor": "Éditeur",
        "about": "À propos"
    },
    "messages": {
        "error": "Erreur",
        "warning": "Avertissement",
        "info": "Information",
        "success": "Succès"
    }
}

TRANSLATIONS_DE = {
    "app": {
        "title": "DeepSeek-OCR Studio",
        "subtitle": "Informationen aus Präsentationen, PDFs und Dokumenten mit Tabellen und Grafiken extrahieren"
    },
    "sidebar": {
        "configuration": "Konfiguration",
        "model_settings": "Modelleinstellungen",
        "model_path": "Modellpfad",
        "resolution_mode": "Auflösungsmodus",
        "advanced_settings": "Erweiterte Einstellungen",
        "prompt_template": "Prompt-Vorlage",
        "pdf_settings": "PDF-Einstellungen"
    },
    "tabs": {
        "upload": "Hochladen & Verarbeiten",
        "results": "Ergebnisse",
        "batch": "Stapelverarbeitung",
        "comparison": "Vergleich",
        "editor": "Editor",
        "about": "Über"
    },
    "messages": {
        "error": "Fehler",
        "warning": "Warnung",
        "info": "Information",
        "success": "Erfolg"
    }
}
