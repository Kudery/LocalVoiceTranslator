"""Constants, defaults and loading/saving of localvoicetranslator_config.json."""

import json
from pathlib import Path

# The config file lives in the project root, next to app.py.
CONFIG = Path(__file__).resolve().parent.parent / "localvoicetranslator_config.json"

# Display name -> Whisper language code
LANGUAGES = {
    "Arabic": "ar",
    "Chinese": "zh",
    "Czech": "cs",
    "Dutch": "nl",
    "English": "en",
    "French": "fr",
    "German": "de",
    "Greek": "el",
    "Hindi": "hi",
    "Hungarian": "hu",
    "Indonesian": "id",
    "Italian": "it",
    "Japanese": "ja",
    "Korean": "ko",
    "Polish": "pl",
    "Portuguese": "pt",
    "Romanian": "ro",
    "Russian": "ru",
    "Spanish": "es",
    "Swedish": "sv",
    "Turkish": "tr",
    "Ukrainian": "uk",
    "Vietnamese": "vi",
}

AUTO_DETECT = "Auto-detect"
SOURCE_LANGUAGES = [AUTO_DETECT] + list(LANGUAGES)

NO_TRANSLATION = "None (transcribe only)"
TARGET_LANGUAGES = list(LANGUAGES) + [NO_TRANSLATION]

MODEL_SIZES = ["tiny", "base", "small", "medium",
               "large-v2", "large-v3", "large-v3-turbo"]
DEVICES = ["auto", "cuda", "cpu"]
COMPUTE_TYPES = ["auto", "float16", "int8_float16", "int8", "float32"]

AUDIO_TYPES = [
    ("Audio files", "*.ogg *.opus *.mp3 *.wav *.m4a *.flac *.aac *.wma *.webm"),
    ("All files", "*.*"),
]

DEFAULTS = {
    "model_size": "large-v3-turbo",
    "device": "auto",
    "compute_type": "auto",
    "model_dir": "",  # empty = default Hugging Face cache (~/.cache/huggingface)
    "vad": True,
    "source_language": AUTO_DETECT,
    "target_language": "English",
    "ollama_url": "http://localhost:11434",
    "ollama_model": "qwen2.5:7b",
}


def load_config(path=CONFIG):
    cfg = dict(DEFAULTS)
    try:
        if path.exists():
            on_disk = json.loads(path.read_text(encoding="utf-8"))
            cfg.update({k: v for k, v in on_disk.items() if k in DEFAULTS})
    except Exception:
        pass
    # Unknown values from a hand-edited file would break the read-only comboboxes.
    if cfg["source_language"] not in SOURCE_LANGUAGES:
        cfg["source_language"] = DEFAULTS["source_language"]
    if cfg["target_language"] not in TARGET_LANGUAGES:
        cfg["target_language"] = DEFAULTS["target_language"]
    return cfg


def save_config(data, path=CONFIG):
    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception:
        pass
