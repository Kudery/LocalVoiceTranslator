"""Processing flow, independent of tkinter so it can be tested.

Progress is reported through notify(kind, value) with kind in
{"status", "label", "orig", "trans"}; the GUI turns these into widget updates.
"""

from .config import AUTO_DETECT, LANGUAGES
from .translation import translate_ollama


def process(audio_path, source_language, target_language, transcriber, settings,
            ollama_url, ollama_model, notify):
    language = None if source_language == AUTO_DETECT else LANGUAGES[source_language]
    target_code = LANGUAGES.get(target_language)  # None = no translation

    notify("status", "Transcribing...")
    notify("label", f"Translation ({target_language}):" if target_code else "Translation:")
    text, info = transcriber.transcribe(audio_path, settings, language)
    notify("orig", text)

    if target_code and text:
        if info.language == target_code:
            # Source is already in the target language: skip the Ollama call.
            notify("trans", text)
        else:
            notify("status", f"Detected: {info.language} - translating via Ollama...")
            notify("trans", translate_ollama(text, target_language, ollama_url, ollama_model))

    notify("status", "Done.")
