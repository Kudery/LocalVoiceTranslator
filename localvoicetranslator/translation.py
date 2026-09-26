"""Translation through the Ollama HTTP API (/api/generate)."""

import json
import urllib.error
import urllib.request

TIMEOUT = 300


def build_prompt(text, target_language):
    return (
        f"You are a professional translator. Translate the following text into "
        f"{target_language}. Return ONLY the translation: no explanations, no "
        f"quotation marks, no original text.\n\nText:\n{text}"
    )


def translate_ollama(text, target_language, ollama_url, ollama_model, timeout=TIMEOUT):
    """Return the translation, or a '[Translation failed: ...]' message (never raises)."""
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = json.dumps({
        "model": ollama_model,
        "prompt": build_prompt(text, target_language),
        "stream": False,
        "options": {"temperature": 0.2},
    }).encode("utf-8")
    req = urllib.request.Request(url, data=payload,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8"))
        return data.get("response", "").strip()
    except urllib.error.URLError as e:
        return f"[Translation failed: Ollama not reachable at {ollama_url} - {e}]"
    except Exception as e:
        return f"[Translation failed: {e}]"
