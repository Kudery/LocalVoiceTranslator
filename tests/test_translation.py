import io
import json
import urllib.error
from unittest import mock

from localvoicetranslator import translation


def _response(data):
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps(data).encode("utf-8")
    return resp


def test_request_and_response():
    with mock.patch("urllib.request.urlopen", return_value=_response({"response": "  Good day.\n"})) as up:
        out = translation.translate_ollama("Buna ziua.", "English", "http://host:11434/", "qwen2.5:7b")

    assert out == "Good day."
    req = up.call_args.args[0]
    assert up.call_args.kwargs["timeout"] == 300
    assert req.full_url == "http://host:11434/api/generate"
    assert req.get_header("Content-type") == "application/json"
    body = json.loads(req.data.decode("utf-8"))
    assert body["model"] == "qwen2.5:7b"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.2}
    assert body["prompt"].endswith("Text:\nBuna ziua.")
    assert "into English." in body["prompt"]


def test_missing_response_gives_empty_string():
    with mock.patch("urllib.request.urlopen", return_value=_response({"done": True})):
        assert translation.translate_ollama("x", "English", "http://h:1", "m") == ""


def test_ollama_unreachable():
    err = urllib.error.URLError("Connection refused")
    with mock.patch("urllib.request.urlopen", side_effect=err):
        out = translation.translate_ollama("x", "English", "http://localhost:11434", "m")
    assert out.startswith("[Translation failed: Ollama not reachable at http://localhost:11434 - ")
    assert "Connection refused" in out


def test_http_error_counts_as_unreachable():
    err = urllib.error.HTTPError("http://h/api/generate", 404, "model not found", {}, io.BytesIO())
    with mock.patch("urllib.request.urlopen", side_effect=err):
        out = translation.translate_ollama("x", "English", "http://h", "m")
    assert "Ollama not reachable" in out and "404" in out


def test_invalid_json():
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = b"not json"
    with mock.patch("urllib.request.urlopen", return_value=resp):
        out = translation.translate_ollama("x", "English", "http://h", "m")
    assert out.startswith("[Translation failed: ")
    assert "not reachable" not in out
