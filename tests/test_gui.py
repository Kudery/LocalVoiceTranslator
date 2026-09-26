"""GUI smoke tests; skipped without tkinter or without a display."""

import json
import time
from unittest import mock

import pytest

tk = pytest.importorskip("tkinter")


@pytest.fixture
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def gui(monkeypatch):
    from localvoicetranslator import gui
    from localvoicetranslator.config import DEFAULTS

    monkeypatch.setattr(gui, "load_config", lambda: dict(DEFAULTS))
    monkeypatch.setattr(gui, "save_config", lambda data: None)
    return gui


def test_gui_builds(root, gui):
    app = gui.LocalVoiceTranslator(root)
    assert app.start_btn.cget("text") == "Transcribe & Translate"
    assert app.target_var.get() == "English"
    assert app.source_var.get() == "Auto-detect"


def test_start_processes_file_end_to_end(root, gui, monkeypatch, tmp_path, whisper_model):
    audio = tmp_path / "message.ogg"
    audio.write_bytes(b"")
    resp = mock.MagicMock()
    resp.__enter__.return_value.read.return_value = json.dumps({"response": "Good day."}).encode()
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: resp)

    app = gui.LocalVoiceTranslator(root)
    app.audio_file = str(audio)
    app._start()
    deadline = time.time() + 5
    while app._busy and time.time() < deadline:
        root.update()
        time.sleep(0.02)

    assert not app._busy
    assert app.txt_orig.get("1.0", "end").strip() == "Buna ziua."
    assert app.txt_trans.get("1.0", "end").strip() == "Good day."
    assert app.trans_label.get() == "Translation (English):"
    assert app.status_var.get() == "Done."
