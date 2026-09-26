import json
from pathlib import Path

from localvoicetranslator import config


def test_defaults_when_file_missing(tmp_path):
    assert config.load_config(tmp_path / "missing.json") == config.DEFAULTS


def test_unknown_keys_ignored(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"model_size": "small", "cli_path": "x.exe"}), encoding="utf-8")
    cfg = config.load_config(path)
    assert cfg["model_size"] == "small"
    assert "cli_path" not in cfg
    assert cfg["ollama_url"] == config.DEFAULTS["ollama_url"]


def test_corrupt_json_gives_defaults(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text("{broken", encoding="utf-8")
    assert config.load_config(path) == config.DEFAULTS


def test_save_and_load_roundtrip(tmp_path):
    path = tmp_path / "cfg.json"
    data = dict(config.DEFAULTS, vad=False, target_language="Spanish", ollama_model="llama3")
    config.save_config(data, path)
    assert config.load_config(path) == data


def test_save_to_unwritable_path_does_not_crash(tmp_path):
    config.save_config({"a": 1}, tmp_path / "no_dir" / "cfg.json")


def test_unknown_languages_fall_back_to_defaults(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"source_language": "Klingon", "target_language": "Elvish"}),
                    encoding="utf-8")
    cfg = config.load_config(path)
    assert cfg["source_language"] == config.DEFAULTS["source_language"]
    assert cfg["target_language"] == config.DEFAULTS["target_language"]


def test_language_lists():
    assert config.SOURCE_LANGUAGES[0] == config.AUTO_DETECT
    assert config.TARGET_LANGUAGES[-1] == config.NO_TRANSLATION
    assert config.AUTO_DETECT not in config.LANGUAGES
    assert config.NO_TRANSLATION not in config.LANGUAGES
    assert len(set(config.LANGUAGES.values())) == len(config.LANGUAGES)


def test_config_lives_in_project_root():
    assert config.CONFIG.parent == Path(__file__).resolve().parent.parent


def test_example_config_is_valid_and_complete():
    path = Path(__file__).resolve().parent.parent / "localvoicetranslator_config.example.json"
    assert json.loads(path.read_text(encoding="utf-8")) == config.DEFAULTS
