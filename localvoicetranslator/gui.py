"""Tkinter GUI. The heavy lifting runs in a background thread."""

import queue
import threading
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from .config import (AUDIO_TYPES, COMPUTE_TYPES, DEVICES, MODEL_SIZES,
                     SOURCE_LANGUAGES, TARGET_LANGUAGES, load_config, save_config)
from .pipeline import process
from .transcription import Transcriber, TranscriptionSettings


class LocalVoiceTranslator:
    def __init__(self, root):
        self.root = root
        self.root.title("LocalVoiceTranslator")
        self.root.geometry("860x760")
        self.root.minsize(740, 620)

        self._busy = False
        self._q = queue.Queue()
        self.audio_file = None
        self.cfg = load_config()

        # Model cache + CPU fallback live in the Transcriber for the whole session
        self.transcriber = Transcriber(notify=self._notify)

        self._build_ui()
        self.root.after(100, self._process_queue)

    # -- Config --------------------------------------------------------------
    def _save_config(self):
        save_config({
            "model_size": self.model_var.get(),
            "device": self.device_var.get(),
            "compute_type": self.compute_var.get(),
            "model_dir": self.model_dir_var.get(),
            "vad": bool(self.vad_var.get()),
            "source_language": self.source_var.get(),
            "target_language": self.target_var.get(),
            "ollama_url": self.ollama_url_var.get(),
            "ollama_model": self.ollama_model_var.get(),
        })

    # -- UI ------------------------------------------------------------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}

        row1 = ttk.Frame(self.root); row1.pack(fill="x", **pad)
        ttk.Button(row1, text="Choose file...", command=self._choose_file).pack(side="left")
        self.file_var = tk.StringVar(value="No file selected")
        ttk.Label(row1, textvariable=self.file_var, foreground="#555").pack(side="left", padx=10)

        opts = ttk.LabelFrame(self.root, text="Settings")
        opts.pack(fill="x", **pad)

        # Model / device / precision
        r = ttk.Frame(opts); r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Whisper model:", width=15).pack(side="left")
        self.model_var = tk.StringVar(value=self.cfg["model_size"])
        ttk.Combobox(r, textvariable=self.model_var, values=MODEL_SIZES,
                     width=18).pack(side="left", padx=4)
        ttk.Label(r, text="Device:").pack(side="left", padx=(12, 0))
        self.device_var = tk.StringVar(value=self.cfg["device"])
        ttk.Combobox(r, textvariable=self.device_var, values=DEVICES,
                     state="readonly", width=8).pack(side="left", padx=4)
        ttk.Label(r, text="Precision:").pack(side="left", padx=(12, 0))
        self.compute_var = tk.StringVar(value=self.cfg["compute_type"])
        ttk.Combobox(r, textvariable=self.compute_var, values=COMPUTE_TYPES,
                     state="readonly", width=13).pack(side="left", padx=4)

        # Model folder (optional) + VAD
        r = ttk.Frame(opts); r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Model folder:", width=15).pack(side="left")
        self.model_dir_var = tk.StringVar(value=self.cfg["model_dir"])
        ttk.Entry(r, textvariable=self.model_dir_var).pack(side="left", fill="x", expand=True, padx=4)
        ttk.Button(r, text="...", width=3, command=self._choose_dir).pack(side="left")
        self.vad_var = tk.BooleanVar(value=bool(self.cfg["vad"]))
        ttk.Checkbutton(r, text="Skip silence (VAD)", variable=self.vad_var).pack(side="left", padx=(12, 0))

        # Source + target language
        r = ttk.Frame(opts); r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Source language:", width=15).pack(side="left")
        self.source_var = tk.StringVar(value=self.cfg["source_language"])
        ttk.Combobox(r, textvariable=self.source_var, values=SOURCE_LANGUAGES,
                     state="readonly", width=22).pack(side="left", padx=4)
        ttk.Label(r, text="Translate to:").pack(side="left", padx=(16, 0))
        self.target_var = tk.StringVar(value=self.cfg["target_language"])
        ttk.Combobox(r, textvariable=self.target_var, values=TARGET_LANGUAGES,
                     state="readonly", width=22).pack(side="left", padx=4)

        # Ollama
        r = ttk.Frame(opts); r.pack(fill="x", padx=6, pady=3)
        ttk.Label(r, text="Ollama URL:", width=15).pack(side="left")
        self.ollama_url_var = tk.StringVar(value=self.cfg["ollama_url"])
        ttk.Entry(r, textvariable=self.ollama_url_var, width=30).pack(side="left", padx=4)
        ttk.Label(r, text="Ollama model:").pack(side="left", padx=(12, 0))
        self.ollama_model_var = tk.StringVar(value=self.cfg["ollama_model"])
        ttk.Entry(r, textvariable=self.ollama_model_var, width=20).pack(side="left", padx=4)

        self.start_btn = ttk.Button(self.root, text="Transcribe & Translate", command=self._start)
        self.start_btn.pack(pady=6)

        statusbar = ttk.Frame(self.root); statusbar.pack(fill="x", padx=8)
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(statusbar, textvariable=self.status_var, foreground="#0a6").pack(side="left")
        self.progress = ttk.Progressbar(statusbar, mode="indeterminate", length=160)
        self.progress.pack(side="right")

        out = ttk.Frame(self.root); out.pack(fill="both", expand=True, **pad)
        ttk.Label(out, text="Original transcript:").pack(anchor="w")
        self.txt_orig = tk.Text(out, height=8, wrap="word"); self.txt_orig.pack(fill="both", expand=True, pady=(0, 6))
        self.trans_label = tk.StringVar(value="Translation:")
        ttk.Label(out, textvariable=self.trans_label).pack(anchor="w")
        self.txt_trans = tk.Text(out, height=8, wrap="word"); self.txt_trans.pack(fill="both", expand=True)

        buttons = ttk.Frame(self.root); buttons.pack(fill="x", **pad)
        ttk.Button(buttons, text="Copy original",
                   command=lambda: self._copy(self.txt_orig)).pack(side="left")
        ttk.Button(buttons, text="Copy translation",
                   command=lambda: self._copy(self.txt_trans)).pack(side="left", padx=6)
        ttk.Button(buttons, text="Save...", command=self._save).pack(side="right")

    # -- Actions -------------------------------------------------------------
    def _choose_file(self):
        path = filedialog.askopenfilename(title="Choose audio file", filetypes=AUDIO_TYPES)
        if path:
            self.audio_file = path
            self.file_var.set(Path(path).name)

    def _choose_dir(self):
        path = filedialog.askdirectory(title="Choose model cache folder")
        if path:
            self.model_dir_var.set(path)

    def _start(self):
        if self._busy:
            return
        if not self.audio_file:
            messagebox.showwarning("No file", "Choose an audio file first.")
            return
        if not Path(self.audio_file).exists():
            messagebox.showerror("Error", f"File not found:\n{self.audio_file}")
            return
        self._save_config()
        self._busy = True
        self.start_btn.config(state="disabled")
        self.progress.start(12)
        self.txt_orig.delete("1.0", "end"); self.txt_trans.delete("1.0", "end")
        # Read tk variables here (main thread), not in the worker
        job = dict(
            audio_path=self.audio_file,
            source_language=self.source_var.get(),
            target_language=self.target_var.get(),
            settings=TranscriptionSettings(
                model_size=self.model_var.get(),
                device=self.device_var.get(),
                compute_type=self.compute_var.get(),
                model_dir=self.model_dir_var.get(),
                vad=bool(self.vad_var.get()),
            ),
            ollama_url=self.ollama_url_var.get(),
            ollama_model=self.ollama_model_var.get(),
        )
        threading.Thread(target=self._worker, args=(job,), daemon=True).start()

    def _worker(self, job):
        try:
            process(transcriber=self.transcriber, notify=self._notify, **job)
        except Exception as e:
            self._notify("error", str(e))
        finally:
            self._notify("end", "")

    # -- Thread -> GUI -------------------------------------------------------
    def _notify(self, kind, value):
        self._q.put((kind, value))

    def _process_queue(self):
        try:
            while True:
                kind, value = self._q.get_nowait()
                if kind == "status":
                    self.status_var.set(value)
                elif kind == "label":
                    self.trans_label.set(value)
                elif kind == "orig":
                    self.txt_orig.delete("1.0", "end"); self.txt_orig.insert("1.0", value)
                elif kind == "trans":
                    self.txt_trans.delete("1.0", "end"); self.txt_trans.insert("1.0", value)
                elif kind == "error":
                    self.status_var.set("Error."); messagebox.showerror("Error", value)
                elif kind == "end":
                    self._busy = False
                    self.start_btn.config(state="normal")
                    self.progress.stop()
        except queue.Empty:
            pass
        self.root.after(100, self._process_queue)

    # -- Helpers -------------------------------------------------------------
    def _copy(self, widget):
        self.root.clipboard_clear()
        self.root.clipboard_append(widget.get("1.0", "end").strip())

    def _save(self):
        path = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Text file", "*.txt")],
                                            initialfile="translation.txt")
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write("=== ORIGINAL ===\n" + self.txt_orig.get("1.0", "end").strip() + "\n\n")
            f.write("=== TRANSLATION ===\n" + self.txt_trans.get("1.0", "end").strip() + "\n")
        self.status_var.set(f"Saved: {Path(path).name}")
