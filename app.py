#!/usr/bin/env python3
"""LocalVoiceTranslator - entry point.

Transcribes voice messages and other audio locally with faster-whisper
(GPU or CPU) and translates the text with a local or remote Ollama model.
Settings are stored in localvoicetranslator_config.json next to this script.
"""

import tkinter as tk

from localvoicetranslator.gui import LocalVoiceTranslator


def main():
    root = tk.Tk()
    LocalVoiceTranslator(root)
    root.mainloop()


if __name__ == "__main__":
    main()
