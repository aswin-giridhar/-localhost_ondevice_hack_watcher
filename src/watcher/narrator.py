"""Narration: speak the agent's decisions (offline TTS) and/or show on-screen.

On-screen narration is always emitted via the websocket; speech is optional and
uses pyttsx3 (fully offline) when enabled and available.
"""
from __future__ import annotations

import threading


class Narrator:
    def __init__(self, enabled: bool = True, tts: bool = False) -> None:
        self.enabled = enabled
        self.tts = tts
        self._engine = None
        if tts:
            try:
                import pyttsx3

                self._engine = pyttsx3.init()
            except Exception:
                self._engine = None

    def say(self, text: str) -> None:
        if not self.enabled or not text:
            return
        if self._engine is not None:
            # Speak off-thread so the perception loop never blocks on audio.
            threading.Thread(target=self._speak, args=(text,), daemon=True).start()

    def _speak(self, text: str) -> None:
        try:
            self._engine.say(text)
            self._engine.runAndWait()
        except Exception:
            pass
