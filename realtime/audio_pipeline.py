"""Realtime audio processing pipelines: STT (Vosk), LLM wrapper, and TTS (Piper)."""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import struct
import subprocess
import wave
import tempfile
from io import BytesIO
from typing import Any, Callable, List, Optional


# ===================== SILENCE FALLBACK =====================
def _silence_wav_bytes(duration_ms: int = 250, sample_rate: int = 22050) -> bytes:
    frame_count = int(sample_rate * (duration_ms / 1000.0))
    samples = struct.pack("<" + "h" * frame_count, *([0] * frame_count))

    buffer = BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(samples)
    return buffer.getvalue()


# ===================== STT =====================
class VoskStreamingSTT:
    def __init__(self) -> None:
        self.model_path = os.getenv("VOSK_MODEL_PATH", "")
        self.sample_rate = float(os.getenv("VOSK_SAMPLE_RATE", "16000"))

        self.enabled = False
        self._model = None

        try:
            vosk = importlib.import_module("vosk")
            Model = getattr(vosk, "Model")
            KaldiRecognizer = getattr(vosk, "KaldiRecognizer")

            if self.model_path:
                self._model = Model(self.model_path)
                self._recognizer_cls = KaldiRecognizer
                self.enabled = True

        except Exception as e:
            print("❌ VOSK INIT ERROR:", e)
            self.enabled = False

    async def process_audio_chunk(self, audio_chunk: bytes) -> str:
        if not self.enabled or not audio_chunk:
            return ""

        def _decode():
            rec = self._recognizer_cls(self._model, self.sample_rate)
            rec.AcceptWaveform(audio_chunk)

            try:
                res = json.loads(rec.Result())
                return res.get("text", "").strip()
            except:
                return ""

        return await asyncio.to_thread(_decode)


# ===================== LLM =====================
async def generate_realtime_response(
    text: str,
    session_id: str,
    emotion: Optional[str],
    therapy_chain_factory: Callable[[], Any],
    chat_history_builder: Callable[[str], List[Any]],
) -> str:

    if not text.strip():
        return ""

    # 🔥 Inject emotion context cleanly
    if isinstance(emotion, dict):
        emotion_text = f"User emotion: {emotion.get('speech', 'neutral')}"
    else:
        emotion_text = f"User emotion: {emotion or 'neutral'}"

    prompt = f"{emotion_text}\nUser: {text}"

    def _run():
        chain = therapy_chain_factory()
        history = chat_history_builder(session_id)

        resp = chain.invoke({
            "chat_history": history,
            "query": prompt
        })

        return getattr(resp, "content", "").strip()

    return await asyncio.to_thread(_run)


# ===================== TTS =====================
class PiperTTS:
    def __init__(self):
        self.piper_bin = os.getenv("PIPER_BIN")
        self.model_path = os.getenv("PIPER_MODEL")

        if not self.piper_bin or not self.model_path:
            raise ValueError("❌ PIPER_BIN or PIPER_MODEL not set")

    async def text_to_audio_bytes(self, text: str) -> bytes:
        if not text.strip():
            return _silence_wav_bytes()

        def _tts_sync():
            """Synchronous TTS processing (runs in thread pool)."""
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                output_path = tmp.name

            try:
                # Run Piper command synchronously (Windows-compatible)
                process = subprocess.run(
                    [self.piper_bin, "--model", self.model_path, "--output_file", output_path],
                    input=text.encode(),
                    capture_output=True,
                    timeout=30
                )

                stderr_msg = process.stderr.decode('utf-8', errors='ignore') if process.stderr else ""
                print("🔍 PIPER STDERR:", stderr_msg)

                if process.returncode != 0:
                    print(f"❌ Piper failed with return code {process.returncode}")
                    return None

                if not os.path.exists(output_path):
                    print("❌ Output file missing")
                    return None

                with open(output_path, "rb") as f:
                    audio = f.read()

                print("✅ AUDIO SIZE:", len(audio))
                os.remove(output_path)
                return audio

            except subprocess.TimeoutExpired:
                print("❌ Piper timeout (>30s)")
                return None
            except Exception as e:
                print(f"❌ Piper execution error: {str(e)}")
                if os.path.exists(output_path):
                    try:
                        os.remove(output_path)
                    except:
                        pass
                return None

        try:
            # Run synchronous TTS in thread pool (non-blocking)
            audio = await asyncio.to_thread(_tts_sync)

            if not audio:
                print("⚠️ Piper produced no audio")
                return _silence_wav_bytes()

            # 🚨 SAFETY CHECK
            if len(audio) < 1000:
                print("⚠️ Audio too small, returning silence")
                return _silence_wav_bytes()

            return audio

        except Exception as e:
            import traceback
            print("❌ TTS ERROR:", str(e))
            print("❌ TTS TRACEBACK:", traceback.format_exc())
            print(f"📋 PIPER_BIN: {self.piper_bin}")
            print(f"📋 PIPER_MODEL: {self.model_path}")
            print(f"📋 PIPER exists: {os.path.exists(self.piper_bin) if self.piper_bin else 'NOT SET'}")
            print(f"📋 MODEL exists: {os.path.exists(self.model_path) if self.model_path else 'NOT SET'}")
            return _silence_wav_bytes()