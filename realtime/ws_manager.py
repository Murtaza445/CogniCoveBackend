"""WebSocket manager for realtime audio streaming therapy sessions."""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from typing import Any, Callable, Dict, List

from realtime.audio_pipeline import PiperTTS, VoskStreamingSTT, generate_realtime_response
from realtime.emotion_pipeline import analyze_face, analyze_speech
from realtime.session_state import RealtimeSession


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("realtime")


class WebSocketSessionManager:
    """Coordinates realtime websocket session lifecycle and processing pipelines."""

    def __init__(
        self,
        get_or_create_session: Callable[[str], Dict[str, Any]],
        build_chat_history_from_session: Callable[[str], List[Any]],
        therapy_chain_factory: Callable[[], Any],
    ) -> None:
        self._get_or_create_session = get_or_create_session
        self._build_chat_history = build_chat_history_from_session
        self._therapy_chain_factory = therapy_chain_factory

        self._connections: Dict[str, Any] = {}
        self._realtime_sessions: Dict[str, RealtimeSession] = {}

        self._stt = VoskStreamingSTT()
        self._tts = PiperTTS()

        # Minimum words before triggering LLM unless end punctuation is detected.
        self._min_words = int(os.getenv("REALTIME_MIN_WORDS", "8"))

    async def connect(self, websocket: Any, session_id: str) -> None:
        """Accept websocket connection and initialize in-memory realtime state."""
        await websocket.accept()
        self._connections[session_id] = websocket
        self._realtime_sessions.setdefault(session_id, RealtimeSession(session_id=session_id))

        await websocket.send_json({
            "type": "session_connected",
            "session_id": session_id,
            "timestamp": datetime.utcnow().isoformat(),
        })

    async def handle_text_message(self, session_id: str, text: str) -> None:
        """Handle optional text control messages sent over websocket."""
        ws = self._connections.get(session_id)
        if not ws:
            return

        payload = (text or "").strip().lower()
        if payload == "ping":
            await ws.send_json({"type": "pong", "session_id": session_id})
            return

        if payload == "interrupt":
            session = self._realtime_sessions.get(session_id)
            if session:
                session.cancel_generation()
                session.mark_ai_speaking(False)
            await ws.send_json({"type": "interrupted", "session_id": session_id})
            return

        if payload == "commit":
            session = self._realtime_sessions.get(session_id)
            if not session:
                return
            committed_text = session.commit_partial_text()
            if committed_text:
                self._append_message(session_id, "user", committed_text)
                await self._start_generation(session_id, committed_text, emotion_label="neutral")

    async def handle_audio_chunk(self, session_id: str, audio_chunk: bytes) -> None:
        """Process one incoming audio chunk from client."""
        ws = self._connections.get(session_id)
        session = self._realtime_sessions.get(session_id)
        if not ws or not session:
            return

        if not audio_chunk:
            return

        logger.info(f"[AUDIO RECEIVED] bytes={len(audio_chunk)}")

        # Barge-in: if user starts speaking while AI response is being generated/played, cancel it.
        if session.is_ai_speaking:
            session.cancel_generation()
            session.mark_ai_speaking(False)
            await ws.send_json({
                "type": "interrupted",
                "reason": "new_user_audio_detected",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat(),
            })

        speech_emotion = await analyze_speech(audio_chunk)
        session.add_emotion(speech_emotion)
        logger.info(f"[SPEECH EMOTION] {speech_emotion}")

        recognized_text = (await self._stt.process_audio_chunk(audio_chunk)).strip()
        logger.info(f"[STT OUTPUT] {recognized_text}")
        if not recognized_text:
            return

        partial_text = session.add_partial_text(recognized_text)
        await ws.send_json({
            "type": "partial_transcript",
            "session_id": session_id,
            "text": partial_text,
            "timestamp": datetime.utcnow().isoformat(),
        })

        if not self._should_finalize_utterance(partial_text):
            return

        finalized_user_text = session.commit_partial_text()
        if not finalized_user_text:
            return

        self._append_message(session_id, "user", finalized_user_text)
        await ws.send_json({
            "type": "final_transcript",
            "session_id": session_id,
            "text": finalized_user_text,
            "timestamp": datetime.utcnow().isoformat(),
        })

        emotion_context = {
            "speech": speech_emotion.get("label", "neutral"),
        }

        await self._start_generation(
            session_id=session_id,
            user_text=finalized_user_text,
            emotion_label=emotion_context,
        )

    async def disconnect_and_persist(self, session_id: str) -> None:
        """Finalize session state and persist pending transcript segments to session store."""
        ws = self._connections.pop(session_id, None)
        session = self._realtime_sessions.pop(session_id, None)

        if session is None:
            if ws and ws.client_state.name != "DISCONNECTED":
                await ws.close()
            return

        session.cancel_generation()
        session.mark_ai_speaking(False)

        pending_text = session.commit_partial_text()
        if pending_text:
            self._append_message(session_id, "user", pending_text)

        if ws and ws.client_state.name != "DISCONNECTED":
            await ws.close()

    def _append_message(self, session_id: str, role: str, content: str) -> None:
        """Append message in the exact schema expected by REST summary/diagnosis routes."""
        if not content:
            return

        session = self._get_or_create_session(session_id)
        messages = session.setdefault("messages", [])
        messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        })

    def _should_finalize_utterance(self, partial_text: str) -> bool:
        """Heuristic endpointing for chunked transcription."""
        txt = (partial_text or "").strip()
        if not txt:
            return False

        if txt.endswith((".", "?", "!")):
            return True

        word_count = len(txt.split())
        return word_count >= self._min_words

    async def _start_generation(self, session_id: str, user_text: str, emotion_label: str) -> None:
        """Start/replace async generation task for session."""
        session = self._realtime_sessions.get(session_id)
        if not session:
            return

        session.cancel_generation()
        session.generation_task = asyncio.create_task(
            self._generate_and_send(session_id=session_id, user_text=user_text, emotion_label=emotion_label)
        )

    async def _generate_and_send(self, session_id: str, user_text: str, emotion_label: str) -> None:
        """Generate LLM reply, synthesize TTS audio, and stream both to client."""
        ws = self._connections.get(session_id)
        session = self._realtime_sessions.get(session_id)
        if not ws or not session:
            return

        try:
            session.mark_ai_speaking(True)

            emotion_text = ""
            if isinstance(emotion_label, dict):
                emotion_text = f"User emotion (speech): {emotion_label.get('speech', 'neutral')}"
            else:
                emotion_text = f"User emotion: {emotion_label}"

            enhanced_input = f"""
            {emotion_text}

            User said:
            {user_text}
            """

            logger.info(f"[LLM INPUT] {user_text}")

            response_text = await generate_realtime_response(
                text=enhanced_input,
                session_id=session_id,
                emotion=emotion_label,
                therapy_chain_factory=self._therapy_chain_factory,
                chat_history_builder=self._build_chat_history,
            )
            logger.info(f"[LLM OUTPUT] {response_text}")

            if not response_text:
                return

            session.add_ai_response(response_text)
            self._append_message(session_id, "assistant", response_text)

            await ws.send_json({
                "type": "assistant_text",
                "session_id": session_id,
                "text": response_text,
                "timestamp": datetime.utcnow().isoformat(),
            })

            audio_bytes = await self._tts.text_to_audio_bytes(response_text)
            logger.info(f"[TTS GENERATED] bytes={len(audio_bytes) if audio_bytes else 0}")
            if audio_bytes:
                await ws.send_bytes(audio_bytes)
                await ws.send_json({
                    "type": "assistant_audio",
                    "session_id": session_id,
                    "bytes": len(audio_bytes),
                    "timestamp": datetime.utcnow().isoformat(),
                })

        except asyncio.CancelledError:
            if ws:
                await ws.send_json({
                    "type": "generation_cancelled",
                    "session_id": session_id,
                    "timestamp": datetime.utcnow().isoformat(),
                })
            raise
        except Exception as ex:
            if ws:
                await ws.send_json({
                    "type": "error",
                    "session_id": session_id,
                    "detail": f"realtime_generation_error: {str(ex)}",
                    "timestamp": datetime.utcnow().isoformat(),
                })
        finally:
            if session:
                session.mark_ai_speaking(False)

    async def handle_face_frame(self, session_id: str, frame_bytes: bytes) -> None:
        """Optional helper to process facial frames if client sends them."""
        session = self._realtime_sessions.get(session_id)
        if not session:
            return

        face_emotion = await analyze_face(frame_bytes)
        logger.info(f"[FACE EMOTION] {face_emotion}")
        session.add_emotion(face_emotion)
