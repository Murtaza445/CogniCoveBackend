"""Session state primitives for realtime websocket interactions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class RealtimeSession:
    """Per-session state for realtime audio/text streaming."""

    session_id: str
    partial_text: str = ""
    full_transcript: List[str] = field(default_factory=list)
    emotion_history: List[Dict[str, Any]] = field(default_factory=list)
    is_ai_speaking: bool = False
    ai_responses: List[str] = field(default_factory=list)
    generation_task: Optional[asyncio.Task] = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def add_partial_text(self, text: str) -> str:
        """Append chunk text to partial transcript and return current value."""
        combined = f"{self.partial_text} {text}".strip() if self.partial_text else text.strip()
        self.partial_text = combined
        return self.partial_text

    def commit_partial_text(self) -> str:
        """Move current partial transcript into the full transcript list."""
        committed = self.partial_text.strip()
        if committed:
            self.full_transcript.append(committed)
        self.partial_text = ""
        return committed

    def add_emotion(self, emotion_payload: Dict[str, Any]) -> None:
        """Track emotion events captured while streaming."""
        payload = dict(emotion_payload)
        payload.setdefault("timestamp", datetime.utcnow().isoformat())
        self.emotion_history.append(payload)

    def add_ai_response(self, text: str) -> None:
        """Track assistant responses generated in realtime."""
        if text and text.strip():
            self.ai_responses.append(text.strip())

    def mark_ai_speaking(self, speaking: bool) -> None:
        """Set AI speaking state used for barge-in interruption logic."""
        self.is_ai_speaking = speaking

    def cancel_generation(self) -> None:
        """Cancel any active generation task for this session."""
        task = self.generation_task
        if task and not task.done():
            task.cancel()
