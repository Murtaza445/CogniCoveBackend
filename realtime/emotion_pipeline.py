"""Emotion analysis pipeline using real speech + facial emotion models."""

from __future__ import annotations

from typing import Any, Dict
import asyncio
import numpy as np
import cv2

# ===== IMPORT MODELS =====
from models_ai.speech_emotion.infer import predict_emotion_from_audio
from models_ai.facial_emotion.infer import FacialEmotionModel


# ===== LOAD MODELS (ONCE) =====
speech_model = predict_emotion_from_audio
face_model = FacialEmotionModel()


# ===== LABEL FIX (VERY IMPORTANT) =====
EMOTION_MAP = {
    "LABEL_0": "angry",
    "LABEL_1": "happy",
    "LABEL_2": "sad",
    "LABEL_3": "neutral",
    "LABEL_4": "fear",
    "LABEL_5": "disgust",
    "LABEL_6": "surprise"
}


# ===== SPEECH EMOTION =====
async def analyze_speech(audio_chunk: bytes) -> Dict[str, Any]:
    if not audio_chunk:
        return {"label": "neutral", "confidence": 0.0, "source": "speech"}

    try:
        result = await asyncio.to_thread(speech_model, audio_chunk)

        raw_label = result.get("label", "neutral")
        confidence = result.get("confidence", 0.0)

        # 🔥 MAP LABEL → REAL EMOTION
        mapped_label = EMOTION_MAP.get(raw_label, raw_label)

        return {
            "label": mapped_label,
            "confidence": round(confidence, 3),
            "source": "speech",
        }

    except Exception as e:
        print("❌ Speech Emotion Error:", e)
        return {"label": "neutral", "confidence": 0.0, "source": "speech"}


# ===== FACIAL EMOTION =====
async def analyze_face(frame: bytes) -> Dict[str, Any]:
    if not frame:
        return {"label": "neutral", "confidence": 0.0, "source": "face"}

    try:
        np_arr = np.frombuffer(frame, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            return {"label": "invalid_frame", "confidence": 0.0, "source": "face"}

        result = await asyncio.to_thread(face_model.predict, img)

        return {
            "label": result.get("emotion", "neutral"),
            "confidence": round(result.get("confidence", 0.0), 3),
            "source": "face",
        }

    except Exception as e:
        print("❌ Facial Emotion Error:", e)
        return {"label": "neutral", "confidence": 0.0, "source": "face"}