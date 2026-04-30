# models/speech_emotion/infer.py

import torch
import librosa
import numpy as np
from transformers import AutoProcessor, AutoModelForAudioClassification

MODEL_DIR = "models/speech_emotion"

# Load once (global)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("🔊 Loading Speech Emotion Model...")

processor = AutoProcessor.from_pretrained(MODEL_DIR)
model = AutoModelForAudioClassification.from_pretrained(MODEL_DIR).to(device)

model.eval()

# Emotion labels (VERY IMPORTANT — adjust if needed)
id2label = model.config.id2label

print("✅ Speech Emotion Model Loaded!")


def predict_emotion_from_audio(audio_bytes: bytes):
    try:
        # Convert bytes → waveform
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # Resample to 16k (required by most models)
        audio_np = librosa.resample(audio_np, orig_sr=16000, target_sr=16000)

        inputs = processor(
            audio_np,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        ).to(device)

        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.nn.functional.softmax(logits, dim=-1)

            confidence, predicted = torch.max(probs, dim=1)

        label = id2label[predicted.item()]
        score = confidence.item()

        return {
            "label": label,
            "confidence": round(score, 3),
            "source": "speech"
        }

    except Exception as e:
        print("❌ SER ERROR:", e)
        return {
            "label": "neutral",
            "confidence": 0.0,
            "source": "speech"
        }