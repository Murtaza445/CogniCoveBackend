import asyncio
import os
from dotenv import load_dotenv
from realtime.audio_pipeline import VoskStreamingSTT, PiperTTS, generate_realtime_response
from realtime.emotion_pipeline import analyze_speech

# Load environment variables from .env file
load_dotenv()


async def run():
    with open("teststt.wav", "rb") as f:
        audio = f.read()

    # STT
    stt = VoskStreamingSTT()
    text = await stt.process_audio_chunk(audio)

    print("\n🧑 USER:", text)

    if not text:
        print("❌ STT failed — no text detected")
        return

    # 🔥 SPEECH EMOTION
    emotion_result = await analyze_speech(audio)
    print("🎭 Emotion:", emotion_result)

    emotion_label = emotion_result["label"]

    # LLM
    response = await generate_realtime_response(
        text=text,
        session_id="test",
        emotion=emotion_label,
        therapy_chain_factory=lambda: __import__("chains").get_therapy_chain(),
        chat_history_builder=lambda x: []
    )

    print("\n🤖 AI:", response)

    # TTS
    tts = PiperTTS()
    audio_out = await tts.text_to_audio_bytes(response)

    if not audio_out:
        print("❌ TTS failed")
        return

    with open("responsefinal.wav", "wb") as f:
        f.write(audio_out)

    print("✅ Full pipeline complete → responsefinal.wav")


if __name__ == "__main__":
    asyncio.run(run())