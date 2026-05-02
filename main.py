"""FastAPI backend for Disorder Chat application."""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import re
import base64
import asyncio
from datetime import datetime
from pydantic import ValidationError

from chains import get_therapy_chain, get_summary_chain, get_classification_chain, get_diagnosis_chain
from models import SummaryModel, ClassificationModel, DiagnosticResults, DiagnosticResultStructured
from vectorstore import get_vectorstore
from constants import DISORDER_CATEGORIES, APP_TITLE, APP_PAGE_TITLE
from hybrid_retrieval import hybrid_classify, hybrid_diagnose
from knowledge_graph import build_knowledge_graph
from langchain_core.messages import HumanMessage, AIMessage
from realtime.ws_manager import WebSocketSessionManager
from realtime.audio_pipeline import VoskStreamingSTT, PiperTTS
from realtime.emotion_pipeline import analyze_speech, analyze_face
from suicide_detection import get_suicide_detector
from email_service import send_crisis_alert_email
from deploy_validate import validate

# Fail fast if model files are missing or are Git LFS pointers
validate()


# Initialize FastAPI app
app = FastAPI(title=APP_TITLE, description="Mental Health Disorder Assessment API")

# Add CORS middleware for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "https://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store (replace with database for production)
sessions_store = {}

# Initialize knowledge graph (cached)
kg = build_knowledge_graph()

# Initialize suicide detection system
print("\n" + "="*80)
suicide_detector = get_suicide_detector()
print("="*80 + "\n")


# ==================== Request/Response Models ====================

class MessageRequest(BaseModel):
    """Request model for therapy messages."""
    session_id: str
    content: str
    facial_emotion: Optional[str] = None
    user_email: Optional[str] = None


class TherapyResponse(BaseModel):
    """Response model for therapy message."""
    session_id: str
    user_message: str
    assistant_message: str
    timestamp: str
    facial_emotion: Optional[str] = None
    crisis_alert: bool = False
    crisis_probability: Optional[float] = None
    crisis_reason: Optional[str] = None


class SessionCreateRequest(BaseModel):
    """Request model to create a new session."""
    user_id: Optional[str] = None


class SessionCreateResponse(BaseModel):
    """Response model when creating a session."""
    session_id: str
    created_at: str


class SessionMessage(BaseModel):
    """Message within a session."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str


class SessionData(BaseModel):
    """Full session data."""
    session_id: str
    user_id: Optional[str]
    created_at: str
    messages: List[SessionMessage]
    summary: Optional[str] = None
    classification: Optional[str] = None
    diagnostic_results: Optional[List[str]] = None
    session_ended: bool = False


class SummaryRequest(BaseModel):
    """Request to generate clinical summary."""
    session_id: str


class SummaryResponse(BaseModel):
    """Response with clinical summary."""
    session_id: str
    summary: str
    timestamp: str


class ClassifyRequest(BaseModel):
    """Request to classify disorder."""
    session_id: str
    summary: str


class ClassifyResponse(BaseModel):
    """Response with disorder classification."""
    session_id: str
    classification: str
    comorbid_categories: List[str]
    timestamp: str


class DiagnoseRequest(BaseModel):
    """Request for diagnostic results."""
    session_id: str
    summary: str
    classification: str


class SymptomRecord(BaseModel):
    """Symptom record for database population."""
    disorder_category: Optional[str] = None
    symptom_name: str
    severity: Optional[str] = None  # 'mild', 'moderate', 'severe'
    duration_days: Optional[int] = None
    frequency: Optional[str] = None  # 'daily', 'weekly', 'occasional'


class ImpairmentRecord(BaseModel):
    """Impairment record for database population."""
    disorder_category: Optional[str] = None
    impairment_area: str  # 'work', 'social', 'academic', 'self_care', 'family'
    severity_level: Optional[str] = None  # 'mild', 'moderate', 'severe'
    description: Optional[str] = None


class PatternAlignment(BaseModel):
    """Pattern alignment for diagnostic confirmation."""
    diagnostic_disorder_name: str
    pattern_name: str
    alignment_percentage: float
    supporting_evidence: List[str] = Field(default_factory=list)
    contradicting_evidence: List[str] = Field(default_factory=list)
    clinical_notes: Optional[str] = None


class ComorbidDiagnosis(BaseModel):
    """Comorbid diagnosis relationship."""
    primary_disorder_name: str
    comorbid_disorder_name: str
    relationship_strength: Optional[float] = None  # 0.0-1.0


class MentalStateTracking(BaseModel):
    """Daily mental state tracking aggregate."""
    tracking_date: str
    average_mood_score: Optional[float] = None  # 0-10
    average_risk_level: Optional[str] = None  # 'low', 'moderate', 'high'
    total_sessions_today: int = 0
    crisis_count: int = 0
    predominant_emotion: Optional[str] = None
    notes: Optional[str] = None


class EnhancedDiagnoseResponse(BaseModel):
    """Enhanced diagnostic response with database-ready structured data."""
    session_id: str
    diagnostic_results: List[DiagnosticResultStructured]
    symptoms: List[SymptomRecord] = Field(default_factory=list)
    impairment_records: List[ImpairmentRecord] = Field(default_factory=list)
    pattern_alignment_summary: List[PatternAlignment] = Field(default_factory=list)
    comorbid_diagnoses: List[ComorbidDiagnosis] = Field(default_factory=list)
    mental_state_tracking: Optional[MentalStateTracking] = None
    session_summary: Optional[str] = None
    overall_mood_score: Optional[float] = None
    overall_risk_level: Optional[str] = None
    timestamp: str


class DiagnoseResponse(BaseModel):
    """Response with diagnostic results."""
    session_id: str
    diagnostic_results: List[DiagnosticResultStructured]
    timestamp: str


class TTSRequest(BaseModel):
    """Request to synthesize speech."""
    text: str


class TTSResponse(BaseModel):
    """Response with synthesized audio."""
    audio_base64: str
    audio_type: str


class FaceEmotionResponse(BaseModel):
    """Response with facial emotion analysis."""
    emotion: str
    confidence: float


class SessionEndRequest(BaseModel):
    """Request to end session."""
    session_id: str


class SessionEndResponse(BaseModel):
    """Response when session ends."""
    session_id: str
    summary: Optional[str] = None
    classification: Optional[str] = None
    diagnostic_results: Optional[List[DiagnosticResultStructured]] = None
    symptoms: List[SymptomRecord] = Field(default_factory=list)
    impairment_records: List[ImpairmentRecord] = Field(default_factory=list)
    ended_at: str


# ==================== Helper Functions ====================

def get_or_create_session(session_id: str, user_id: Optional[str] = None) -> dict:
    """Get or create session in store."""
    if session_id not in sessions_store:
        sessions_store[session_id] = {
            "session_id": session_id,
            "user_id": user_id,
            "created_at": datetime.utcnow().isoformat(),
            "messages": [],
            "summary": None,
            "classification": None,
            "diagnostic_results": None,
            "session_ended": False,
        }
    return sessions_store[session_id]


def build_chat_history_from_session(session_id: str) -> List:
    """Build LangChain chat history from session messages."""
    session = sessions_store.get(session_id, {})
    messages = session.get("messages", [])
    
    chat_history = []
    for msg in messages:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(AIMessage(content=msg["content"]))
    
    return chat_history


def parse_json_response(text: str) -> Optional[dict]:
    """Extract and parse JSON from text response."""
    if not text:
        return None
    
    s = text.strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        # Try to extract JSON from text
        match = re.search(r"\{.*\}", s, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
    return None

def extract_symptoms_from_summary(summary_json: str, category: Optional[str] = None) -> List[SymptomRecord]:
    """Extract structured symptom records from clinical summary."""
    symptoms = []
    try:
        data = json.loads(summary_json)
        symptom_list = data.get("symptom_list", [])
        duration = data.get("duration_requirements", "")
        
        # Parse duration to days (rough estimate)
        duration_days = None
        if duration:
            weeks = re.findall(r'(\d+)\s*weeks?', duration.lower())
            days = re.findall(r'(\d+)\s*days?', duration.lower())
            if weeks:
                duration_days = int(weeks[0]) * 7
            elif days:
                duration_days = int(days[0])
        
        for symptom_name in symptom_list:
            if isinstance(symptom_name, str):
                symptoms.append(SymptomRecord(
                    disorder_category=category,
                    symptom_name=symptom_name.strip(),
                    severity="moderate",  # Default, could be enhanced by NLP
                    duration_days=duration_days,
                    frequency="ongoing"
                ))
    except Exception:
        pass
    
    return symptoms


def extract_impairment_from_summary(summary_json: str, category: Optional[str] = None) -> List[ImpairmentRecord]:
    """Extract structured impairment records from clinical summary."""
    impairments = []
    try:
        data = json.loads(summary_json)
        impairment_desc = data.get("impairment_requirement", "")
        overview = data.get("overview", "")
        
        # Map keywords to impairment areas
        impairment_keywords = {
            "work": ["work", "job", "workplace", "productivity", "employment"],
            "social": ["social", "relationships", "friends", "family interaction", "isolation"],
            "academic": ["school", "academic", "study", "grades", "learning"],
            "self_care": ["hygiene", "self-care", "grooming", "sleep", "eating"],
            "family": ["family", "parent", "spouse", "child"]
        }
        
        combined_text = (impairment_desc + " " + overview).lower()
        
        for area, keywords in impairment_keywords.items():
            if any(k in combined_text for k in keywords):
                impairments.append(ImpairmentRecord(
                    disorder_category=category,
                    impairment_area=area,
                    severity_level="moderate",
                    description=impairment_desc if area == "work" else overview
                ))
    except Exception:
        pass
    
    return impairments


def build_pattern_alignment(diagnostic_results_str: str, disorder_name: str) -> List[PatternAlignment]:
    """Build pattern alignment records from diagnostic results."""
    alignments = []
    try:
        data = json.loads(diagnostic_results_str)
        alignment_patterns = [
            {
                "pattern_name": "Symptom Matching",
                "evidence_field": "matched_criteria"
            },
            {
                "pattern_name": "Duration Criteria",
                "evidence_field": "duration_met"
            },
            {
                "pattern_name": "Functional Impairment",
                "evidence_field": "impairment_met"
            }
        ]
        
        for pattern in alignment_patterns:
            value = data.get(pattern["evidence_field"])
            if isinstance(value, bool):
                alignment_pct = 100.0 if value else 0.0
            elif isinstance(value, (int, float)):
                alignment_pct = min(100.0, float(value))
            else:
                alignment_pct = 50.0
            
            alignments.append(PatternAlignment(
                diagnostic_disorder_name=disorder_name,
                pattern_name=pattern["pattern_name"],
                alignment_percentage=alignment_pct,
                supporting_evidence=[data.get("recommendation", "")],
                contradicting_evidence=[data.get("recommendation", "")] if data.get("exclusion_triggered") else [],
                clinical_notes=data.get("recommendation")
            ))
    except Exception:
        pass
    
    return alignments


def build_comorbid_relationships(classification_json: str, primary_disorder: str) -> List[ComorbidDiagnosis]:
    """Build comorbid diagnosis relationships from classification."""
    relationships = []
    try:
        data = json.loads(classification_json)
        comorbid_cats = data.get("comorbid_categories", [])
        comorbid_scores = data.get("comorbidity_scores", {})
        
        for comorbid_cat in comorbid_cats:
            score = float(comorbid_scores.get(comorbid_cat, 0.5))
            relationships.append(ComorbidDiagnosis(
                primary_disorder_name=primary_disorder,
                comorbid_disorder_name=comorbid_cat,
                relationship_strength=score
            ))
    except Exception:
        pass
    
    return relationships


def compute_mental_state_tracking(session_data: dict) -> Optional[MentalStateTracking]:
    """Compute daily mental state tracking from session data."""
    try:
        messages = session_data.get("messages", [])
        total_sessions = len([m for m in messages if m.get("role") == "user"])
        
        # Count crisis indicators
        crisis_count = 0
        predominant_emotions = []
        negative_emotion_keywords = ["suicide", "harm", "crisis", "emergency", "danger"]
        
        for msg in messages:
            content = msg.get("content", "").lower()
            if any(keyword in content for keyword in negative_emotion_keywords):
                crisis_count += 1
            
            # Extract emotions (simplified)
            emotion_keywords = ["sad", "anxious", "angry", "hopeless", "stressed", "overwhelmed"]
            for emotion in emotion_keywords:
                if emotion in content:
                    predominant_emotions.append(emotion)
        
        predominant = predominant_emotions[0] if predominant_emotions else "neutral"
        
        return MentalStateTracking(
            tracking_date=datetime.utcnow().isoformat(),
            average_mood_score=5.0,  # Default neutral
            average_risk_level="low" if crisis_count == 0 else ("high" if crisis_count > 2 else "moderate"),
            total_sessions_today=total_sessions,
            crisis_count=crisis_count,
            predominant_emotion=predominant,
            notes=f"Session analysis completed with {total_sessions} interactions"
        )
    except Exception:
        return None


def convert_diagnostic_to_structured(
    diagnostic_json_str: str, 
    conversation_summary: Optional[str] = None,
    overview: Optional[str] = None
) -> Optional[DiagnosticResultStructured]:
    """Convert a diagnostic result JSON string to DiagnosticResultStructured."""
    try:
        data = json.loads(diagnostic_json_str)
        return DiagnosticResultStructured(
            disorder_name=data.get("disorder_name", ""),
            category=data.get("category"),
            matched_criteria=data.get("matched_criteria"),
            total_required=data.get("total_required"),
            alignment_score=data.get("alignment_score"),
            confidence_level=data.get("confidence_level"),
            duration_met=data.get("duration_met"),
            impairment_met=data.get("impairment_met"),
            exclusion_triggered=data.get("exclusion_triggered"),
            recommendation=data.get("recommendation"),
            conversation_summary=conversation_summary,
            overview=overview
        )
    except Exception:
        return None


# Realtime websocket manager (separate from REST workflow)
realtime_ws_manager = WebSocketSessionManager(
    get_or_create_session=get_or_create_session,
    build_chat_history_from_session=build_chat_history_from_session,
    therapy_chain_factory=get_therapy_chain,
)

# ==================== API Endpoints ====================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "app": APP_TITLE}


@app.post("/api/sessions/create", response_model=SessionCreateResponse)
async def create_session(request: SessionCreateRequest):
    """Create a new session."""
    session_id = f"session_{datetime.utcnow().timestamp()}"
    session = get_or_create_session(session_id, request.user_id)
    
    return SessionCreateResponse(
        session_id=session_id,
        created_at=session["created_at"]
    )


@app.get("/api/sessions/{session_id}", response_model=SessionData)
async def get_session(session_id: str):
    """Retrieve session data."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    messages = [
        SessionMessage(
            role=msg["role"],
            content=msg["content"],
            timestamp=msg["timestamp"]
        )
        for msg in session["messages"]
    ]
    
    return SessionData(
        session_id=session["session_id"],
        user_id=session.get("user_id"),
        created_at=session["created_at"],
        messages=messages,
        summary=session.get("summary"),
        classification=session.get("classification"),
        diagnostic_results=session.get("diagnostic_results"),
        session_ended=session.get("session_ended", False)
    )


@app.post("/api/therapy", response_model=TherapyResponse)
async def therapy_message(request: MessageRequest):
    """
    Process therapy message and return response immediately.
    Crisis detection runs in background without blocking therapy response.
    """
    # Get or create session
    session = get_or_create_session(request.session_id)
    
    # Add user message
    user_msg = {
        "role": "user",
        "content": request.content,
        "timestamp": datetime.utcnow().isoformat()
    }
    session["messages"].append(user_msg)
    
    try:
        # ==================== THERAPY PIPELINE (FAST) ====================
        # Generate therapy response immediately - this is the main pipeline
        therapy_chain = get_therapy_chain()
        chat_history = build_chat_history_from_session(request.session_id)
        
        query = request.content
        if request.facial_emotion:
            query = f"[Detected user emotion — facial expression: {request.facial_emotion}]\n{query}"

        response = therapy_chain.invoke({
            "chat_history": chat_history,
            "query": query
        })
        response_text = response.content
        
        # Add assistant message to session
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        }
        session["messages"].append(assistant_msg)
        
        # ==================== SYNCHRONOUS CRISIS DETECTION (FAST) ====================
        # Run ELECTRA synchronously (fast) to determine if we need to alert the frontend
        crisis_alert = False
        crisis_probability = None
        crisis_reason = None
        electra_result = None

        if suicide_detector.is_available():
            try:
                electra_result = suicide_detector.predict(request.content, session_id=request.session_id)
                suicide_detector.print_result(electra_result, session_id=request.session_id)
                crisis_probability = electra_result.get("probabilities", {}).get("suicidal")
                if crisis_probability is not None and crisis_probability >= 0.85:
                    crisis_alert = True
                    crisis_reason = electra_result.get("alert_reason") or f"ELECTRA suicidal probability {crisis_probability:.2%}"
            except Exception as e:
                print(f"⚠️  Synchronous ELECTRA detection error: {str(e)}")

        # ==================== CRISIS EMAIL ALERT (BACKGROUND) ====================
        if crisis_alert and request.user_email:
            asyncio.create_task(
                asyncio.to_thread(
                    send_crisis_alert_email,
                    request.user_email,
                    user_name=None,
                    session_id=request.session_id,
                    message_preview=request.content,
                )
            )

        # ==================== FULL CRISIS DETECTION (BACKGROUND) ====================
        # Run LLM detection + moderate tracking in background without blocking response
        async def run_crisis_detection_background(electra_result=electra_result):
            """Run crisis detection in background - fire and forget."""
            try:
                # Run LLM detection
                try:
                    llm_result = suicide_detector.predict_with_llm(request.content, session_id=request.session_id)
                    suicide_detector.print_result(llm_result, session_id=request.session_id)
                except Exception as e:
                    print(f"⚠️  LLM detection error: {str(e)}")
                    llm_result = None

                # ==================== MODERATE TRACKING + FINAL RE-CHECK ====================
                tracking_result = None
                final_recheck_result = None
                try:
                    tracking_result = suicide_detector.update_moderate_tracking_and_maybe_recheck(
                        session_id=request.session_id,
                        text=request.content,
                        electra_result=electra_result,
                        llm_result=llm_result,
                        timestamp=user_msg["timestamp"],
                    )
                    final_recheck_result = tracking_result.get("final_recheck") if tracking_result else None

                    # Print tracking info (non-emergency)
                    if tracking_result and tracking_result.get("alert_reason"):
                        print(f"\n⚠️  {tracking_result['alert_reason']}")
                        print(f"    Session ID: {request.session_id}")

                    # Print final re-check result if it ran
                    if final_recheck_result is not None:
                        suicide_detector.print_result(final_recheck_result, session_id=request.session_id)
                except Exception as e:
                    print(f"⚠️  Moderate tracking error: {str(e)}")
                    tracking_result = None
                    final_recheck_result = None
                
                # Store detection results in session
                if electra_result is not None or llm_result is not None:
                    if 'suicide_detections' not in session:
                        session['suicide_detections'] = []
                    session['suicide_detections'].append({
                        'timestamp': user_msg['timestamp'],
                        'user_message': request.content,
                        'electra_result': electra_result,
                        'llm_result': llm_result,
                        'moderate_tracking': tracking_result,
                        'llm_final_recheck_result': final_recheck_result
                    })
            except Exception as e:
                print(f"❌ Background crisis detection error: {str(e)}")
        
        # Start full crisis detection as background task (fire and forget)
        asyncio.create_task(run_crisis_detection_background())
        
        # ==================== RETURN THERAPY RESPONSE IMMEDIATELY ====================
        # Therapy response now includes synchronous crisis alert info
        return TherapyResponse(
            session_id=request.session_id,
            user_message=request.content,
            assistant_message=response_text,
            timestamp=assistant_msg["timestamp"],
            facial_emotion=request.facial_emotion,
            crisis_alert=crisis_alert,
            crisis_probability=crisis_probability,
            crisis_reason=crisis_reason
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating therapy response: {str(e)}")


@app.post("/api/therapy/audio")
async def therapy_audio(
    session_id: str,
    file: UploadFile = File(...),
    facial_emotion: Optional[str] = Form(None),
    tts: bool = Form(True),
    user_email: Optional[str] = Form(None)
):
    """
    Process audio input and return therapy response.
    
    Endpoint for Next.js frontend to send audio files.
    
    Request:
    - session_id: Query parameter (session identifier)
    - file: Audio file (binary, WAV format, 16-bit, 16kHz, mono)
    - facial_emotion: Optional aggregated facial emotion (form field)
    - tts: Whether to synthesize audio response (default true for backward compat)
    
    Response:
    {
      "session_id": "session_123",
      "user_text": "What was transcribed from audio",
      "ai_response": "Therapy response text",
      "emotion": "emotion_label",
      "audio_base64": "<base64-encoded WAV or null>",
      "audio_type": "audio/wav",
      "timestamp": "2024-04-12T..."
    }
    """
    
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id required")
    
    try:
        # Step 1: Read audio bytes from uploaded file
        audio_bytes = await file.read()
        
        if not audio_bytes or len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty audio file")
        
        # Step 2: Get or create session
        session = get_or_create_session(session_id)
        
        # ==================== STEP 3: STT ====================
        stt = VoskStreamingSTT()
        
        if not stt.enabled:
            raise HTTPException(status_code=503, detail="STT service not available. Vosk model not loaded.")
        
        # Decode audio to text using Vosk
        user_text = await stt.process_audio_chunk(audio_bytes)
        
        if not user_text or user_text.strip() == "":
            raise HTTPException(status_code=400, detail="❌ No speech detected. Please speak clearly and try again.")
        
        # ==================== STEP 4: ADD TO SESSION ====================
        user_msg = {
            "role": "user",
            "content": user_text.strip(),
            "timestamp": datetime.utcnow().isoformat()
        }
        session["messages"].append(user_msg)
        
        # ==================== STEP 5: EMOTION + THERAPY (FAST PATH) ====================
        # Generate emotion analysis and therapy response (core pipeline)
        async def analyze_emotion_async():
            """Run emotion analysis asynchronously."""
            try:
                emotion_result = await analyze_speech(audio_bytes)
                return emotion_result.get("label", "neutral") if isinstance(emotion_result, dict) else "neutral"
            except Exception as e:
                print(f"Emotion analysis warning: {e}")
                return "neutral"
        
        async def generate_therapy_async():
            """Generate therapy response asynchronously."""
            therapy_chain = get_therapy_chain()
            chat_history = build_chat_history_from_session(session_id)

            # Build emotion-aware query
            emotion_parts = []
            if emotion_label and emotion_label != "neutral":
                emotion_parts.append(f"speech tone: {emotion_label}")
            if facial_emotion and facial_emotion != "neutral":
                emotion_parts.append(f"facial expression: {facial_emotion}")

            if emotion_parts:
                augmented_query = f"[Detected user emotion — {', '.join(emotion_parts)}]\n{user_text.strip()}"
            else:
                augmented_query = user_text.strip()

            return therapy_chain.invoke({
                "chat_history": chat_history,
                "query": augmented_query
            })
        
        # Run emotion analysis first, then therapy generation (therapy needs emotion context)
        emotion_label = await analyze_emotion_async()
        llm_response = await generate_therapy_async()
        
        # Get response text
        response_text = llm_response.content
        
        # Add assistant message to session
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        }
        session["messages"].append(assistant_msg)
        
        # ==================== STEP 5B: SYNCHRONOUS CRISIS DETECTION (FAST) ====================
        # Run ELECTRA synchronously (fast) to determine if we need to alert the frontend
        crisis_alert = False
        crisis_probability = None
        crisis_reason = None
        electra_result = None

        if suicide_detector.is_available():
            try:
                electra_result = suicide_detector.predict(user_text.strip(), session_id=session_id)
                suicide_detector.print_result(electra_result, session_id=session_id)
                crisis_probability = electra_result.get("probabilities", {}).get("suicidal")
                if crisis_probability is not None and crisis_probability >= 0.85:
                    crisis_alert = True
                    crisis_reason = electra_result.get("alert_reason") or f"ELECTRA suicidal probability {crisis_probability:.2%}"
            except Exception as e:
                print(f"⚠️  Synchronous ELECTRA detection error: {str(e)}")

        # ==================== CRISIS EMAIL ALERT (BACKGROUND) ====================
        if crisis_alert and user_email:
            asyncio.create_task(
                asyncio.to_thread(
                    send_crisis_alert_email,
                    user_email,
                    user_name=None,
                    session_id=session_id,
                    message_preview=user_text.strip(),
                )
            )

        # ==================== FULL CRISIS DETECTION (BACKGROUND) ====================
        # Run LLM detection + moderate tracking in background without blocking response
        async def run_crisis_detection_background(electra_result=electra_result):
            """Run crisis detection in background - fire and forget."""
            try:
                # Run LLM detection
                try:
                    llm_result = suicide_detector.predict_with_llm(user_text.strip(), session_id=session_id)
                    suicide_detector.print_result(llm_result, session_id=session_id)
                except Exception as e:
                    print(f"⚠️  LLM detection error: {str(e)}")
                    llm_result = None

                # ==================== MODERATE TRACKING + FINAL RE-CHECK ====================
                tracking_result = None
                final_recheck_result = None
                try:
                    tracking_result = suicide_detector.update_moderate_tracking_and_maybe_recheck(
                        session_id=session_id,
                        text=user_text.strip(),
                        electra_result=electra_result,
                        llm_result=llm_result,
                        timestamp=user_msg["timestamp"],
                    )
                    final_recheck_result = tracking_result.get("final_recheck") if tracking_result else None

                    if tracking_result and tracking_result.get("alert_reason"):
                        print(f"\n⚠️  {tracking_result['alert_reason']}")
                        print(f"    Session ID: {session_id}")

                    if final_recheck_result is not None:
                        suicide_detector.print_result(final_recheck_result, session_id=session_id)
                except Exception as e:
                    print(f"⚠️  Moderate tracking error: {str(e)}")
                    tracking_result = None
                    final_recheck_result = None
                
                # Store detection results in session
                if electra_result is not None or llm_result is not None:
                    if 'suicide_detections' not in session:
                        session['suicide_detections'] = []
                    session['suicide_detections'].append({
                        'timestamp': user_msg['timestamp'],
                        'user_message': user_text.strip(),
                        'electra_result': electra_result,
                        'llm_result': llm_result,
                        'moderate_tracking': tracking_result,
                        'llm_final_recheck_result': final_recheck_result
                    })
            except Exception as e:
                print(f"❌ Background crisis detection error: {str(e)}")
        
        # Start full crisis detection as background task (fire and forget)
        asyncio.create_task(run_crisis_detection_background())
        
        # ==================== STEP 6: TTS (OPTIONAL) ====================
        audio_base64 = None
        if tts:
            try:
                tts_engine = PiperTTS()
                audio_response = await tts_engine.text_to_audio_bytes(response_text)
                
                if not audio_response or len(audio_response) < 1000:
                    print(f"⚠️ TTS produced invalid audio: {len(audio_response) if audio_response else 0} bytes")
                else:
                    # Encode response audio as base64 for JSON response
                    audio_base64 = base64.b64encode(audio_response).decode('utf-8')
            except Exception as e:
                print(f"⚠️ TTS processing failed: {str(e)}")
                # Do NOT raise — text response is still valuable
        
        # ==================== STEP 7: RETURN ====================
        return {
            "session_id": session_id,
            "user_text": user_text.strip(),
            "ai_response": response_text,
            "emotion": emotion_label,
            "audio_base64": audio_base64,
            "audio_type": "audio/wav" if audio_base64 else None,
            "timestamp": datetime.utcnow().isoformat(),
            "crisis_alert": crisis_alert,
            "crisis_probability": crisis_probability,
            "crisis_reason": crisis_reason,
        }
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Audio therapy error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing audio: {str(e)}")


@app.post("/api/summary", response_model=SummaryResponse)
async def generate_summary(request: SummaryRequest):
    """Generate clinical summary from session messages."""
    if request.session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[request.session_id]
    
    # Skip if already generated
    if session.get("summary") is not None:
        return SummaryResponse(
            session_id=request.session_id,
            summary=session["summary"],
            timestamp=datetime.utcnow().isoformat()
        )
    
    try:
        summary_chain = get_summary_chain()
        chat_history = build_chat_history_from_session(request.session_id)
        
        summary_resp = summary_chain.invoke({"chat_history": chat_history})
        summary_text = summary_resp.content
        
        # Validate with Pydantic
        validated = None
        if summary_text:
            s = summary_text.strip()
            try:
                validated = SummaryModel.model_validate_json(s)
            except ValidationError:
                match = re.search(r"\{.*\}", s, re.DOTALL)
                if match:
                    try:
                        validated = SummaryModel.model_validate_json(match.group(0))
                    except Exception:
                        validated = None
        
        if validated:
            summary_json = validated.model_dump_json(indent=2)
        else:
            summary_json = summary_text or "Unable to generate summary."
        
        session["summary"] = summary_json
        
        return SummaryResponse(
            session_id=request.session_id,
            summary=summary_json,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating summary: {str(e)}")


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify_disorder(request: ClassifyRequest):
    """Classify disorder categories using hybrid FAISS+KG retrieval."""
    if request.session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[request.session_id]
    
    # Skip if already classified
    if session.get("classification") is not None:
        try:
            cls_data = json.loads(session["classification"])
            comorbid = cls_data.get("comorbid_categories", [])
        except:
            comorbid = []
        
        return ClassifyResponse(
            session_id=request.session_id,
            classification=session["classification"],
            comorbid_categories=comorbid,
            timestamp=datetime.utcnow().isoformat()
        )
    
    try:
        print(f"📋 CLASSIFY: Input summary type: {type(request.summary)}")
        print(f"📋 CLASSIFY: Input summary length: {len(request.summary) if request.summary else 0}")
        print(f"📋 CLASSIFY: Input summary (first 200 chars): {request.summary[:200] if request.summary else 'NONE'}")
        
        cls_validated = hybrid_classify(request.summary)
        print(f"✅ CLASSIFY: hybrid_classify returned: {type(cls_validated)}")
        
        if cls_validated:
            classification_json = cls_validated.model_dump_json(indent=2)
            comorbid_categories = cls_validated.comorbid_categories or []
            print(f"✅ CLASSIFY: Classification complete - {len(comorbid_categories)} comorbid categories")
        else:
            classification_json = "Unable to classify."
            comorbid_categories = []
            print("⚠️ CLASSIFY: hybrid_classify returned None")
        
        session["classification"] = classification_json
        
        return ClassifyResponse(
            session_id=request.session_id,
            classification=classification_json,
            comorbid_categories=comorbid_categories,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        import traceback
        print(f"❌ CLASSIFY ERROR: {str(e)}")
        print(f"❌ CLASSIFY TRACEBACK:\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Error classifying disorder: {str(e)}")


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def generate_diagnosis(request: DiagnoseRequest):
    """Generate diagnostic results for detected categories."""
    if request.session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[request.session_id]
    
    # Extract conversation summary and overview for context
    conversation_summary = None
    overview = None
    try:
        summary_data = json.loads(request.summary)
        conversation_summary = summary_data.get("conversation_summary")
        overview = summary_data.get("overview")
    except:
        pass
    
    # Skip if already diagnosed
    if session.get("diagnostic_results") is not None:
        diagnostic_structured = []
        for result_str in session["diagnostic_results"]:
            if isinstance(result_str, str):
                structured = convert_diagnostic_to_structured(result_str, conversation_summary, overview)
                if structured:
                    diagnostic_structured.append(structured)
        
        return DiagnoseResponse(
            session_id=request.session_id,
            diagnostic_results=diagnostic_structured,
            timestamp=datetime.utcnow().isoformat()
        )
    
    try:
        # Parse classification
        cls_validated = None
        try:
            cls_data = json.loads(request.classification)
            cls_validated = ClassificationModel(**cls_data)
        except Exception:
            pass
        
        if cls_validated:
            results = hybrid_diagnose(request.summary, cls_validated)
            diagnostic_results_raw = results if results else []
        else:
            diagnostic_results_raw = []
        
        # Convert to structured format
        diagnostic_structured = []
        for result_str in diagnostic_results_raw:
            structured = convert_diagnostic_to_structured(result_str, conversation_summary, overview)
            if structured:
                diagnostic_structured.append(structured)
        
        session["diagnostic_results"] = diagnostic_results_raw  # Store raw for caching
        
        return DiagnoseResponse(
            session_id=request.session_id,
            diagnostic_results=diagnostic_structured,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diagnosis: {str(e)}")


@app.post("/api/diagnose/enhanced", response_model=EnhancedDiagnoseResponse)
async def generate_diagnosis_enhanced(request: DiagnoseRequest):
    """Generate enhanced diagnostic results with structured database-ready data."""
    if request.session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[request.session_id]
    
    try:
        # Parse classification
        cls_validated = None
        cls_json = request.classification
        try:
            cls_data = json.loads(cls_json)
            cls_validated = ClassificationModel(**cls_data)
        except Exception:
            pass
        
        # Generate diagnostic results
        if cls_validated:
            diagnostic_results = hybrid_diagnose(request.summary, cls_validated)
        else:
            diagnostic_results = []
        
        # Extract primary disorder from classification
        primary_disorder = "Unknown"
        try:
            cls_data = json.loads(cls_json)
            primary_disorder = cls_data.get("predicted_category", "Unknown")
        except:
            pass
        
        # Extract structured data for database
        all_symptoms: List[SymptomRecord] = extract_symptoms_from_summary(request.summary, primary_disorder)
        all_impairments: List[ImpairmentRecord] = extract_impairment_from_summary(request.summary, primary_disorder)
        
        all_patterns: List[PatternAlignment] = []
        
        # Extract conversation summary and overview for context
        conversation_summary = None
        overview = None
        try:
            summary_data = json.loads(request.summary)
            conversation_summary = summary_data.get("conversation_summary")
            overview = summary_data.get("overview")
        except:
            pass
        
        # Convert diagnostic results to structured format
        diagnostic_structured = []
        for diag_result_str in diagnostic_results:
            try:
                diag_data = json.loads(diag_result_str)
                disorder_name = diag_data.get("disorder_name", "Unknown")
                patterns = build_pattern_alignment(diag_result_str, disorder_name)
                all_patterns.extend(patterns)
                
                # Convert to structured format
                structured = convert_diagnostic_to_structured(diag_result_str, conversation_summary, overview)
                if structured:
                    diagnostic_structured.append(structured)
            except:
                pass
        
        # Build comorbid relationships
        comorbid_rels: List[ComorbidDiagnosis] = build_comorbid_relationships(cls_json, primary_disorder)
        
        # Compute mental state tracking
        mental_tracking = compute_mental_state_tracking(session)
        
        # Extract session-level summary fields
        session_summary = conversation_summary
        overall_mood_score = mental_tracking.average_mood_score if mental_tracking else None
        overall_risk_level = mental_tracking.average_risk_level if mental_tracking else None
        
        return EnhancedDiagnoseResponse(
            session_id=request.session_id,
            diagnostic_results=diagnostic_structured,
            symptoms=all_symptoms,
            impairment_records=all_impairments,
            pattern_alignment_summary=all_patterns,
            comorbid_diagnoses=comorbid_rels,
            mental_state_tracking=mental_tracking,
            session_summary=session_summary,
            overall_mood_score=overall_mood_score,
            overall_risk_level=overall_risk_level,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating enhanced diagnosis: {str(e)}")


@app.post("/api/sessions/{session_id}/end", response_model=SessionEndResponse)
async def end_session(session_id: str):
    """End a session and return all results."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    session["session_ended"] = True
    
    # Reset suicide detection tracking for this session
    suicide_detector.reset_session(session_id)
    
    # Extract conversation summary and overview for context
    conversation_summary = None
    overview = None
    try:
        summary_data = json.loads(session.get("summary", "{}"))
        conversation_summary = summary_data.get("conversation_summary")
        overview = summary_data.get("overview")
    except:
        pass
    
    # Extract primary disorder from classification
    primary_disorder = "Unknown"
    try:
        cls_data = json.loads(session.get("classification", "{}"))
        primary_disorder = cls_data.get("predicted_category", "Unknown")
    except:
        pass
    
    # Convert diagnostic results to structured format
    diagnostic_structured = []
    if session.get("diagnostic_results"):
        for result_str in session["diagnostic_results"]:
            if isinstance(result_str, str):
                structured = convert_diagnostic_to_structured(result_str, conversation_summary, overview)
                if structured:
                    diagnostic_structured.append(structured)
    
    # Extract symptoms and impairments
    all_symptoms = extract_symptoms_from_summary(session.get("summary", "{}"), primary_disorder)
    all_impairments = extract_impairment_from_summary(session.get("summary", "{}"), primary_disorder)
    
    return SessionEndResponse(
        session_id=session_id,
        summary=session.get("summary"),
        classification=session.get("classification"),
        diagnostic_results=diagnostic_structured if diagnostic_structured else None,
        symptoms=all_symptoms,
        impairment_records=all_impairments,
        ended_at=datetime.utcnow().isoformat()
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_store[session_id]
    return {"message": f"Session {session_id} deleted"}


@app.websocket("/ws/video/{session_id}")
async def video_audio_ws(websocket: WebSocket, session_id: str):
    """Realtime websocket endpoint for chunked audio processing and AI response streaming."""
    await realtime_ws_manager.connect(websocket, session_id)

    try:
        while True:
            payload = await websocket.receive()

            if payload.get("bytes") is not None:
                await realtime_ws_manager.handle_audio_chunk(session_id, payload["bytes"])
                continue

            text_payload = payload.get("text")
            if text_payload is not None:
                await realtime_ws_manager.handle_text_message(session_id, text_payload)

    except WebSocketDisconnect:
        pass
    finally:
        await realtime_ws_manager.disconnect_and_persist(session_id)


@app.post("/api/emotion/face", response_model=FaceEmotionResponse)
async def analyze_facial_emotion(file: UploadFile = File(...)):
    """
    Accept a webcam frame (JPEG/PNG) and return the detected facial emotion.
    """
    try:
        frame_bytes = await file.read()
        if not frame_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")

        result = await analyze_face(frame_bytes)

        return FaceEmotionResponse(
            emotion=result.get("label", "neutral"),
            confidence=result.get("confidence", 0.0)
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ Face emotion endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Face analysis error: {str(e)}")


@app.post("/api/tts", response_model=TTSResponse)
async def text_to_speech(request: TTSRequest):
    """
    Synthesize speech from text on demand.
    Used by the frontend play button on every AI message.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=400, detail="Text is required")

    try:
        tts_engine = PiperTTS()
        audio_bytes = await tts_engine.text_to_audio_bytes(request.text.strip())

        if not audio_bytes or len(audio_bytes) < 1000:
            raise HTTPException(status_code=503, detail="TTS produced invalid audio")

        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
        return TTSResponse(audio_base64=audio_base64, audio_type="audio/wav")
    except HTTPException:
        raise
    except Exception as e:
        print(f"❌ TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"TTS error: {str(e)}")


# ==================== Root Endpoint ====================

@app.get("/")
async def root():
    """Root endpoint with API documentation."""
    return {
        "app": APP_TITLE,
        "description": "Mental Health Disorder Assessment API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "create_session": "POST /api/sessions/create",
            "get_session": "GET /api/sessions/{session_id}",
            "therapy": "POST /api/therapy",
            "summary": "POST /api/summary",
            "classify": "POST /api/classify",
            "diagnose": "POST /api/diagnose",
            "diagnose_enhanced": "POST /api/diagnose/enhanced (returns database-ready structured data)",
            "end_session": "POST /api/sessions/{session_id}/end",
            "delete_session": "DELETE /api/sessions/{session_id}",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
