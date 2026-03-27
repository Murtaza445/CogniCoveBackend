"""FastAPI backend for Disorder Chat application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List
import json
import re
from datetime import datetime
from pydantic import ValidationError

from chains import get_therapy_chain, get_summary_chain, get_classification_chain, get_diagnosis_chain
from models import SummaryModel, ClassificationModel, DiagnosticResults
from vectorstore import get_vectorstore
from constants import DISORDER_CATEGORIES, APP_TITLE, APP_PAGE_TITLE
from hybrid_retrieval import hybrid_classify, hybrid_diagnose
from knowledge_graph import build_knowledge_graph
from langchain_core.messages import HumanMessage, AIMessage


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


# ==================== Request/Response Models ====================

class MessageRequest(BaseModel):
    """Request model for therapy messages."""
    session_id: str
    content: str


class TherapyResponse(BaseModel):
    """Response model for therapy message."""
    session_id: str
    user_message: str
    assistant_message: str
    timestamp: str


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


class DiagnoseResponse(BaseModel):
    """Response with diagnostic results."""
    session_id: str
    diagnostic_results: List[str]
    timestamp: str


class SessionEndRequest(BaseModel):
    """Request to end session."""
    session_id: str


class SessionEndResponse(BaseModel):
    """Response when session ends."""
    session_id: str
    summary: Optional[str] = None
    classification: Optional[str] = None
    diagnostic_results: Optional[List[str]] = None
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


@app.post("/api/sessions/{session_id}", response_model=SessionData)
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
    """Process therapy message and generate response."""
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
        # Get therapy chain and build chat history
        therapy_chain = get_therapy_chain()
        chat_history = build_chat_history_from_session(request.session_id)
        
        # Generate response
        response = therapy_chain.invoke({
            "chat_history": chat_history,
            "query": request.content
        })
        response_text = response.content
        
        # Add assistant message
        assistant_msg = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat()
        }
        session["messages"].append(assistant_msg)
        
        return TherapyResponse(
            session_id=request.session_id,
            user_message=request.content,
            assistant_message=response_text,
            timestamp=assistant_msg["timestamp"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating therapy response: {str(e)}")


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
        cls_validated = hybrid_classify(request.summary)
        
        if cls_validated:
            classification_json = cls_validated.model_dump_json(indent=2)
            comorbid_categories = cls_validated.comorbid_categories or []
        else:
            classification_json = "Unable to classify."
            comorbid_categories = []
        
        session["classification"] = classification_json
        
        return ClassifyResponse(
            session_id=request.session_id,
            classification=classification_json,
            comorbid_categories=comorbid_categories,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error classifying disorder: {str(e)}")


@app.post("/api/diagnose", response_model=DiagnoseResponse)
async def generate_diagnosis(request: DiagnoseRequest):
    """Generate diagnostic results for detected categories."""
    if request.session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[request.session_id]
    
    # Skip if already diagnosed
    if session.get("diagnostic_results") is not None:
        return DiagnoseResponse(
            session_id=request.session_id,
            diagnostic_results=session["diagnostic_results"],
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
            diagnostic_results = results if results else []
        else:
            diagnostic_results = []
        
        session["diagnostic_results"] = diagnostic_results
        
        return DiagnoseResponse(
            session_id=request.session_id,
            diagnostic_results=diagnostic_results,
            timestamp=datetime.utcnow().isoformat()
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating diagnosis: {str(e)}")


@app.post("/api/sessions/{session_id}/end", response_model=SessionEndResponse)
async def end_session(session_id: str):
    """End a session and return all results."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions_store[session_id]
    session["session_ended"] = True
    
    return SessionEndResponse(
        session_id=session_id,
        summary=session.get("summary"),
        classification=session.get("classification"),
        diagnostic_results=session.get("diagnostic_results"),
        ended_at=datetime.utcnow().isoformat()
    )


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions_store:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions_store[session_id]
    return {"message": f"Session {session_id} deleted"}


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
            "end_session": "POST /api/sessions/{session_id}/end",
            "delete_session": "DELETE /api/sessions/{session_id}",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
