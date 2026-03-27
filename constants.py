"""Constants and configuration for CogniCove Therapy Assistant."""

from dotenv import load_dotenv

# Load API keys
load_dotenv()

# Vectorstore path
DB_FAISS_PATH = "vectorstore/db_faiss"

# Disorder categories
DISORDER_CATEGORIES = [
    "Depressive Disorders",
    "Anxiety Disorder",
    "Trauma- and Stressor-Related Disorders",
    "Bipolar and related Disorder",
    "Obsessive-Compulsive and Related Disorders",
]

# LLM Configuration
LLM_MODEL_NAME = "openai/gpt-oss-120b"
THERAPY_TEMPERATURE = 0.7
DETERMINISTIC_TEMPERATURE = 0.0

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Vectorstore retrieval parameters
OVERVIEW_RETRIEVAL_K = 10
CATEGORY_RETRIEVAL_K = 1000
FALLBACK_RETRIEVAL_K = 50

# App Configuration
APP_TITLE = "CogniCove Therapy Assistant"
APP_PAGE_TITLE = "🧠 CogniCove – AI Therapy Support"

# Classification confidence thresholds
PRIMARY_CONFIDENCE_THRESHOLD = 0.8  # Primary disorder must have confidence > 0.8, otherwise return "None"
COMORBIDITY_CONFIDENCE_THRESHOLD = 0.8  # Only include comorbid disorders with confidence > 0.8
MAX_COMORBIDITIES = 2  # Maximum number of comorbid disorders to report

# Knowledge Graph settings
KG_COMORBIDITY_THRESHOLD = 0.4   # Minimum edge weight to surface a comorbidity
SYMPTOM_MATCH_THRESHOLD = 0.7  # Cosine-similarity floor for symptom matching
