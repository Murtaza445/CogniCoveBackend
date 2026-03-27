# CogniCove Therapy Assistant - Code Structure

## Project Organization

The application is now organized into modular, reusable components:

### 📁 File Structure

```
Disorder_chat/
├── frontfyp.py           (Entry point - run with: streamlit run frontfyp.py)
├── ui.py                 (Streamlit UI components and main app logic)
├── chains.py             (LangChain chain definitions)
├── models.py             (Pydantic data models)
├── vectorstore.py        (FAISS vectorstore utilities)
├── constants.py          (Configuration and constants)
├── disorders_chunks.py   (Data chunks - existing)
├── build_disorder_vectorstores.py (Build script - existing)
└── requirements.txt      (Dependencies)
```

---

## 🔧 Module Descriptions

### **`frontfyp.py`** (Entry Point)
- Minimal entry point that imports and runs the app
- Execute with: `streamlit run frontfyp.py`

### **`ui.py`** (Streamlit UI & App Logic)
- All Streamlit UI components and rendering logic
- Session state management
- Main application flow orchestration
- Functions:
  - `initialize_session_state()` — Initialize session variables
  - `reset_session()` — Clear session for new session
  - `display_sidebar_controls()` — Render sidebar buttons
  - `display_chat_history()` — Show chat messages
  - `handle_therapy_response()` — Process user input and therapy response
  - `generate_clinical_summary()` — Extract summary from conversation
  - `classify_disorder_category()` — Classify and diagnose
  - `display_results()` — Show summary, classification, diagnosis
  - `main()` — Main Streamlit app

### **`chains.py`** (LangChain Chains)
- All LangChain chain definitions
- Temperature and model configuration managed here
- Functions:
  - `get_therapy_chain()` — Therapeutic conversation chain
  - `get_summary_chain()` — Clinical summary extraction chain
  - `get_classification_chain()` — Disorder category classification chain
  - `get_diagnosis_chain()` — Diagnostic results generation chain

### **`models.py`** (Pydantic Models)
- Data validation and serialization models
- Classes:
  - `SummaryModel` — Clinical summary schema
  - `ClassificationModel` — Category classification schema
  - `DiagnosticResults` — Diagnostic assessment schema

### **`vectorstore.py`** (FAISS Utilities)
- Vectorstore loading and caching
- Metadata-aware retrieval with fallbacks
- Functions:
  - `get_vectorstore()` — Load cached FAISS index
  - `retrieve_overview_chunks()` — Get overview/criteria docs
  - `retrieve_category_chunks()` — Get category-specific docs

### **`constants.py`** (Configuration)
- All constants, settings, and configurations
- Model names, temperatures, retrieval parameters
- Disorder categories list
- Paths and API settings

---

## 🔄 Application Flow

```
User Opens App
    ↓
Initialize Session State → Load Vectorstore (cached)
    ↓
Display Chat Interface
    ↓
User Types Message
    ↓
[THERAPY CHAIN] → Therapy response (multi-turn)
    ↓
User Clicks "End Session"
    ↓
[SUMMARY CHAIN] → Extract JSON summary (Pydantic validated)
    ↓
[CLASSIFICATION CHAIN] → Get overview chunks → Classify category
    ↓
[DIAGNOSIS CHAIN] → Get category chunks → Generate diagnostic results
    ↓
Display Results:
  - 📋 Clinical Summary (JSON)
  - 🏷️ Disorder Classification (JSON)
  - 📊 Diagnostic Results (JSON + Metrics)
```

---

## 🚀 Running the App

### Normal Mode
```bash
streamlit run frontfyp.py
```

### Debug Mode
```bash
streamlit run frontfyp.py --logger.level=debug
```

---

## 📦 Dependencies

See `requirements.txt` for full list. Key packages:
- `streamlit` — Web UI framework
- `langchain` — LLM orchestration
- `langchain-groq` — Groq API integration
- `langchain-huggingface` — Embeddings
- `faiss-cpu` — Vector search
- `pydantic` — Data validation

---

## 🔧 Configuration

Edit `constants.py` to change:
- LLM model and temperatures
- Embedding model
- Vectorstore path
- Retrieval k values
- Disorder categories
- App titles

---

## 🎯 Benefits of This Structure

✅ **Modular** — Each component has a single responsibility  
✅ **Reusable** — Import functions in other projects  
✅ **Testable** — Easy to unit test individual components  
✅ **Maintainable** — Changes isolated to relevant files  
✅ **Scalable** — Add new features without touching existing code  
✅ **Readable** — Clear separation of concerns  

---

## 📝 Adding New Features

### Add a new LLM chain?
→ Add to `chains.py`, import in `ui.py`

### Change model temperature?
→ Edit `constants.py`

### Add a new Pydantic model?
→ Add to `models.py`

### Change UI layout?
→ Edit functions in `ui.py`

### Add new retrieval logic?
→ Add function to `vectorstore.py`

---

## 📧 Support

For questions about the code structure or any other issues, refer to the module docstrings and inline comments.
