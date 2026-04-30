# Quick Start: Testing Parallel Processing

## 3-Step Verification

### Step 1: Start Server
```bash
cd Chat-CogniCoveModel
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```
**Terminal Output Should Show:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
INFO:     Application startup complete
🔹 Loading ELECTRA Suicide Detection Model...
✅ ELECTRA Model loaded successfully!
```

### Step 2: Run Test in Another Terminal
```bash
cd Chat-CogniCoveModel
python test_parallel_processing.py
```

### Step 3: Verify Results

#### Response Time Check
```
✅ PASS: Average Response Time: 1.85s
✅ PASS: LIKELY PARALLEL (fast response)

❌ FAIL: Average Response Time: 5.50s  
❌ FAIL: POSSIBLY SEQUENTIAL (slow response)
```

#### Crisis Detection Check
```
✅ PASS:
🔬 ELECTRA Detection: ✅ Present
🤖 LLM Detection: ✅ Present
✅ BOTH DETECTIONS WORKING!

❌ FAIL:
🔬 ELECTRA Detection: ✅ Present
🤖 LLM Detection: ❌ Missing
❌ One or both detections missing!
```

---

## What Happens in Parallel

```
┌─────────────────────────────────────────────────┐
│            TEXT THERAPY REQUEST                 │
│     "I want to end my life tonight"             │
└────────────────┬────────────────────────────────┘
                 │
     ┌───────────┴───────────┐
     │  PARALLEL PROCESSING  │
     │                       │
  ┌──▼──────┐  ┌───────┐  ┌─▼────────┐  ┌────────┐
  │ ELECTRA │  │  LLM  │  │ THERAPY  │  │SUMMARY │
  │Detection│  │Risk   │  │ LLM      │  │Builder │
  │         │  │ Analysis   │          │  │(optional)
  │ 0.2s    │  │ 2.8s      │ 1.5s     │  │ 2.0s   │
  └──┬──────┘  └───┬───┘  └──┬──────┘  └────┬────┘
     │             │          │             │
     └─────────────┼──────────┼─────────────┘
                   │          │
            3 seconds max →────┴────→ All results ready!
                                    (NOT 5-6 seconds)
```

---

## Real Example Output

### Running the Test
```bash
$ python test_parallel_processing.py

════════════════════════════════════════════════════════════════════════════════
🧪 TEST 1: TEXT THERAPY ENDPOINT - PARALLEL PROCESSING
════════════════════════════════════════════════════════════════════════════════

📌 Testing: suicide_direct
────────────────────────────────────────────────────────────────────────────────
Message: I am done with this life, I can't take it anymore.

✅ Response received in 1.85s
   User: I am done with this life, I can't take it anymore....
   AI: I hear intense pain in what you've shared, and I want you to know...
   Timestamp: 2026-04-25T10:30:45.123456

📊 Crisis Detection Results:

   Detection #1:
   Timestamp: 2026-04-25T10:30:45.100000
   🔬 ELECTRA: SUICIDE
      Confidence: 92.34%
      ⚠️  ALERT: DIRECT SUICIDE INTENT DETECTED
   🤖 LLM: CRITICAL
      Confidence: 94.67%
      Indicators: wants to end life, immediate
      ⚠️  ALERT: LLM: CRITICAL RISK DETECTED
```

---

## Terminal Watchers' Guide

### Watch FastAPI Terminal While Testing

#### ✅ GOOD (Parallel Execution)
```
[10:30:45.100] 🔬 ELECTRA SUICIDE DETECTION ANALYSIS
[10:30:45.105] Text: I am done with this life...
[10:30:45.110] 🤖 LLM SUICIDE RISK ANALYSIS         ← Starts almost immediately!
[10:30:45.115] Risk Level: CRITICAL
[10:30:45.500] Therapy: I hear that you're in pain...
[10:30:47.100] All responses complete
────────────────────────────────────────────
Total time: ~2.0 seconds ✅
```

#### ❌ BAD (Sequential Execution)
```
[10:30:45.100] 🔬 ELECTRA SUICIDE DETECTION ANALYSIS
[10:30:45.500] ✅ ELECTRA complete
[10:30:45.510] 🤖 LLM SUICIDE RISK ANALYSIS         ← Waits for ELECTRA!
[10:30:48.200] ✅ LLM complete
[10:30:48.210] Therapy: I hear that you're...       ← Waits for LLM!
[10:30:50.100] All responses complete
────────────────────────────────────────────
Total time: ~5.0 seconds ❌
```

---

## Component Response Times (Expected)

| Component | Time | Notes |
|-----------|------|-------|
| ELECTRA | 100-200ms | Fast ML model |
| LLM (Groq) | 1000-3000ms | Slowest, varies with load |
| Therapy Generation | 1000-2000ms | Depends on context |
| Emotion Analysis | 500-1000ms | Audio processing |
| Summary (optional) | 1500-2500ms | Full conversation analysis |

**Sequential Total:** 5000-8500ms ❌  
**Parallel Total:** 1000-3000ms ✅

---

## Quick Troubleshooting

### Issue: Response > 4 seconds
```
Check terminal output:
- Do ELECTRA and LLM start at same time? 
  YES ✅ → Parallel working, just slow inference
  NO ❌  → Sequential execution detected
```

### Issue: No LLM results
```
Check .env file:
- GROQ_API_KEY set?
  YES ✅ → Maybe API rate limit
  NO ❌  → Add key to .env

Check terminal:
- Any GROQ API errors?
  NO ERRORS ✅ → Try again
  ERRORS ❌ → API issue
```

### Issue: Only ELECTRA, no LLM  
```
1. Verify model path:
   Crisis Detection Model/electra_suicidal_text_detector/
   
2. Try single request again:
   curl -X POST http://localhost:8000/api/therapy \
     -H "Content-Type: application/json" \
     -d '{"session_id":"test","content":"test message"}'
```

---

## Test Message Examples

### For Crisis Detection Testing
```
Suicide Intent (Direct):
"I want to end my life tonight"
"I'm going to kill myself"

Suicide Intent (Indirect):
"I don't see the point anymore"
"Everything is pointless"

Normal/Safe:
"I'm having a good day"
"I love my family"
```

---

## Key Success Indicators

| Indicator | Status | Meaning |
|-----------|--------|---------|
| Response: 1-3s | ✅ | Parallel execution working |
| Response: 4+s | ❌ | Check for sequential execution |
| Both detections | ✅ | Dual model working |
| One detection | ❌ | One model failing |
| Alerts triggered | ✅ | Crisis detection active |
| Terminal output: parallel | ✅ | asyncio working |
| Terminal output: sequential | ❌ | Check asyncio.gather() |

---

## Next Step: Emergency Integration

Once parallel execution verified:

```python
# In your endpoint
if electra_result['is_alert'] or llm_result['is_alert']:
    # Trigger emergency tool
    emergency_response = await call_emergency_service(
        session_id=session_id,
        user_message=request.content,
        electra_alert=electra_result['is_alert'],
        llm_alert=llm_result['is_alert']
    )
```

---

## Debug Command

View all detections for a session:
```bash
curl http://localhost:8000/api/sessions/test_session_123 | jq '.suicide_detections'
```

Expected output:
```json
[
  {
    "timestamp": "2026-04-25T10:30:45.123Z",
    "user_message": "I want to end my life",
    "electra_result": { ... },
    "llm_result": { ... }
  }
]
```
