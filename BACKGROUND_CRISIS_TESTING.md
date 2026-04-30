# Background Crisis Detection - Terminal Testing Guide

## Quick Start (2 Minutes)

### Terminal 1: Start FastAPI Server
```bash
cd Chat-CogniCoveModel
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Expected Output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000
Application startup complete
🔹 Loading ELECTRA Suicide Detection Model...
✅ ELECTRA Model loaded successfully!
```

### Terminal 2: Run Test
```bash
python test_background_crisis.py
```

---

## How It Works Now

### Architecture Flow

```
USER MESSAGE
    ↓
┌───────────────────────────────────────────────┐
│  THERAPY PIPELINE (FAST)                      │
│  - Get chat history                           │
│  - Generate therapy response                  │
│  - Return to client immediately (~0.5s)       │
└────────┬──────────────────────────────────────┘
         │
         ├──→ IMMEDIATE RESPONSE TO CLIENT ✅
         │
         └──→ asyncio.create_task()
             ↓
         ┌─────────────────────────────────────┐
         │ BACKGROUND CRISIS DETECTION (async) │
         │ - Run ELECTRA detection             │
         │ - Run LLM analysis                  │
         │ - Print results to terminal         │
         │ - Store in session (~2-3s)          │
         └─────────────────────────────────────┘
```

### Timeline

```
T=0s     Client sends message
T=0.1s   Therapy response starts generating
T=0.5s   ✅ THERAPY RESPONSE RETURNED TO CLIENT
T=0.6s   🔬 ELECTRA detection starts in background
T=0.8s   🤖 LLM detection starts in background
T=2.5s   Crisis detection completes, results printed
T=3.0s   Alert/tool will be called here (you'll add this later)
```

---

## Terminal Output Example

### Test Script Output (Terminal 2)
```
════════════════════════════════════════════════════════════════════════════════
  BACKGROUND CRISIS DETECTION TEST
════════════════════════════════════════════════════════════════════════════════

✅ FastAPI server is running!

════════════════════════════════════════════════════════════════════════════════
📤 SENDING: SUICIDE INTENT (DIRECT)
════════════════════════════════════════════════════════════════════════════════
Message: i can not take it anymore , its getting out of my nerve , this is it...

✅ THERAPY RESPONSE (received in 0.52s)
   AI: I hear that you're experiencing intense pain right now...

📌 NOTE: Crisis detection is running in BACKGROUND!
   Watch the FastAPI server terminal for detection results...
```

### FastAPI Terminal Output (Terminal 1)
**While test is running, you'll see:**

```
[10:30:45] User connected
[10:30:45] Processing therapy request
[10:30:45] ✅ Therapy response generated in 0.52s

────────────────────────────────────────────────────────────────────────────████

[10:30:46] Starting background crisis detection...

[10:30:46]
════════════════════════════════════════════════════════════════════════════════
🔬 ELECTRA SUICIDE DETECTION ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Text: I want to end my life tonight, I can't take this anymore
Prediction: SUICIDE (94.23%)
  → Non-Suicidal: 0.0577
  → Suicidal: 0.9423
Low Confidence Count in Session: 0/5

⚠️  🚨 ALERT TRIGGERED: DIRECT SUICIDE INTENT DETECTED
    Timestamp: 2026-04-25T10:30:46.234567
    Session ID: crisis_test_1234567890
════════════════════════════════════════════════════════════════════════════════

[10:30:47]
════════════════════════════════════════════════════════════════════════════════
🤖 LLM SUICIDE RISK ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Text: I want to end my life tonight, I can't take this anymore
Risk Level: CRITICAL (Confidence: 96.00%)
Indicators: wants to end life, tonight, can't take anymore
Reasoning: Direct, specific, and imminent suicidal ideation with temporal element
Recommendation: Immediate crisis intervention required

⚠️  🚨 ALERT TRIGGERED: LLM: CRITICAL RISK DETECTED
    Timestamp: 2026-04-25T10:30:47.845123
    Session ID: crisis_test_1234567890
════════════════════════════════════════════════════════════════════════════════
```

**Key Observations:**
- ✅ Therapy response came back in **0.52 seconds**
- ✅ Crisis detection started **immediately after** (fire and forget)
- ✅ Results printed to terminal **1-2 seconds later**
- ✅ No blocking of therapy response!

---

## What You're Seeing

### ELECTRA Results
```
🔬 ELECTRA SUICIDE DETECTION ANALYSIS
Text: [truncated message]
Prediction: SUICIDE (94.23%)
  → Non-Suicidal: 0.0577
  → Suicidal: 0.9423
```
- **Prediction**: suicide or non-suicide
- **Confidence %**: How confident the model is
- **Probabilities**: Detailed breakdown

### LLM Results
```
🤖 LLM SUICIDE RISK ANALYSIS
Risk Level: CRITICAL
Indicators: [specific phrases detected]
Reasoning: [clinical assessment]
Recommendation: [action suggested]
```
- **Risk Level**: low/moderate/high/critical
- **Indicators**: Specific warning phrases found
- **Reasoning**: Clinical analysis
- **Recommendation**: What to do next

### Alerts
```
⚠️  🚨 ALERT TRIGGERED: [REASON]
```
Appears when:
- ELECTRA: Direct suicide detection OR 5+ cumulative low-confidence messages
- LLM: High/Critical risk level detected

---

## Safe Message Test

Sending a normal message:
```
Message: I'm having a good day today and feeling better

✅ THERAPY RESPONSE (received in 0.48s)

[Terminal shows:]
════════════════════════════════════════════════════════════════════════════════
🔬 ELECTRA SUICIDE DETECTION ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Text: I'm having a good day today and feeling better
Prediction: NON-SUICIDE (97.45%)
  → Non-Suicidal: 0.9745
  → Suicidal: 0.0255

════════════════════════════════════════════════════════════════════════════════
🤖 LLM SUICIDE RISK ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Risk Level: LOW
Indicators: [none]
Recommendation: Standard therapeutic support

════════════════════════════════════════════════════════════════════════════════
```

No alert = safe message ✅

---

## Response Time Expectations

| Message Type | Response Time | Crisis Detection |
|--------------|---------------|------------------|
| Normal | 0.4-0.8s | 1-2s after |
| Suicide/High Risk | 0.4-0.8s | 1-3s after |
| Many messages (cumulative) | 0.4-0.8s | Tracks over time |

Key: **Therapy always stays fast < 1s** 🚀

---

## Where to Add Emergency Function Later

When you're ready to replace terminal output with actual alerts:

**Current (Terminal Output):**
```python
# In suicide_detection.py - print_result()
print(f"\n⚠️  🚨 ALERT TRIGGERED: {result['alert_reason']}")
```

**Future (Emergency Function):**
```python
# Add in background crisis detection
if electra_result['is_alert'] or llm_result['is_alert']:
    await call_emergency_service(
        session_id=session_id,
        message=user_text,
        electra_alert=electra_result['is_alert'],
        llm_alert=llm_result['is_alert'],
        risk_level=llm_result.get('risk_level')
    )
```

---

## Test Script Sections

### 1. Test Direct Suicide
```python
test_message(
    "I want to end my life tonight, I can't take this anymore",
    "SUICIDE INTENT (DIRECT)"
)
```
→ Should trigger both ELECTRA and LLM alerts

### 2. Test Normal Message
```python
test_message(
    "I'm having a good day today and feeling better",
    "NORMAL MESSAGE"
)
```
→ Should show low risk for both

### 3. Test Indirect Suicide
```python
test_message(
    "I don't see the point of living anymore",
    "SUICIDE INTENT (INDIRECT)"
)
```
→ May trigger LLM but depends on ELECTRA confidence

---

## Verification Checklist

- [ ] FastAPI server starts without errors
- [ ] ELECTRA model loads successfully
- [ ] Test script runs
- [ ] Therapy response returns in < 1 second
- [ ] Crisis detection starts immediately after (fire and forget)
- [ ] Terminal shows ELECTRA results within 1-2 seconds
- [ ] Terminal shows LLM results within 2-3 seconds
- [ ] Alerts trigger on suicide messages
- [ ] No alerts on safe messages
- [ ] Session stores both detections

---

## Terminal Monitoring Tips

### Watch for These Signs:

**Good Signs ✅**
- Therapy response < 1s
- Detections appear 1-3s after response
- Both ELECTRA and LLM results present
- Alerts only on dangerous messages

**Bad Signs ❌**
- Therapy response > 2s (blocking on crisis detection)
- Only ELECTRA, no LLM results
- Detections never appear (background task not running)
- Alerts on every message (too sensitive)

---

## How to View Session Data

Get all crisis detections for a session:
```bash
curl http://localhost:8000/api/sessions/crisis_test_1234567890 | jq '.suicide_detections'
```

Example output:
```json
[
  {
    "timestamp": "2026-04-25T10:30:46.234567",
    "user_message": "I want to end my life...",
    "electra_result": {
      "prediction": "suicide",
      "confidence": 0.9423,
      "is_alert": true,
      "alert_reason": "DIRECT SUICIDE INTENT DETECTED"
    },
    "llm_result": {
      "risk_level": "critical",
      "confidence": 0.96,
      "is_alert": true,
      "alert_reason": "LLM: CRITICAL RISK DETECTED"
    }
  }
]
```

---

## Next: Replace Terminal Output

Once you verify terminal detection works, you'll replace the terminal output in `suicide_detection.py`:

```python
# Add emergency function placeholder
async def trigger_emergency_alert(detection_result, session_id):
    """Called when crisis is detected"""
    # This is where your emergency call function will go
    print(f"[PLACEHOLDER] Emergency alert would trigger here")
    # await emergency_call_service(...)  # Add this later
```

Then update background detection:
```python
if electra_result['is_alert'] or llm_result['is_alert']:
    asyncio.create_task(trigger_emergency_alert(...))
```

---
