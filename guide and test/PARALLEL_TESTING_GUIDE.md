# Parallel Processing Test Guide - Crisis Detection

## Overview
This guide shows you how to verify that all processing (therapy, summary, disorder classification, and crisis detection) happen in **parallel**, not sequentially.

## Quick Start

### 1. Start the FastAPI Server
```bash
cd Chat-CogniCoveModel
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Run the Test Script
```bash
python test_parallel_processing.py
```

## Understanding Parallel vs Sequential Execution

### Sequential (SLOW - NOT WHAT WE WANT)
```
User Message
    ↓
Therapy Response:      ████████ (2 sec)
    ↓
Summary Generation:    ██████████████ (3 sec)
    ↓
Crisis Detection:      ██████████ (2.5 sec)
    ↓
Total:                 ~7.5 seconds ❌
```

### Parallel (FAST - WHAT WE WANT)
```
User Message
    ↓
│ Therapy:             ██████████ (2 sec)
├ Summary:            ██████████████ (3 sec)
├ Crisis ELECTRA:      ████ (0.5 sec)
└ Crisis LLM:          ██████████████ (3 sec)
    ↓
Total:                 3 seconds (max of all) ✅
```

## Test Results Interpretation

### Test 1: Text Therapy Endpoint Parallel Processing

**What it does:**
- Sends 3 different messages (suicide, normal distress, indirect suicide)
- Measures response time for each
- Shows crisis detection results

**What to look for:**
```
✅ Response received in 1.50s    ← Fast = parallel
✅ Response received in 2.80s    ← Reasonable = parallel
✅ Response received in 2.95s    ← Good = parallel

❌ Response received in 6.20s    ← Slow = probably sequential
```

**Success Criteria:**
- Response times are 1-3 seconds
- Both ELECTRA and LLM results present in crisis detection
- Therapy response is substantive

### Test 2: Parallel Timing Verification

**Sequential Expected Times:**
```
ELECTRA Detection:    ~100-200ms
LLM Detection:        ~1000-3000ms
Emotion Analysis:     ~500-1000ms
Therapy Generation:   ~1000-2000ms
────────────────────────────────
TOTAL (Sequential):   ~3600-6200ms ❌
```

**Parallel Expected Times:**
```
All tasks run simultaneously:
TOTAL (Parallel):     ~1000-3000ms ✅
(= time of slowest task)
```

**Example Output:**
```
Request 1: ✅ 1.85s
Request 2: ✅ 2.10s
Request 3: ✅ 1.95s

Average Response Time: 1.97s
✅ LIKELY PARALLEL (fast response)
```

### Test 3: Verify Both Detections

**Output Structure:**
```
🔬 ELECTRA Detection: ✅ Present
   Prediction: suicide
   Alert: True

🤖 LLM Detection: ✅ Present
   Risk Level: critical
   Alert: True

✅ BOTH DETECTIONS WORKING!
```

**What this means:**
- ✅ Both models are running
- ✅ Both store results in session
- ✅ Both contribute to crisis alerting

### Test 4: Terminal Output Verification

**Watch the FastAPI Terminal While Testing**

#### Good Sign (Parallel) ✅
```
2026-04-25 10:30:45.123 - 🔬 ELECTRA SUICIDE DETECTION ANALYSIS
2026-04-25 10:30:45.150 - 🤖 LLM SUICIDE RISK ANALYSIS  (starts almost immediately)
2026-04-25 10:30:45.200 - Therapy Response: [generating...]
2026-04-25 10:30:47.100 - All responses ready
```
→ Output starts almost simultaneously = PARALLEL ✅

#### Bad Sign (Sequential) ❌
```
2026-04-25 10:30:45.123 - 🔬 ELECTRA SUICIDE DETECTION...
2026-04-25 10:30:45.500 - ✅ ELECTRA complete
2026-04-25 10:30:45.510 - 🤖 LLM SUICIDE RISK...
2026-04-25 10:30:48.200 - ✅ LLM complete
2026-04-25 10:30:48.210 - Therapy Response [generating...]
2026-04-25 10:30:50.100 - ✅ Therapy complete
```
→ Output in strict order = SEQUENTIAL ❌

## Test Session Data Structure

Each test creates data like this:

```json
{
  "session_id": "test_session_1234567890",
  "messages": [
    {
      "role": "user",
      "content": "I want to end my life"
    },
    {
      "role": "assistant",
      "content": "I hear that you're in pain..."
    }
  ],
  "suicide_detections": [
    {
      "timestamp": "2026-04-25T10:30:45.123Z",
      "user_message": "I want to end my life",
      "electra_result": {
        "prediction": "suicide",
        "confidence": 0.92,
        "is_alert": true,
        "alert_reason": "DIRECT SUICIDE INTENT DETECTED"
      },
      "llm_result": {
        "risk_level": "critical",
        "confidence": 0.94,
        "indicators": ["wants to end life", "direct statement"],
        "is_alert": true,
        "alert_reason": "LLM: CRITICAL RISK DETECTED"
      }
    }
  ]
}
```

## Key Metrics to Check

### Response Time
| Range | Interpretation |
|-------|-----------------|
| < 2.0s | Excellent parallel (LLM + all others) |
| 2.0-3.0s | Good parallel |
| 3.0-4.0s | Borderline (check terminal output) |
| > 4.0s | Likely sequential OR slow inference |

### Crisis Detection Presence
```
ELECTRA Only:           ⚠️ One detector working
LLM Only:              ⚠️ One detector working
Both Present:          ✅ Dual detection working
Neither:               ❌ Crisis detection broken
```

### Alert Triggers
```
Both trigger on same message:      ✅ Independent detection
Only one triggers:                 ⚠️ Check thresholds
Neither trigger (should):          ❌ Check model path
```

## Manual Testing with cURL

### Test Text Endpoint
```bash
curl -X POST http://localhost:8000/api/therapy \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "manual_test_1",
    "content": "I dont see the point of living anymore"
  }' | jq .
```

**Measure time to response (should be 1-3 seconds)**

### View Session Data
```bash
curl http://localhost:8000/api/sessions/manual_test_1 | jq '.suicide_detections'
```

### Test Audio Endpoint
```bash
curl -X POST http://localhost:8000/api/therapy/audio \
  -F "session_id=audio_test_1" \
  -F "file=@path/to/audio.wav"
```

**Measure time to response (should be 2-4 seconds including STT)**

## Troubleshooting

### Problem: Response times > 5 seconds
**Possible causes:**
1. Sequential execution (check code)
2. GPU not available (check terminal for "Device: cpu")
3. ELECTRA model file path issues
4. LLM API timeout

**Check:**
```bash
# Find response in terminal looking for:
Device: cuda  # If GPU available
Device: cpu   # If only CPU (slower)
```

### Problem: Only ELECTRA detection, no LLM
**Possible causes:**
1. LLM API authentication issue
2. Groq API key missing/expired
3. Rate limiting

**Fix:**
```bash
# Check .env file
cat .env | grep GROQ
```

### Problem: Inconsistent response times
**Possible causes:**
1. Network latency
2. System load
3. LLM API response variance

**Solution:**
Run test multiple times, calculate average

## Expected Flow with All Components

### Text Endpoint: `/api/therapy`
```
Request: User Message
    ↓
[PARALLEL SECTION START]
├── ELECTRA Detection      → ELECTRA output (visible in terminal)
├── LLM Detection          → LLM output (visible in terminal)
├── Therapy Generation     → Therapy output (visible in terminal)
└── (Optional) Summary     → If requested via /api/summary
[PARALLEL SECTION END]
    ↓
Response: {
  "user_message": "...",
  "assistant_message": "...",
  "timestamp": "..."
}
    ↓
Store in Session:
- Therapy message
- Both detections (ELECTRA + LLM)
- Crisis alerts if triggered
```

### Audio Endpoint: `/api/therapy/audio`
```
Request: Audio File
    ↓
Single Task: STT (speech to text)
    ↓
[PARALLEL SECTION START]
├── ELECTRA Detection
├── LLM Detection  
├── Emotion Analysis
└── Therapy Generation
[PARALLEL SECTION END]
    ↓
Single Task: TTS (text to speech)
    ↓
Response: {
  "user_text": "...",
  "ai_response": "...",
  "emotion": "...",
  "audio_base64": "..."
}
```

## Advanced Monitoring

### Watch Terminal in Real-time
```bash
# Terminal 1: Start server
uvicorn main:app --reload

# Terminal 2: Run test while watching Terminal 1
python test_parallel_processing.py
```

### Check Request/Response Times in Code
Edit `test_parallel_processing.py` to add more granular timing:
```python
start = time.time()
response = requests.post(...)  # Make request
end = time.time()
print(f"Network time: {end - start:.2f}s")
```

## Success Checklist

- [ ] Test script runs without errors
- [ ] Response times are 1-3 seconds
- [ ] Both ELECTRA and LLM results present in detections
- [ ] Terminal shows overlapping output (not sequential)
- [ ] Therapy response is generated
- [ ] Session data stores both detections
- [ ] Alerts trigger on suicide messages
- [ ] Average time < 3 seconds = PARALLEL ✅

## Next Steps

Once parallel execution is verified:
1. Integrate emergency call tool when `is_alert == True`
2. Set up logging for audit trail
3. Add database persistence for detections
4. Implement user notification system
