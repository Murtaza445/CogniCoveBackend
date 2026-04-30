# ⚡ Quick Start: Test Crisis Detection Terminal Output

## 1️⃣ Start FastAPI Server

Open **Terminal 1**:
```bash
cd Chat-CogniCoveModel
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Wait for this output:**
```
INFO:     Uvicorn running on http://0.0.0.0:8000 [Press ENTER to continue]
🔹 Loading ELECTRA Suicide Detection Model...
✅ ELECTRA Model loaded successfully!
Application startup complete
```

---

## 2️⃣ Run Test in Second Terminal

Open **Terminal 2**:
```bash
python test_background_crisis.py
```

---

## 3️⃣ Watch Terminal 1 (FastAPI)

As test runs, you'll see crisis detection results appear in **Terminal 1** (NOT Terminal 2):

```
[10:30:46]
════════════════════════════════════════════════════════════════════════════════
🔬 ELECTRA SUICIDE DETECTION ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Text: I want to end my life tonight, I can't take this anymore
Prediction: SUICIDE (94.23%)
⚠️  🚨 ALERT TRIGGERED: DIRECT SUICIDE INTENT DETECTED

════════════════════════════════════════════════════════════════════════════════
🤖 LLM SUICIDE RISK ANALYSIS
════════════════════════════════════════════════════════════════════════════════
Risk Level: CRITICAL
⚠️  🚨 ALERT TRIGGERED: LLM: CRITICAL RISK DETECTED
════════════════════════════════════════════════════════════════════════════════
```

---

## 📊 Expected Results

| Test | Therapy Response | Crisis Detection |
|------|------------------|------------------|
| Suicide (direct) | ~0.5s | ✅ Both alert |
| Normal message | ~0.5s | ✅ No alert |
| Suicide (indirect) | ~0.5s | ✅ LLM alerts |

---

## 🎯 What This Shows

✅ **Therapy returns in < 1 second**
✅ **Crisis detection runs in background**
✅ **Terminal shows results 1-3 seconds after response**
✅ **No blocking of user response!**

---

## Next Step

Once verified, you'll replace terminal output with:
```python
# In suicide_detection.py - add this function
async def trigger_emergency_alert(detection_result):
    """Your emergency call function goes here"""
    # await emergency_service.call(...)
    pass
```

Then update main.py:
```python
if electra_result['is_alert'] or llm_result['is_alert']:
    asyncio.create_task(trigger_emergency_alert(...))
```

---
