# Suicide Detection Integration Guide

## Overview
The suicide detection system integrates the ELECTRA model into your Chat-CogniCove application to detect suicidal intent in real-time during therapy conversations. The system has two alert mechanisms:

1. **Direct Detection**: High confidence prediction of suicidal intent
2. **Cumulative Tracking**: Alerts after 5+ messages with low confidence in "non-suicidal" classification

## Features

### ✅ What's Implemented
- **ELECTRA Model Integration**: Uses fine-tuned ELECTRA model for binary classification (suicide/non-suicide)
- **Session-Based Tracking**: Tracks confidence scores per session
- **Alert System**: Triggers alerts when:
  - Suicide intent is detected (confidence > 50%)
  - 5+ messages have low confidence of "not suicide" (< 60% confidence)
- **Terminal Output**: Formatted, color-coded output for monitoring
- **Automatic Cleanup**: Resets session tracking when session ends

### 🔄 Message Flow
```
User Message
    ↓
[Added to session]
    ↓
[Suicide Detection Analysis]
    ├─ Prediction (suicide/non-suicide)
    ├─ Confidence Score
    ├─ Probabilities (detailed breakdown)
    └─ Alert Check (if triggered)
    ↓
[Print Result to Terminal]
    ↓
[Continue with normal therapy response]
```

## File Structure

### New/Modified Files
```
Chat-CogniCoveModel/
├── suicide_detection.py          [NEW] Main detection module
├── test_suicide_detection.py      [NEW] Test/demo script
├── main.py                        [MODIFIED] Integrated detection into therapy endpoint
└── SUICIDE_DETECTION.md           [NEW] This file
```

### Required Model Files
```
Crisis Detection Model/
└── electra_suicidal_text_detector/
    ├── config.json
    ├── pytorch_model.bin
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

## Usage

### 1. Initialize Detector (Automatic in FastAPI)
The detector is automatically initialized when `main.py` starts:
```python
suicide_detector = get_suicide_detector()
```

### 2. Process Messages
Messages are automatically checked when sent to `/api/therapy`:
```python
# The endpoint automatically calls:
suicide_result = suicide_detector.predict(user_message, session_id=session_id)
suicide_detector.print_result(suicide_result, session_id=session_id)
```

### 3. Response Data
Each message stores detection results in the session:
```python
session['suicide_detections'] = [
    {
        'timestamp': '2026-04-23T...',
        'user_message': 'text...',
        'detection_result': {
            'prediction': 'suicide|non-suicide',
            'confidence': 0.85,
            'probabilities': {
                'non-suicidal': 0.15,
                'suicidal': 0.85
            },
            'is_alert': True,
            'alert_reason': 'DIRECT SUICIDE INTENT DETECTED'
        }
    }
]
```

### 4. Testing
Run the test script to verify functionality:
```bash
python test_suicide_detection.py
```

Expected output shows:
- ✓ Direct detection tests
- ✓ Cumulative tracking tests  
- ✓ Mixed message handling
- ✓ Alert conditions

## Terminal Output Example

### Successful Detection
```
================================================================================
🔍 SUICIDE DETECTION ANALYSIS
================================================================================
Text: I am done with this life, I cannot take it anymore.
Prediction: SUICIDE (85.23%)
  → Non-Suicidal: 0.1477
  → Suicidal: 0.8523
Low Confidence Count in Session: 0/5

⚠️  🚨 ALERT TRIGGERED: DIRECT SUICIDE INTENT DETECTED
    Timestamp: 2026-04-23T10:30:45.123456
    Session ID: session_123
================================================================================
```

### Low Confidence Accumulation
```
================================================================================
🔍 SUICIDE DETECTION ANALYSIS
================================================================================
Text: It would be better if I just disappeared.
Prediction: NON-SUICIDE (42.31%)
  → Non-Suicidal: 0.4231
  → Suicidal: 0.5769
Low Confidence Count in Session: 5/5

⚠️  🚨 ALERT TRIGGERED: MULTIPLE LOW CONFIDENCE MESSAGES (5 messages with <0.6 confidence)
    Timestamp: 2026-04-23T10:32:15.654321
    Session ID: session_123
================================================================================
```

## Configuration

### Thresholds (in `suicide_detection.py`)
```python
self.LOW_CONFIDENCE_THRESHOLD = 0.6      # Below this = concerning
self.LOW_CONFIDENCE_ALERT_COUNT = 5      # Alert after 5 messages
self.ALERT_REASON_SUICIDE = 0.5          # Direct alert threshold
```

To modify thresholds:
```python
detector = get_suicide_detector()
detector.LOW_CONFIDENCE_THRESHOLD = 0.7  # 70% confidence required
detector.LOW_CONFIDENCE_ALERT_COUNT = 3  # Alert after 3 messages
```

## Next Steps: Emergency Call Tool

### Integration Point
When `is_alert == True`, your future emergency call tool can:

```python
if suicide_result['is_alert']:
    # Call emergency response tool
    emergency_tool = get_emergency_response_tool()
    emergency_tool.initiate_call(
        session_id=session_id,
        alert_type=suicide_result['alert_reason'],
        user_message=suicide_result['text'],
        confidence=suicide_result['confidence']
    )
```

### Recommended Emergency Tool Features
- SMS notification to emergency contact
- Automated phone call with crisis resources
- Alert to therapy provider
- Session escalation flag
- Incident logging

## Error Handling

### If Model Not Found
```
⚠️  WARNING: Model not found at [path]
   Suicide detection will be disabled
   Please ensure electra_suicidal_text_detector is in Crisis Detection Model/
```

Solution:
1. Ensure model directory exists with all files
2. Check directory permissions
3. Run with absolute path: `python main.py`

### If CUDA Not Available
The system automatically falls back to CPU:
```
Device: cpu
```
(GPU recommended for production, but CPU works fine)

## API Integration

### For Next.js Frontend
No frontend changes needed! Detection happens server-side.

The detection results are stored in session but not returned in `TherapyResponse` to avoid exposing sensitive detection data.

To access detection history:
```python
# GET /api/sessions/{session_id}
# Check: session.suicide_detections[]
```

## Performance

### Model Loading
- **One-time**: ~5 seconds on first request
- **Per-message**: ~50-200ms (depends on message length, GPU/CPU)

### Memory Usage
- **Model**: ~400MB (GPU) or ~600MB (CPU)
- **Session tracking**: <1MB per session

## Security & Privacy

✅ **What's Not Shared**
- Detection results are NOT sent to frontend
- Only console output for monitoring
- Results stored in server memory only

⚠️ **For Production**
- Implement database logging for audit trail
- Add access controls to detection history
- Encrypt sensitive fields in database
- Consider HIPAA/GDPR compliance for healthcare data

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Model not loading | Check Crisis Detection Model path |
| Predictions seem wrong | Verify model files are complete |
| Slow processing | Check CPU/GPU availability |
| High false positives | Adjust thresholds (see Configuration) |
| Session not resetting | Ensure `/api/sessions/{id}/end` is called |

## References

- **Model**: ELECTRA (https://huggingface.co/google/electra-base)
- **Task**: Binary text classification (suicide intent detection)
- **Framework**: PyTorch + Transformers
- **Integration**: FastAPI

## Support

For issues or improvements:
1. Check test output: `python test_suicide_detection.py`
2. Review terminal output for error messages
3. Check model directory structure
4. Verify all dependencies installed
