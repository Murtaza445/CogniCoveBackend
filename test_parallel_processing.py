"""
Test script to verify parallel processing of messages with crisis detection.
Tests both text and audio endpoints to confirm all tasks run simultaneously.
"""

import asyncio
import time
import requests
import json
from datetime import datetime


# Configuration
BASE_URL = "http://localhost:8000"
SESSION_ID = f"test_session_{datetime.now().timestamp()}"


def print_header(title):
    """Print a formatted header."""
    print("\n" + "="*80)
    print(f"🧪 {title}")
    print("="*80)


def print_section(title):
    """Print a formatted section."""
    print(f"\n📌 {title}")
    print("-"*80)


# ==================== TEST 1: TEXT ENDPOINT PARALLEL PROCESSING ====================

def test_text_therapy_parallel():
    """Test parallel processing in text therapy endpoint."""
    print_header("TEST 1: TEXT THERAPY ENDPOINT - PARALLEL PROCESSING")
    
    test_messages = [
        {
            "text": "I am done with this life, I can't take it anymore.",
            "type": "suicide_direct"
        },
        {
            "text": "I'm struggling with my thoughts and feeling very overwhelmed.",
            "type": "normal_distress"
        },
        {
            "text": "I don't see the point anymore, everything is pointless.",
            "type": "suicide_indirect"
        }
    ]
    
    for msg_data in test_messages:
        print_section(f"Testing: {msg_data['type']}")
        print(f"Message: {msg_data['text']}\n")
        
        # Measure START time
        start_time = time.time()
        
        # Send request
        response = requests.post(
            f"{BASE_URL}/api/therapy",
            json={
                "session_id": SESSION_ID,
                "content": msg_data['text']
            }
        )
        
        # Measure END time
        end_time = time.time()
        response_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Response received in {response_time:.2f}s")
            print(f"   User: {result['user_message'][:60]}...")
            print(f"   AI: {result['assistant_message'][:80]}...")
            print(f"   Timestamp: {result['timestamp']}")
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"   {response.text}")
        
        # Fetch session to see crisis detection results
        print("\n📊 Crisis Detection Results:")
        fetch_session_crisis_data(SESSION_ID)
        print()


# ==================== TEST 2: CHECK SESSION CRISIS DATA ====================

def fetch_session_crisis_data(session_id):
    """Fetch and display crisis detection data from session."""
    response = requests.get(f"{BASE_URL}/api/sessions/{session_id}")
    
    if response.status_code == 200:
        session = response.json()
        
        # Check if crisis detections exist
        if 'suicide_detections' in session and session['suicide_detections']:
            for idx, detection in enumerate(session['suicide_detections'], 1):
                print(f"\n   Detection #{idx}:")
                print(f"   Timestamp: {detection['timestamp']}")
                
                # ELECTRA Results
                if detection.get('electra_result'):
                    electra = detection['electra_result']
                    print(f"   🔬 ELECTRA: {electra.get('prediction', 'N/A').upper()}")
                    print(f"      Confidence: {electra.get('confidence', 0):.2%}")
                    if electra.get('is_alert'):
                        print(f"      ⚠️  ALERT: {electra.get('alert_reason')}")
                
                # LLM Results
                if detection.get('llm_result'):
                    llm = detection['llm_result']
                    print(f"   🤖 LLM: {llm.get('risk_level', 'unknown').upper()}")
                    print(f"      Confidence: {llm.get('confidence', 0):.2%}")
                    if llm.get('indicators'):
                        print(f"      Indicators: {', '.join(llm['indicators'][:2])}")
                    if llm.get('is_alert'):
                        print(f"      ⚠️  ALERT: {llm.get('alert_reason')}")
        else:
            print("   ℹ️  No crisis detections yet")
    else:
        print(f"   ❌ Error fetching session: {response.status_code}")


# ==================== TEST 3: PARALLEL TIMING COMPARISON ====================

def test_parallel_timing():
    """
    Compare timing to verify parallel execution.
    
    If sequential:
      - ELECTRA (200ms) + LLM (3000ms) + Therapy (2000ms) = ~5200ms
    
    If parallel:
      - Max(ELECTRA, LLM, Therapy) = ~3000ms
    """
    print_header("TEST 2: PARALLEL TIMING VERIFICATION")
    
    print_section("Sequential Execution (Expected if NOT parallel)")
    print("ELECTRA Detection:    ~100-200ms")
    print("LLM Detection:        ~1000-3000ms")
    print("Emotion Analysis:     ~500-1000ms")
    print("Therapy Generation:   ~1000-2000ms")
    print("─" * 40)
    print("TOTAL TIME (sequential): ~3600-6200ms")
    
    print_section("Parallel Execution (Expected with asyncio)")
    print("ELECTRA:     →")
    print("LLM:         ────→")
    print("Emotion:     ──→")
    print("Therapy:     ───→")
    print("─" * 40)
    print("TOTAL TIME (parallel):   ~1000-3000ms (max of all)")
    
    print_section("Actual Test Results")
    
    test_msg = "I'm feeling really down and don't know if I can continue."
    response_times = []
    
    for i in range(3):
        start = time.time()
        response = requests.post(
            f"{BASE_URL}/api/therapy",
            json={
                "session_id": SESSION_ID,
                "content": test_msg
            }
        )
        elapsed = time.time() - start
        response_times.append(elapsed)
        
        status = "✅" if response.status_code == 200 else "❌"
        print(f"   Request {i+1}: {status} {elapsed:.2f}s")
    
    avg_time = sum(response_times) / len(response_times)
    print(f"\n   Average Response Time: {avg_time:.2f}s")
    
    if avg_time < 3.5:
        print("   ✅ LIKELY PARALLEL (fast response)")
    else:
        print("   ⚠️  POSSIBLY SEQUENTIAL (slow response)")


# ==================== TEST 4: VERIFY BOTH DETECTIONS RUN ====================

def test_both_detections_run():
    """Verify that BOTH ELECTRA and LLM detections are executed."""
    print_header("TEST 3: VERIFY BOTH DETECTIONS ARE EXECUTED")
    
    # Send a message that should trigger both detections
    test_msg = "I want to end my life tonight"
    
    print_section("Sending test message")
    print(f"Message: {test_msg}\n")
    
    start = time.time()
    response = requests.post(
        f"{BASE_URL}/api/therapy",
        json={
            "session_id": SESSION_ID,
            "content": test_msg
        }
    )
    elapsed = time.time() - start
    
    print(f"Response Time: {elapsed:.2f}s")
    
    if response.status_code == 200:
        # Check session for both detections
        session_response = requests.get(f"{BASE_URL}/api/sessions/{SESSION_ID}")
        if session_response.status_code == 200:
            session = session_response.json()
            
            if session.get('suicide_detections'):
                latest = session['suicide_detections'][-1]
                
                electra_present = "electra_result" in latest and latest['electra_result'] is not None
                llm_present = "llm_result" in latest and latest['llm_result'] is not None
                
                print_section("Detection Results")
                print(f"🔬 ELECTRA Detection: {'✅ Present' if electra_present else '❌ Missing'}")
                if electra_present:
                    e = latest['electra_result']
                    print(f"   Prediction: {e.get('prediction')}")
                    print(f"   Alert: {e.get('is_alert')}")
                
                print(f"🤖 LLM Detection: {'✅ Present' if llm_present else '❌ Missing'}")
                if llm_present:
                    l = latest['llm_result']
                    print(f"   Risk Level: {l.get('risk_level')}")
                    print(f"   Alert: {l.get('is_alert')}")
                
                if electra_present and llm_present:
                    print("\n✅ BOTH DETECTIONS WORKING!")
                else:
                    print("\n❌ One or both detections missing!")


# ==================== TEST 5: AUDIO ENDPOINT PARALLEL ====================

def test_audio_endpoint_parallel():
    """Test parallel processing with audio endpoint."""
    print_header("TEST 4: AUDIO ENDPOINT - PARALLEL PROCESSING")
    
    print_section("Requirements for audio test")
    print("1. You need an audio file (WAV format, 16-bit, 16kHz)")
    print("2. Audio should contain speech")
    print("3. For testing: record 'I want to end my life' or similar")
    print()
    print("Example with curl:")
    print("curl -X POST http://localhost:8000/api/therapy/audio")
    print('  -F "file=@audio.wav"')
    print('  -F "session_id=test_session_123"')
    print()
    print("Expected parallel execution:")
    print("  [STT] → [ELECTRA] ← → [LLM] ← → [Emotion] ← → [Therapy]")
    print("          (all 4 tasks in parallel)")


# ==================== TEST 6: CHECK TERMINAL OUTPUT ====================

def test_terminal_output_verification():
    """Guide user to check terminal output for parallel execution."""
    print_header("TEST 5: VERIFY PARALLEL EXECUTION IN TERMINAL")
    
    print_section("What to look for in FastAPI terminal output:")
    print()
    print("✅ Sign of parallel execution (GOOD):")
    print("   • ELECTRA output appears")
    print("   • LLM output appears")  
    print("   • Therapy output appears")
    print("   • All timestamps are very close/simultaneous")
    print("   • Variable response times (1-3 seconds)")
    print()
    print("❌ Sign of sequential execution (BAD):")
    print("   • Output appears in strict order: ELECTRA → LLM → Therapy")
    print("   • Consistent delay between each step")
    print("   • Response always ~4-5 seconds")
    print()
    print("📝 Terminal should show:")
    print("   🔬 ELECTRA SUICIDE DETECTION ANALYSIS")
    print("   ...[output]...")
    print("   🤖 LLM SUICIDE RISK ANALYSIS")
    print("   ...[output]...")
    print("   (potentially overlapping timestamps)")


# ==================== MAIN ====================

def main():
    """Run all tests."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  PARALLEL PROCESSING TEST SUITE - Crisis Detection Integration".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    # Check if server is running
    print_section("Pre-flight Check")
    try:
        response = requests.get(f"{BASE_URL}/docs")
        if response.status_code == 200:
            print("✅ FastAPI server is running on http://localhost:8000")
        else:
            print("⚠️  Server might not be responding correctly")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to server!")
        print("\n   Start the server with:")
        print("   cd Chat-CogniCoveModel")
        print("   uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    try:
        # Run tests
        test_text_therapy_parallel()
        test_parallel_timing()
        test_both_detections_run()
        test_audio_endpoint_parallel()
        test_terminal_output_verification()
        
        # Summary
        print_header("TEST SUMMARY")
        print("""
📊 INTERPRETATION GUIDE:

1. RESPONSE TIMES:
   • < 2.5s = Excellent parallel execution ✅
   • 2.5-3.5s = Good parallel execution ✅
   • 3.5-5s = Possible partial parallelization ⚠️
   • > 5s = Likely sequential ❌

2. CRISIS DETECTIONS:
   • Both ELECTRA and LLM results = ✅ Working
   • Only one present = ❌ Check code

3. SESSION DATA:
   View full session at: http://localhost:8000/api/sessions/{session_id}

4. MONITOR TERMINAL:
   Watch FastAPI terminal while running tests to see:
   • Order of output
   • Timestamps
   • Processing sequence

📌 SESSION ID FOR THIS TEST:
   {session_id}

   View data: GET /api/sessions/{session_id}
   End session: POST /api/sessions/{session_id}/end
        """.format(session_id=SESSION_ID))
        
    except Exception as e:
        print(f"\n❌ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
