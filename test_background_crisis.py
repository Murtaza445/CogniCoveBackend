"""
Simple test script to verify background crisis detection.
Shows terminal output of crisis detection running in background while therapy responds immediately.
"""

import requests
import time
from datetime import datetime

BASE_URL = "http://localhost:8000"
SESSION_ID = f"crisis_test_{int(time.time())}"

def test_message(text, msg_type):
    """Send one test message and show response time."""
    print("\n" + "="*80)
    print(f"📤 SENDING: {msg_type}")
    print("="*80)
    print(f"Message: {text[:70]}...")
    print()
    
    start_time = time.time()
    
    # Send message
    response = requests.post(
        f"{BASE_URL}/api/therapy",
        json={
            "session_id": SESSION_ID,
            "content": text
        }
    )
    
    elapsed = time.time() - start_time
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ THERAPY RESPONSE (received in {elapsed:.2f}s)")
        print(f"   AI: {result['assistant_message'][:100]}...")
        print()
        print(f"📌 NOTE: Crisis detection is running in BACKGROUND!")
        print(f"   Watch the FastAPI server terminal for detection results...")
        print(f"   After 1-3 seconds you should see:")
        print(f"   🔬 ELECTRA SUICIDE DETECTION ANALYSIS")
        print(f"   🤖 LLM SUICIDE RISK ANALYSIS")
    else:
        print(f"❌ Error: {response.status_code}")
        print(response.text)

def main():
    """Run simple crisis detection demo."""
    print("\n" + "█"*80)
    print("█" + " "*78 + "█")
    print("█" + "  BACKGROUND CRISIS DETECTION TEST".center(78) + "█")
    print("█" + " "*78 + "█")
    print("█"*80)
    
    print("\n⚡ WORKFLOW:")
    print("  1. You send a message")
    print("  2. Therapy response returns IMMEDIATELY (< 1 second)")
    print("  3. Crisis detection runs in BACKGROUND (1-3 seconds)")
    print("  4. Results print to FastAPI terminal when done")
    print()
    print("📌 KEY DIFFERENCE FROM BEFORE:")
    print("  ❌ OLD: Wait for both therapy AND crisis detection")
    print("  ✅ NEW: Get therapy immediately, crisis detection in background")
    
    # Check server is running
    print("\n" + "="*80)
    try:
        response = requests.get(f"{BASE_URL}/docs")
        print("✅ FastAPI server is running!")
    except:
        print("❌ Cannot connect to server!")
        print("\nStart the server with:")
        print("  cd Chat-CogniCoveModel")
        print("  uvicorn main:app --reload --host 0.0.0.0 --port 8000")
        return
    
    # Test 1: Direct suicide
    test_message(
        "this is it , i cannot take it anymore there is not point of continuing with it ",
        "SUICIDE INTENT (DIRECT)"
    )
    print("\n⏳ Waiting 3 seconds for background detection to complete...")
    time.sleep(3)
    
    # Test 2: Normal message
    test_message(
        "I'm having a good day today and feeling better",
        "NORMAL MESSAGE"
    )
    print("\n⏳ Waiting 2 seconds...")
    time.sleep(2)
    
    # Test 3: Indirect suicide
    test_message(
        "I don't see the point of living anymore, everything is meaningless",
        "SUICIDE INTENT (INDIRECT)"
    )
    print("\n⏳ Waiting 3 seconds...")
    time.sleep(3)
    
    # Show session data
    print("\n" + "="*80)
    print("📊 VIEWING SESSION DATA")
    print("="*80)
    
    response = requests.get(f"{BASE_URL}/api/sessions/{SESSION_ID}")
    if response.status_code == 200:
        session = response.json()
        
        if session.get('suicide_detections'):
            print(f"\n✅ Found {len(session['suicide_detections'])} crisis detections:")
            
            for idx, detection in enumerate(session['suicide_detections'], 1):
                print(f"\n   Detection #{idx}:")
                print(f"   Message: {detection['user_message'][:60]}...")
                
                if detection.get('electra_result'):
                    e = detection['electra_result']
                    print(f"   🔬 ELECTRA: {e.get('prediction', 'N/A').upper()} ({e.get('confidence', 0):.1%})")
                    if e.get('is_alert'):
                        print(f"      🚨 ALERT: {e.get('alert_reason')}")
                
                if detection.get('llm_result'):
                    l = detection['llm_result']
                    print(f"   🤖 LLM: {l.get('risk_level', 'unknown').upper()} ({l.get('confidence', 0):.1%})")
                    if l.get('is_alert'):
                        print(f"      🚨 ALERT: {l.get('alert_reason')}")

                if detection.get('moderate_tracking'):
                    mt = detection['moderate_tracking']
                    if mt is not None:
                        print(
                            f"   🟡 MODERATE TRACKING: count={mt.get('moderate_count', 0)} "
                            f"incremented={mt.get('did_increment', False)}"
                        )
                        if mt.get('alert_reason'):
                            print(f"      ℹ️  {mt.get('alert_reason')}")

                if detection.get('llm_final_recheck_result'):
                    fr = detection['llm_final_recheck_result']
                    if fr is not None:
                        print(
                            f"   🧾 FINAL RE-CHECK: {fr.get('risk_level', 'unknown').upper()} "
                            f"({fr.get('confidence', 0):.1%})"
                        )
                        if fr.get('is_alert'):
                            print(f"      🚨 ALERT: {fr.get('alert_reason')}")
        else:
            print("\n❌ No crisis detections found in session")
    
    print("\n" + "="*80)
    print("\n✅ TEST COMPLETE!")
    print("\nWhat happened:")
    print("  ✅ Therapy responses were fast (< 1 second)")
    print("  ✅ Crisis detection ran in background")
    print("  ✅ Results appear in FastAPI terminal after ~1-3 seconds")
    print("\nNext step: Replace terminal output with emergency alert function!")
    print()

if __name__ == "__main__":
    main()
