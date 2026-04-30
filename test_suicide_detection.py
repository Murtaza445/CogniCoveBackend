"""
Test script for suicide detection integration.
This demonstrates the suicide detection system in action.
"""

import sys
from datetime import datetime
from suicide_detection import get_suicide_detector


def test_suicide_detection():
    """Test the suicide detection system."""
    
    print("\n" + "="*80)
    print("🧪 SUICIDE DETECTION INTEGRATION TEST")
    print("="*80 + "\n")
    
    # Initialize detector
    detector = get_suicide_detector()
    
    if not detector.is_available():
        print("❌ Model not available. Please ensure:")
        print("   1. ELECTRA model is in: Crisis Detection Model/electra_suicidal_text_detector/")
        print("   2. All required files are present (config.json, pytorch_model.bin, etc.)")
        return
    
    # Test 1: Direct suicide detection
    print("\n" + "-"*80)
    print("TEST 1: Direct Suicide Detection")
    print("-"*80)
    
    test_messages_direct = [
        "I am done with this life, I cannot take it anymore.",
        "I feel good today, life is beautiful.",
    ]
    
    session_id = "test_session_1"
    for msg in test_messages_direct:
        print(f"\n📝 Testing message: '{msg}'")
        result = detector.predict(msg, session_id=session_id)
        detector.print_result(result, session_id=session_id)
    
    detector.reset_session(session_id)
    
    # Test 2: Multiple low confidence messages
    print("\n" + "-"*80)
    print("TEST 2: Multiple Low Confidence Messages (Cumulative Alert)")
    print("-"*80)
    
    test_messages_cumulative = [
        "I don't know if I should continue.",
        "Everything seems meaningless lately.",
        "I can't keep doing this.",
        "Nobody understands my pain.",
        "It would be better if I just disappeared.",
    ]
    
    session_id = "test_session_2"
    for i, msg in enumerate(test_messages_cumulative, 1):
        print(f"\n📝 Message {i}/5: '{msg}'")
        result = detector.predict(msg, session_id=session_id)
        detector.print_result(result, session_id=session_id)
    
    detector.reset_session(session_id)
    
    # Test 3: Mixed messages
    print("\n" + "-"*80)
    print("TEST 3: Mixed Messages (Safe & Concerning)")
    print("-"*80)
    
    test_messages_mixed = [
        "I'm having a great day!",
        "I love spending time with family.",
        "I'm not sure anymore if life is worth living.",
        "The weather is nice today.",
        "I want to end it all.",
    ]
    
    session_id = "test_session_3"
    for i, msg in enumerate(test_messages_mixed, 1):
        print(f"\n📝 Message {i}/5: '{msg}'")
        result = detector.predict(msg, session_id=session_id)
        detector.print_result(result, session_id=session_id)
    
    detector.reset_session(session_id)
    
    print("\n" + "="*80)
    print("✅ TEST COMPLETED")
    print("="*80)
    print("\n📌 KEY FEATURES:")
    print("  ✓ Direct suicide detection (high confidence)")
    print("  ✓ Cumulative low-confidence tracking (5+ messages)")
    print("  ✓ Session-based tracking (resets per session)")
    print("  ✓ Formatted terminal output with alerts")
    print("  ✓ Ready for emergency call tool integration")
    print()


if __name__ == "__main__":
    test_suicide_detection()
