"""Suicide detection module using ELECTRA model and LLM analysis."""

import torch
from transformers import ElectraForSequenceClassification, ElectraTokenizer
from sklearn.preprocessing import LabelEncoder
import os
from typing import Dict, Optional, Tuple
from datetime import datetime
from collections import defaultdict
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
import json
from constants import CRISIS_DETECTION_LLM_MODEL_NAME, DETERMINISTIC_TEMPERATURE


class SuicideDetector:
    """ELECTRA-based suicide detection system with session tracking."""
    
    def __init__(self, model_path: str = None):
        """
        Initialize suicide detector.
        
        Args:
            model_path: Path to ELECTRA model directory. 
                       If None, tries to use default location in Crisis Detection Model.
        """
        self.model = None
        self.tokenizer = None
        self.device = None
        self.label_encoder = None
        
        # Session tracking: {session_id: [confidence_scores]}
        self.session_low_confidence_tracker = defaultdict(list)
        self.LOW_CONFIDENCE_THRESHOLD = 0.6  # Below this = low confidence of "not suicide"
        self.LOW_CONFIDENCE_ALERT_COUNT = 5  # Alert after 5 low confidence messages
        
        # Combined (LLM+ELECTRA): session tracking for moderate-risk accumulation
        # Counter increments ONCE per message if:
        # - LLM risk_level == "moderate" OR
        # - ELECTRA suicidal probability >= 0.5
        self.session_moderate_risk_tracker = defaultdict(int)  # {session_id: count}
        self.session_moderate_risk_messages = defaultdict(list)  # {session_id: [ {text,timestamp,signals...}, ... ]}
        self.session_last_recheck_count = defaultdict(int)  # {session_id: last_count_rechecked}
        self.MODERATE_RISK_ALERT_COUNT = 5
        self.MODERATE_BUFFER_MAX = 10
        
        self._initialize_model(model_path)
    
    def _initialize_model(self, model_path: Optional[str] = None):
        """Initialize ELECTRA model and tokenizer."""
        if model_path is None:
            # Try to find the model in the project directory
            base_dir = os.path.dirname(os.path.abspath(__file__))
            # Try nested structure first (electra_suicidal_text_detector/electra_suicidal_text_detector)
            model_path = os.path.join(
                base_dir, 
                "Crisis Detection Model", 
                "electra_suicidal_text_detector",
                "electra_suicidal_text_detector"
            )
            
            # Fallback to flat structure if nested doesn't exist
            if not os.path.exists(model_path):
                model_path = os.path.join(
                    base_dir, 
                    "Crisis Detection Model", 
                    "electra_suicidal_text_detector"
                )
        
        try:
            if not os.path.exists(model_path):
                print(f"⚠️  WARNING: Model not found at {model_path}")
                print("   Checked paths:")
                print(f"   - {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Crisis Detection Model', 'electra_suicidal_text_detector', 'electra_suicidal_text_detector')}")
                print(f"   - {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Crisis Detection Model', 'electra_suicidal_text_detector')}")
                print("   Suicide detection will be disabled")
                self.model = None
                return
            
            print(f"🔹 Loading ELECTRA Suicide Detection Model from: {model_path}")
            
            # Load tokenizer and model
            self.tokenizer = ElectraTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = ElectraForSequenceClassification.from_pretrained(model_path, local_files_only=True)
            
            # Setup device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            
            # Setup label encoder
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(['non-suicide', 'suicide'])
            
            print(f"✅ ELECTRA Model loaded successfully!")
            print(f"   Device: {self.device}")
            print("=" * 80)
            
        except Exception as e:
            print(f"❌ ERROR loading ELECTRA model: {str(e)}")
            print("   Suicide detection will be disabled")
            self.model = None
    
    def is_available(self) -> bool:
        """Check if model is available."""
        return self.model is not None
    
    def predict(self, text: str, session_id: str = None) -> Dict:
        """
        Predict if text contains suicidal intent.
        
        Args:
            text: Input text to analyze
            session_id: Session ID for tracking multiple messages
            
        Returns:
            Dict with prediction, confidence, and probabilities
        """
        if not self.is_available():
            return {
                'text': text,
                'prediction': 'unknown',
                'confidence': 0.0,
                'probabilities': {'non-suicidal': 0.0, 'suicidal': 0.0},
                'is_alert': False,
                'alert_reason': 'Model not available'
            }
        
        try:
            # Use modern tokenizer syntax (encode_plus is deprecated)
            encoding = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].to(self.device)
            attention_mask = encoding['attention_mask'].to(self.device)
            
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=1)
                prediction = torch.argmax(probs, dim=1)
            
            class_name = self.label_encoder.inverse_transform(prediction.cpu().numpy())[0]
            confidence = probs[0][prediction.item()].item()
            
            non_suicide_prob = probs[0][0].item()
            suicide_prob = probs[0][1].item()
            
            # Track low-confidence messages (kept for ELECTRA-only heuristic)
            is_alert = False
            alert_reason = None
            if session_id is not None:
                if non_suicide_prob < self.LOW_CONFIDENCE_THRESHOLD:
                    self.session_low_confidence_tracker[session_id].append(non_suicide_prob)
                    
                    if len(self.session_low_confidence_tracker[session_id]) >= self.LOW_CONFIDENCE_ALERT_COUNT:
                        is_alert = True
                        alert_reason = f"MULTIPLE LOW CONFIDENCE MESSAGES ({self.LOW_CONFIDENCE_ALERT_COUNT} messages with <{self.LOW_CONFIDENCE_THRESHOLD} confidence)"
            
            return {
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'prediction': class_name,
                'confidence': confidence,
                'probabilities': {
                    'non-suicidal': non_suicide_prob,
                    'suicidal': suicide_prob
                },
                'suicidal_probability': suicide_prob,
                'is_alert': is_alert,
                'alert_reason': alert_reason,
                'low_confidence_count': len(self.session_low_confidence_tracker.get(session_id, []))
            }
        
        except Exception as e:
            print(f"❌ Error in suicide detection: {str(e)}")
            return {
                'text': text,
                'prediction': 'error',
                'confidence': 0.0,
                'probabilities': {'non-suicidal': 0.0, 'suicidal': 0.0},
                'is_alert': False,
                'alert_reason': f'Prediction error: {str(e)}'
            }
    
    def predict_with_llm(self, text: str, session_id: str = None) -> Dict:
        """
        Detect suicidal intent using LLM analysis.
        
        Args:
            text: Input text to analyze
            session_id: Session ID for tracking
            
        Returns:
            Dict with LLM prediction, reasoning, and alert status
        """
        try:
            llm = ChatGroq(model_name=CRISIS_DETECTION_LLM_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)
            
            system_prompt = """You are a clinical expert in suicide risk assessment. 
Analyze the given text for indicators of suicidal intent, self-harm thoughts, and crisis markers.

RESPOND ONLY with a JSON object (no other text):
{{
    "has_suicide_intent": true/false,
    "confidence": 0.0-1.0,
    "risk_level": "low/moderate/high/critical",
    "indicators": ["list of specific concerning phrases or patterns"],
    "reasoning": "brief clinical assessment",
    "recommendation": "clinical recommendation"
}}"""
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", "{text}")
            ])
            
            chain = prompt | llm
            response = chain.invoke({"text": text})
            response_text = response.content
            
            # Parse JSON response
            try:
                # Try to extract JSON from response
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    llm_result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
            except (json.JSONDecodeError, ValueError) as e:
                print(f"⚠️  Warning: Could not parse LLM response: {str(e)}")
                llm_result = {
                    "has_suicide_intent": False,
                    "confidence": 0.0,
                    "risk_level": "unknown",
                    "indicators": [],
                    "reasoning": f"Parse error: {str(e)}",
                    "recommendation": "Manual review recommended"
                }
            
            risk_level = str(llm_result.get("risk_level", "unknown") or "unknown").strip().lower()
            has_intent = bool(llm_result.get("has_suicide_intent", False))

            # Alert only for HIGH/CRITICAL (or explicit intent). Moderate is tracked separately
            is_alert = bool(has_intent or risk_level in ["high", "critical"])
            alert_reason = None
            if is_alert:
                if has_intent:
                    alert_reason = "LLM: SUICIDE INTENT DETECTED"
                else:
                    alert_reason = f"LLM: {risk_level.upper()} RISK DETECTED"
            
            return {
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'model': 'LLM',
                'has_suicide_intent': has_intent,
                'confidence': llm_result.get("confidence", 0.0),
                'risk_level': risk_level,
                'indicators': llm_result.get("indicators", []),
                'reasoning': llm_result.get("reasoning", ""),
                'recommendation': llm_result.get("recommendation", ""),
                'is_alert': is_alert,
                'alert_reason': alert_reason
            }
        
        except Exception as e:
            print(f"❌ Error in LLM suicide detection: {str(e)}")
            return {
                'text': text[:100] + ('...' if len(text) > 100 else ''),
                'model': 'LLM',
                'has_suicide_intent': False,
                'confidence': 0.0,
                'risk_level': 'unknown',
                'indicators': [],
                'reasoning': '',
                'recommendation': '',
                'is_alert': False,
                'alert_reason': f'LLM error: {str(e)}'
            }

    def _should_increment_moderate_counter(self, electra_result: Optional[Dict], llm_result: Optional[Dict]) -> bool:
        """Return True if this message should increment the moderate counter (once per message)."""
        llm_moderate = False
        if llm_result is not None:
            llm_moderate = str(llm_result.get("risk_level", "")).strip().lower() == "moderate"

        electra_over_50 = False
        if electra_result is not None:
            try:
                electra_over_50 = float(electra_result.get("probabilities", {}).get("suicidal", 0.0)) >= 0.5
            except Exception:
                electra_over_50 = False

        return bool(llm_moderate or electra_over_50)

    def update_moderate_tracking_and_maybe_recheck(
        self,
        *,
        session_id: str,
        text: str,
        electra_result: Optional[Dict],
        llm_result: Optional[Dict],
        timestamp: Optional[str] = None,
    ) -> Dict:
        """Update moderate-only buffer + counter and run final re-check at threshold.

        Rules:
        - Increment counter once per message if (LLM moderate OR ELECTRA suicidal>=0.5)
        - Do NOT attach buffer to every message detection.
        - When counter hits 5, run a final LLM re-check using buffered moderate messages.
        """
        if not session_id:
            return {
                "moderate_count": 0,
                "did_increment": False,
                "final_recheck": None,
                "is_alert": False,
                "alert_reason": None,
            }

        did_increment = self._should_increment_moderate_counter(electra_result, llm_result)
        if did_increment:
            self.session_moderate_risk_tracker[session_id] += 1
            moderate_count = self.session_moderate_risk_tracker[session_id]

            self.session_moderate_risk_messages[session_id].append(
                {
                    "text": text,
                    "timestamp": timestamp or datetime.utcnow().isoformat(),
                    "llm_risk_level": (llm_result or {}).get("risk_level"),
                    "llm_confidence": (llm_result or {}).get("confidence"),
                    "electra_suicidal_prob": (electra_result or {}).get("probabilities", {}).get("suicidal"),
                }
            )

            # keep only last N
            if len(self.session_moderate_risk_messages[session_id]) > self.MODERATE_BUFFER_MAX:
                self.session_moderate_risk_messages[session_id] = self.session_moderate_risk_messages[session_id][
                    -self.MODERATE_BUFFER_MAX :
                ]
        else:
            moderate_count = self.session_moderate_risk_tracker.get(session_id, 0)

        # Run final re-check only when transitioning to threshold (hit exactly 5)
        final_recheck = None
        is_alert = False
        alert_reason = None

        if did_increment and moderate_count == self.MODERATE_RISK_ALERT_COUNT and self.session_last_recheck_count[session_id] < moderate_count:
            self.session_last_recheck_count[session_id] = moderate_count
            final_recheck = self.final_recheck_moderate_buffer(session_id=session_id)
            if final_recheck and final_recheck.get("is_alert"):
                is_alert = True
                alert_reason = final_recheck.get("alert_reason") or "FINAL RECHECK: HIGH RISK"
            else:
                alert_reason = f"CUMULATIVE MODERATE COUNT HIT {moderate_count}/{self.MODERATE_RISK_ALERT_COUNT} (final re-check did not escalate)"
        elif did_increment:
            alert_reason = f"MODERATE COUNTER INCREMENTED ({moderate_count}/{self.MODERATE_RISK_ALERT_COUNT})"

        return {
            "moderate_count": moderate_count,
            "did_increment": did_increment,
            "final_recheck": final_recheck,
            "is_alert": is_alert,
            "alert_reason": alert_reason,
        }

    def final_recheck_moderate_buffer(self, *, session_id: str) -> Optional[Dict]:
        """Run a final LLM re-check using buffered moderate messages for the session."""
        buffer = self.session_moderate_risk_messages.get(session_id) or []
        if not buffer:
            return None

        llm = ChatGroq(model_name=CRISIS_DETECTION_LLM_MODEL_NAME, temperature=DETERMINISTIC_TEMPERATURE)

        system_prompt = """You are a clinical expert in suicide risk assessment.

You will be given a list of user messages that were flagged as borderline/moderate concern across multiple turns.
Your task: make a FINAL risk assessment for the session based ONLY on these messages.

Be cautious about false positives. Escalate to HIGH/CRITICAL only if there is clear evidence of self-harm intent, planning, imminent risk, or repeated strong suicidal ideation.

RESPOND ONLY with a JSON object (no other text):
{{
  "has_suicide_intent": true/false,
  "confidence": 0.0-1.0,
  "risk_level": "low/moderate/high/critical",
  "indicators": ["specific phrases/patterns"],
  "reasoning": "brief clinical assessment",
  "recommendation": "clinical recommendation"
}}
"""

        # Build a compact context block
        lines = []
        for i, item in enumerate(buffer[-self.MODERATE_BUFFER_MAX :], 1):
            t = (item.get("text") or "").strip().replace("\n", " ")
            if len(t) > 240:
                t = t[:240] + "..."
            lines.append(f"{i}. {t}")

        user_block = "\n".join(lines)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "Moderate-flagged messages:\n{messages}")
            ]
        )
        chain = prompt | llm

        response = chain.invoke({"messages": user_block})
        response_text = response.content

        # Parse JSON
        try:
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response_text[json_start:json_end]
                llm_result = json.loads(json_str)
            else:
                raise ValueError("No JSON found in response")
        except (json.JSONDecodeError, ValueError) as e:
            return {
                "model": "LLM_FINAL_RECHECK",
                "has_suicide_intent": False,
                "confidence": 0.0,
                "risk_level": "unknown",
                "indicators": [],
                "reasoning": f"Parse error: {str(e)}",
                "recommendation": "Manual review recommended",
                "is_alert": False,
                "alert_reason": "FINAL RECHECK FAILED TO PARSE",
            }

        risk_level = str(llm_result.get("risk_level", "unknown") or "unknown").strip().lower()
        has_intent = bool(llm_result.get("has_suicide_intent", False))
        is_alert = bool(has_intent or risk_level in ["high", "critical"])

        alert_reason = None
        if is_alert:
            if has_intent:
                alert_reason = "FINAL RECHECK: SUICIDE INTENT DETECTED"
            else:
                alert_reason = f"FINAL RECHECK: {risk_level.upper()} RISK DETECTED"

        return {
            "text": "(final re-check on buffered moderate messages)",
            "model": "LLM_FINAL_RECHECK",
            "has_suicide_intent": has_intent,
            "confidence": llm_result.get("confidence", 0.0),
            "risk_level": risk_level,
            "indicators": llm_result.get("indicators", []),
            "reasoning": llm_result.get("reasoning", ""),
            "recommendation": llm_result.get("recommendation", ""),
            "is_alert": is_alert,
            "alert_reason": alert_reason,
        }
    
    def print_result(self, result: Dict, session_id: str = None):
        """Print formatted detection result."""
        if result.get('model') in ('LLM', 'LLM_FINAL_RECHECK'):
            # LLM Result formatting
            print("\n" + "="*80)
            header = "🤖 LLM SUICIDE RISK ANALYSIS" if result.get('model') == 'LLM' else "🤖 LLM FINAL RE-CHECK"
            print(header)
            print("="*80)
            print(f"Text: {result['text']}")
            print(f"Risk Level: {result['risk_level'].upper()} (Confidence: {result['confidence']:.2%})")
            
            if result['indicators']:
                print(f"Indicators: {', '.join(result['indicators'][:3])}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Recommendation: {result['recommendation']}")
        else:
            # ELECTRA Result formatting
            if not self.is_available() and result['prediction'] == 'unknown':
                print(f"⚠️  Model not available - skipping detection")
                return
            
            print("\n" + "="*80)
            print("🔬 ELECTRA SUICIDE DETECTION ANALYSIS")
            print("="*80)
            print(f"Text: {result['text']}")
            print(f"Prediction: {result['prediction'].upper()} ({result['confidence']:.2%})")
            print(f"  → Non-Suicidal: {result['probabilities']['non-suicidal']:.4f}")
            print(f"  → Suicidal: {result['probabilities']['suicidal']:.4f}")
            
            if session_id:
                low_conf_count = result.get('low_confidence_count', 0)
                print(f"Low Confidence Count in Session: {low_conf_count}/{self.LOW_CONFIDENCE_ALERT_COUNT}")
        
        if result['is_alert'] or result.get('alert_reason'):
            if result['is_alert']:
                print(f"\n⚠️  🚨 ALERT TRIGGERED: {result['alert_reason']}")
                print(f"    Timestamp: {datetime.utcnow().isoformat()}")
            else:
                # Show alert info even if not triggering emergency (for moderate risk tracking)
                print(f"\n⚠️  {result['alert_reason']}")
            if session_id:
                print(f"    Session ID: {session_id}")
        
        print("="*80)
    
    def reset_session(self, session_id: str):
        """Reset tracking for a session."""
        if session_id in self.session_low_confidence_tracker:
            del self.session_low_confidence_tracker[session_id]
        if session_id in self.session_moderate_risk_tracker:
            del self.session_moderate_risk_tracker[session_id]
        if session_id in self.session_moderate_risk_messages:
            del self.session_moderate_risk_messages[session_id]
        if session_id in self.session_last_recheck_count:
            del self.session_last_recheck_count[session_id]


# Global singleton instance
_detector = None

def get_suicide_detector(model_path: str = None) -> SuicideDetector:
    """Get or create global suicide detector instance."""
    global _detector
    if _detector is None:
        _detector = SuicideDetector(model_path)
    return _detector
