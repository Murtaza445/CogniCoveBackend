"""Email service for sending crisis alert notifications."""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Optional
from datetime import datetime


# SMTP Configuration from environment
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "Murtazajohar123@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)


def _build_crisis_email_html(user_name: str, session_id: str, message_preview: str, timestamp: str) -> str:
    """Build a professional HTML email template for crisis alerts."""
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Emergency Crisis Alert - CogniCove</title>
        <style>
            body {{ margin: 0; padding: 0; background-color: #f4f4f4; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }}
            .container {{ max-width: 600px; margin: 0 auto; background-color: #ffffff; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
            .header {{ background-color: #dc2626; padding: 30px 20px; text-align: center; }}
            .header h1 {{ color: #ffffff; margin: 0; font-size: 24px; font-weight: 600; }}
            .header-icon {{ font-size: 40px; margin-bottom: 10px; }}
            .content {{ padding: 30px 25px; }}
            .alert-box {{ background-color: #fef2f2; border-left: 4px solid #dc2626; padding: 18px; margin-bottom: 24px; border-radius: 0 6px 6px 0; }}
            .alert-box h2 {{ color: #991b1b; margin: 0 0 8px 0; font-size: 18px; }}
            .alert-box p {{ color: #7f1d1d; margin: 0; font-size: 14px; line-height: 1.5; }}
            .detail-row {{ margin-bottom: 16px; }}
            .detail-label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }}
            .detail-value {{ font-size: 15px; color: #111827; font-weight: 500; }}
            .message-preview {{ background-color: #f9fafb; border: 1px solid #e5e7eb; border-radius: 6px; padding: 14px; font-style: italic; color: #374151; font-size: 14px; line-height: 1.5; }}
            .resources {{ background-color: #eff6ff; border-radius: 6px; padding: 20px; margin-top: 24px; }}
            .resources h3 {{ color: #1e40af; margin: 0 0 12px 0; font-size: 16px; }}
            .resources ul {{ margin: 0; padding-left: 20px; color: #1e3a8a; font-size: 14px; line-height: 1.8; }}
            .resources a {{ color: #2563eb; text-decoration: none; font-weight: 500; }}
            .footer {{ background-color: #f9fafb; padding: 20px; text-align: center; border-top: 1px solid #e5e7eb; }}
            .footer p {{ color: #9ca3af; font-size: 12px; margin: 0; }}
            .footer-brand {{ color: #4b5563; font-weight: 600; font-size: 13px; margin-bottom: 6px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <div class="header-icon">🚨</div>
                <h1>Emergency Crisis Alert</h1>
            </div>
            <div class="content">
                <div class="alert-box">
                    <h2>High-Risk Crisis Detected</h2>
                    <p>Our AI system has detected potentially harmful language indicating a high risk of self-harm or suicidal ideation during your therapy session. Your safety is our highest priority.</p>
                </div>

                <div class="detail-row">
                    <div class="detail-label">User</div>
                    <div class="detail-value">{user_name or 'Unknown User'}</div>
                </div>

                <div class="detail-row">
                    <div class="detail-label">Session ID</div>
                    <div class="detail-value">{session_id}</div>
                </div>

                <div class="detail-row">
                    <div class="detail-label">Timestamp</div>
                    <div class="detail-value">{timestamp}</div>
                </div>

                <div class="detail-row">
                    <div class="detail-label">Flagged Message Preview</div>
                    <div class="message-preview">"{message_preview}"</div>
                </div>

                <div class="resources">
                    <h3>Immediate Help Resources</h3>
                    <ul>
                        <li><strong>National Suicide Prevention Lifeline:</strong> Call or text <a href="tel:988">988</a></li>
                        <li><strong>Crisis Text Line:</strong> Text <strong>HOME</strong> to <a href="sms:741741">741741</a></li>
                        <li><strong>Emergency Services:</strong> Call <a href="tel:911">911</a> (US) or your local emergency number</li>
                        <li><strong>International Association for Suicide Prevention:</strong> <a href="https://www.iasp.info/resources/Crisis_Centres/">Find a crisis center near you</a></li>
                    </ul>
                </div>
            </div>
            <div class="footer">
                <div class="footer-brand">CogniCove AI Therapy Assistant</div>
                <p>This is an automated alert. Please reach out to a mental health professional or emergency service if you are in crisis.</p>
            </div>
        </div>
    </body>
    </html>
    """


def send_crisis_alert_email(
    to_email: str,
    user_name: Optional[str] = None,
    session_id: Optional[str] = None,
    message_preview: Optional[str] = None,
) -> bool:
    """
    Send a crisis alert email to the user.

    Args:
        to_email: Recipient email address
        user_name: Name of the user
        session_id: Therapy session ID
        message_preview: Preview of the flagged message

    Returns:
        True if email was sent successfully, False otherwise
    """
    if not SMTP_PASSWORD:
        print("⚠️  SMTP_PASSWORD not configured. Crisis alert email not sent.")
        return False

    if not to_email:
        print("⚠️  No recipient email provided. Crisis alert email not sent.")
        return False

    try:
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
        safe_preview = (message_preview or "N/A")[:300]
        if len(message_preview or "") > 300:
            safe_preview += "..."

        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "🚨 URGENT: Crisis Alert from CogniCove"
        msg["From"] = f"CogniCove Alerts <{SMTP_FROM_EMAIL}>"
        msg["To"] = to_email

        # Plain text version
        text_body = f"""
EMERGENCY CRISIS ALERT - CogniCove

Our AI system has detected high-risk language indicating potential self-harm or suicidal ideation.

User: {user_name or 'Unknown'}
Session: {session_id or 'N/A'}
Time: {timestamp}
Flagged Message: {safe_preview}

IMMEDIATE HELP RESOURCES:
- National Suicide Prevention Lifeline: Call or text 988
- Crisis Text Line: Text HOME to 741741
- Emergency Services: Call 911

Please reach out to a mental health professional or emergency service if you are in crisis.
        """.strip()

        # HTML version
        html_body = _build_crisis_email_html(
            user_name=user_name,
            session_id=session_id or "N/A",
            message_preview=safe_preview,
            timestamp=timestamp,
        )

        msg.attach(MIMEText(text_body, "plain"))
        msg.attach(MIMEText(html_body, "html"))

        # Connect and send
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_FROM_EMAIL, to_email, msg.as_string())

        print(f"✅ Crisis alert email sent successfully to {to_email}")
        return True

    except Exception as e:
        print(f"❌ Failed to send crisis alert email: {str(e)}")
        return False
