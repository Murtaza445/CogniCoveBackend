# Next.js Integration Guide

## FastAPI Backend Setup

The `main.py` file is a FastAPI server that provides all the backend functionality for your disorder chat application.

### Running the FastAPI Server

```bash
# Install dependencies if not already installed
pip install fastapi uvicorn

# Run the server
python main.py

# Or with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`
- API docs: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## API Endpoints Reference

### 1. Create Session
**POST** `/api/sessions/create`

Request:
```json
{
  "user_id": "optional_user_id"
}
```

Response:
```json
{
  "session_id": "session_1234567890",
  "created_at": "2024-03-26T10:30:00"
}
```

---

### 2. Get Session Data
**GET** `/api/sessions/{session_id}`

Response:
```json
{
  "session_id": "session_1234567890",
  "user_id": "user_123",
  "created_at": "2024-03-26T10:30:00",
  "messages": [
    {
      "role": "user",
      "content": "I'm feeling anxious",
      "timestamp": "2024-03-26T10:31:00"
    },
    {
      "role": "assistant",
      "content": "I understand you're feeling anxious...",
      "timestamp": "2024-03-26T10:31:05"
    }
  ],
  "summary": null,
  "classification": null,
  "diagnostic_results": null,
  "session_ended": false
}
```

---

### 3. Send Therapy Message
**POST** `/api/therapy`

Request:
```json
{
  "session_id": "session_1234567890",
  "content": "I'm feeling anxious and stressed"
}
```

Response:
```json
{
  "session_id": "session_1234567890",
  "user_message": "I'm feeling anxious and stressed",
  "assistant_message": "I hear you. Anxiety can be overwhelming...",
  "timestamp": "2024-03-26T10:31:05"
}
```

---

### 4. Generate Clinical Summary
**POST** `/api/summary`

Request:
```json
{
  "session_id": "session_1234567890"
}
```

Response:
```json
{
  "session_id": "session_1234567890",
  "summary": "{\"symptoms\": [...], \"mood_patterns\": [...]}",
  "timestamp": "2024-03-26T10:35:00"
}
```

---

### 5. Classify Disorder
**POST** `/api/classify`

Request:
```json
{
  "session_id": "session_1234567890",
  "summary": "{...json summary...}"
}
```

Response:
```json
{
  "session_id": "session_1234567890",
  "classification": "{\"primary_category\": \"Depression\", \"confidence\": 0.85}",
  "comorbid_categories": ["Anxiety Disorder", "Sleep Disorder"],
  "timestamp": "2024-03-26T10:36:00"
}
```

---

### 6. Generate Diagnostic Results
**POST** `/api/diagnose`

Request:
```json
{
  "session_id": "session_1234567890",
  "summary": "{...json summary...}",
  "classification": "{...json classification...}"
}
```

Response:
```json
{
  "session_id": "session_1234567890",
  "diagnostic_results": [
    "{\"disorder_name\": \"Major Depressive Disorder\", \"confidence_level\": \"High\", ...}",
    "{\"disorder_name\": \"Generalized Anxiety Disorder\", \"confidence_level\": \"Moderate\", ...}"
  ],
  "timestamp": "2024-03-26T10:37:00"
}
```

---

### 7. End Session
**POST** `/api/sessions/{session_id}/end`

Response:
```json
{
  "session_id": "session_1234567890",
  "summary": "{...}",
  "classification": "{...}",
  "diagnostic_results": [...],
  "ended_at": "2024-03-26T10:45:00"
}
```

---

### 8. Delete Session
**DELETE** `/api/sessions/{session_id}`

Response:
```json
{
  "message": "Session session_1234567890 deleted"
}
```

---

## Next.js Example Implementation

Create a custom hook `hooks/useTherapyAPI.ts`:

```typescript
import { useState } from 'react';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
}

interface SessionData {
  session_id: string;
  messages: Message[];
  summary: string | null;
  classification: string | null;
  diagnostic_results: string[] | null;
  session_ended: boolean;
}

export const useTherapyAPI = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const createSession = async (userId?: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/sessions/create`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: userId }),
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create session');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getSession = async (sessionId: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}`);
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch session');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async (sessionId: string, content: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/therapy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, content }),
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to send message');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const generateSummary = async (sessionId: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/summary`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate summary');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const classifyDisorder = async (sessionId: string, summary: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/classify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, summary }),
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify disorder');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const generateDiagnosis = async (sessionId: string, summary: string, classification: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/diagnose`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, summary, classification }),
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate diagnosis');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const endSession = async (sessionId: string) => {
    try {
      setLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/sessions/${sessionId}/end`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      return await response.json();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to end session');
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    createSession,
    getSession,
    sendMessage,
    generateSummary,
    classifyDisorder,
    generateDiagnosis,
    endSession,
    loading,
    error,
  };
};
```

---

## Example Next.js Page

```typescript
'use client';

import { useState, useEffect } from 'react';
import { useTherapyAPI } from '@/hooks/useTherapyAPI';

export default function ChatPage() {
  const api = useTherapyAPI();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<any[]>([]);
  const [input, setInput] = useState('');
  const [sessionEnded, setSessionEnded] = useState(false);

  // Create session on mount
  useEffect(() => {
    const initSession = async () => {
      const data = await api.createSession();
      setSessionId(data.session_id);
    };
    initSession();
  }, []);

  // Send message
  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!sessionId || !input.trim()) return;

    try {
      const response = await api.sendMessage(sessionId, input);
      setMessages([
        ...messages,
        { role: 'user', content: input, timestamp: new Date().toISOString() },
        {
          role: 'assistant',
          content: response.assistant_message,
          timestamp: response.timestamp,
        },
      ]);
      setInput('');
    } catch (err) {
      console.error('Error sending message:', err);
    }
  };

  // End session and generate results
  const handleEndSession = async () => {
    if (!sessionId) return;

    try {
      // Generate summary
      const summary = await api.generateSummary(sessionId);

      // Classify disorder
      const classification = await api.classifyDisorder(sessionId, summary.summary);

      // Generate diagnosis
      const diagnosis = await api.generateDiagnosis(
        sessionId,
        summary.summary,
        classification.classification
      );

      // End session
      await api.endSession(sessionId);

      setSessionEnded(true);
    } catch (err) {
      console.error('Error ending session:', err);
    }
  };

  if (!sessionId) return <div>Loading...</div>;

  return (
    <div className="flex flex-col h-screen">
      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4">
        {messages.map((msg, idx) => (
          <div key={idx} className={`mb-4 ${msg.role === 'user' ? 'text-right' : ''}`}>
            <div
              className={`inline-block p-2 rounded ${
                msg.role === 'user' ? 'bg-blue-500 text-white' : 'bg-gray-200'
              }`}
            >
              {msg.content}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      {!sessionEnded && (
        <form onSubmit={handleSendMessage} className="p-4 border-t">
          <div className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="How are you feeling today?"
              className="flex-1 p-2 border rounded"
              disabled={api.loading}
            />
            <button
              type="submit"
              className="px-4 py-2 bg-blue-500 text-white rounded"
              disabled={api.loading}
            >
              Send
            </button>
            <button
              type="button"
              onClick={handleEndSession}
              className="px-4 py-2 bg-red-500 text-white rounded"
              disabled={api.loading}
            >
              End Session
            </button>
          </div>
        </form>
      )}
    </div>
  );
}
```

---

## Environment Variables

In your Next.js `.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Notes

- The FastAPI server uses in-memory session storage. For production, implement database persistence.
- All timestamps are in ISO 8601 format (UTC).
- CORS is configured to allow requests from `localhost:3000`.
- For production, update the `allow_origins` list in `main.py`.
