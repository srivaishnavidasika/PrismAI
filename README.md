# 🧠 PrismAI

PrismAI is an AI-powered multi-agent coding assistant that analyzes, explains, fixes, scores, and generates personalized practice for student code across multiple programming languages.

Built with **FastAPI + Groq LLM + Modular Multi-Agent Architecture**.

---

## 🚀 Problem Statement

Students learning to code often struggle with:

- Lack of personalized feedback
- Understanding logical mistakes
- Identifying recurring weaknesses
- Getting targeted practice for improvement

Most tools only check syntax or give generic responses.

**PrismAI solves this by acting as an adaptive AI tutor** that not only analyzes code but learns from user mistakes and generates personalized improvement plans.

---

## 🚀 Features

### 🔎 Code Analysis

- Detects syntax errors
- Identifies logical mistakes
- Highlights inefficiencies
- Assigns clarity score (0–10)
- Generates concise summary

### 🛠 Fix Agent

- Identifies root issue
- Returns corrected code
- Explains the fix clearly

### 📘 Explanation Agent

- Beginner mode (simple breakdown)
- Interview mode (deeper technical explanation)
- Step-by-step logic explanation

### 📊 Scoring Agent

- Syntax score
- Logic score
- Clarity score
- Robustness score
- Overall evaluation (0–10)

### 🧩 Practice Agent

- Generates similar problems
- Provides a challenge problem
- Offers structured hints

### 🧠 Adaptive Learning Agent

- Tracks recurring user mistakes
- Identifies common weakness patterns
- Generates 5 targeted practice questions
- Builds personalized improvement plans

---

## 🏗 Architecture

PrismAI uses a modular multi-agent pipeline:

- `analyzer_agent`
- `pedagogy_agent`
- `fix_agent`
- `practice_agent`
- `scoring_agent`
- `adaptive_feedback_agent`
- `memory_agent`
- `intent_router`

All agents are orchestrated through a centralized:

```
run_pipeline()
```

Each agent operates independently with:

- Structured JSON output
- Timeout protection
- Graceful fallback handling
- Safe parsing mechanisms

---

## 🛡 Safety & Reliability

- Prompt injection detection
- Input validation (max 5000 characters)
- Graceful fallback responses (no crashes)
- No thread deadlocks
- Safe JSON parsing
- CORS enabled for frontend integration

---

## 🌐 API Endpoint

### POST `/run`

**Request Body:**

```json
{
  "code": "string",
  "language": "c | cpp | java | python | csharp",
  "mode": "beginner | interview",
  "user_query": "string",
  "user_id": "string",
  "intent": "analyze | explain | fix | practice | score | full_review | adaptive"
}
```

---

## 🧾 Example Full Review Response

```json
{
  "analysis": {...},
  "explanation": "...",
  "fix": {...},
  "practice": {...},
  "score": {...}
}
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository

```bash
git clone <repo-url>
cd PrismAI
```

### 2️⃣ Create virtual environment

```bash
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Add environment variables

Create a `.env` file:

```
GROQ_API_KEY=your_api_key_here
```

### 5️⃣ Run the server

```bash
uvicorn app.main:app --reload
```

Open:

```
http://127.0.0.1:8000/docs
```

---

PrismAI/
│
├── .vscode/
│
├── backend/
│ ├── app/
│ │ ├── **pycache**/
│ │ ├── agents/
│ │ │ ├── **pycache**/
│ │ │ ├── analyzer_agent.py
│ │ │ ├── execution_agent.py
│ │ │ ├── fix_agent.py
│ │ │ ├── intent_router.py
│ │ │ ├── memory_agent.py
│ │ │ ├── mistake_fixer_agent.py
│ │ │ ├── pedagogy_agent.py
│ │ │ ├── practice_agent.py
│ │ │ └── scoring_agent.py
│ │ │
│ │ ├── memory/
│ │ │ ├── **pycache**/
│ │ │ ├── memory_data.json
│ │ │ └── memory_store.py
│ │ │
│ │ ├── services/
│ │ │ ├── **pycache**/
│ │ │ ├── llm_service.py
│ │ │ ├── pipeline.py
│ │ │ └── sandbox.py
│ │ │
│ │ ├── utils/
│ │ │ ├── **pycache**/
│ │ │ ├── formatters.py
│ │ │ ├── injection_guard.py
│ │ │ └── validators.py
│ │ │
│ │ ├── config.py
│ │ ├── main.py
│ │ └── schemas.py
│ │
│ ├── tests/
│ │ ├── test_pipeline.py
│ │ └── test_security.py
│ │
│ ├── venv/
│ │ ├── Include/
│ │ ├── Lib/
│ │ ├── Scripts/
│ │ ├── .gitignore
│ │ └── pyvenv.cfg
│ │
│ ├── .env
│ ├── requirements.txt
│ │
│ └── docs/
│ └── test_cases.md
│
├── frontend/
│ ├── app.js
│ ├── index.html
│ ├── prismai-logo.png
│ └── styles.css
│
└── README.md

```

## 🧠 What Makes This Unique?

- True multi-agent architecture (not a single LLM call)
- Structured JSON outputs for frontend stability
- Adaptive personalized feedback engine
- Memory-based improvement tracking
- Graceful fallback system (no infinite spinners)
- Language-agnostic extensibility

---

## ⚠ Limitations

- LLM output depends on model availability
- Practice generation may vary in difficulty

---

## 🎯 Vision

PrismAI is not just a code analyzer.

It is a personalized AI learning engine designed to:

- Identify weaknesses
- Track improvement
- Adapt difficulty
- Guide learners toward mastery

---

## 🔮 Future Improvements

- Performance benchmarking
- Progress tracking dashboard
- Skill-level progression system
- Model auto-fallback strategy

---

## 🧑‍💻 Authors

Built for hackathon innovation.  
Designed to combine AI reasoning with structured educational feedback.

---

⭐ If you like this project, give it a star.
