# 📊 Ultra SQL Agent

An AI-powered **PostgreSQL Chat Agent** built with:

* 🧠 Google Gemini (LLM)
* 🔎 RAG using FAISS
* 🗄️ SQL Generation with Guardrails
* 💬 Persistent Chat History
* ⚡ Streamlit UI
* 🔐 Built-in Security Layer

---

## 🚀 Overview

**Ultra SQL Agent** is an intelligent data assistant that:

* Understands natural language questions
* Generates safe SQL queries automatically
* Uses Retrieval-Augmented Generation (RAG) for better accuracy
* Prevents destructive queries (DROP / DELETE / UPDATE)
* Stores chat history persistently
* Displays query results with explanations

All in **one unified LLM call** for efficiency.

---

## 🏗️ Architecture

```
User Question
      ↓
LLM (Intent + SQL + Language Detection)
      ↓
If DB Query:
    → Retrieve similar examples (FAISS RAG)
    → Execute safe SELECT query
    → Generate concise summary
      ↓
Display:
    - Table
    - AI Summary
    - Stored History
```

---

## ✨ Features

### 🔹 1. Unified LLM Pipeline

Single structured output using Pydantic:

* Intent Detection
* SQL Generation
* Language Detection
* Security Classification

---

### 🔹 2. RAG with FAISS

* Few-shot examples stored in `fewshots.json`
* Vector search for similar queries
* Local FAISS index caching
* Improves SQL accuracy

---

### 🔹 3. Security Layer

Automatically blocks:

* `DROP`
* `DELETE`
* `UPDATE`

Only allows:

* Safe `SELECT` queries

---

### 🔹 4. PostgreSQL Best Practices

* Forced double quotes for identifiers
* Proper `TIMESTAMP` casting for date fields
* Safe grouping rules
* Date truncation handling

---

### 🔹 5. Persistent Chat Memory

* SQLite-based storage
* Session management
* Restore previous chats
* Store DataFrame results

---

### 🔹 6. Clean Streamlit Interface

* Sidebar session manager
* New chat creation
* Chat history loading
* Table rendering
* Auto summary generation

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit**
* **LangChain**
* **Google Gemini**
* **FAISS**
* **PostgreSQL**
* **SQLite**
* **SQLAlchemy**
* **Pydantic**

---

## 📦 Installation

```bash
git clone https://github.com/your-username/ultra-sql-agent.git
cd ultra-sql-agent

pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a `.env` file:

```env
DB_URL=postgresql+psycopg2://user:password@host:port/database
LANGSMITH_API_KEY=your_langsmith_key
GOOGLE_API_KEY=your_google_api_key
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
ultra-sql-agent/
│
├── app.py
├── fewshots.json
├── faiss_index/
├── persistent_history.db
├── .env
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works (Technical Deep Dive)

### 1️⃣ Intent Classification

The model outputs:

* `DB_QUERY`
* `GENERAL_CHAT`
* `SECURITY_VIOLATION`

Using structured JSON parsing.

---

### 2️⃣ RAG Enhancement

* Converts few-shot examples into embeddings
* Uses FAISS similarity search
* Injects top-k examples into prompt
* Improves SQL generation accuracy

---

### 3️⃣ Execution Safety

* SQL parsed only if intent = DB_QUERY
* Queries wrapped with `sqlalchemy.text`
* No write operations allowed

---

### 4️⃣ Response Strategy

For DB queries:

* Execute SQL
* Show table
* Generate 2–3 sentence summary
* Store everything in history

---

## 🧪 Example Use Cases

* Sales analytics
* Financial dashboards
* Invoice reporting
* Monthly aggregations
* Data exploration
* Business intelligence assistant

---

## 🔮 Future Improvements

* Multi-agent architecture
* Automatic query correction loop
* Query plan validation
* Role-based database access
* Advanced caching
* Streaming responses
* Docker deployment
* Cloud deployment (GCP / Azure)

---

## 🛡️ Security Notes

This system:

* Prevents destructive queries
* Uses structured output validation
* Enforces SQL rules
* Avoids raw LLM execution

Still recommended to:

* Use read-only DB user
* Restrict DB permissions at database level
