import streamlit as st
import pandas as pd
import json
import os
import sqlite3
from datetime import datetime
from typing import List, Literal, Optional
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# LangChain Imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from pydantic import BaseModel, ConfigDict

# ==============================
# 1️⃣ Environment Setup
# ==============================
load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY")

# ==============================
# 2️⃣ Data Models
# ==============================
class UnifiedResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    intent: Literal["DB_QUERY", "GENERAL_CHAT", "SECURITY_VIOLATION"]
    language: Literal["ar", "en"]
    sql_query: Optional[str] = None
    relevant_tables: Optional[List[str]] = None
    chat_response: Optional[str] = None


# ==============================
# 3️⃣ Ultra SQL Agent
# ==============================
class UltraSQLAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-lite",
            temperature=0
        )

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

        self.engine = create_engine(os.getenv("DB_URL"))

        self._init_history_db()
        self.vector_db = self._prepare_rag()

    # --------------------------
    # Persistent History
    # --------------------------
    def _init_history_db(self):
        conn = sqlite3.connect('persistent_history.db')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS messages (
                session_id TEXT,
                role TEXT,
                content TEXT,
                dataframe_json TEXT,
                timestamp DATETIME
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                title TEXT
            )
        ''')
        conn.commit()
        self.hist_conn = conn

    # --------------------------
    # RAG Setup (FAISS Cached on Disk ✅)
    # --------------------------
    def _prepare_rag(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(base_dir, "fewshots.json")
        faiss_path = os.path.join(base_dir, "faiss_index")

        if os.path.exists(faiss_path):
            return FAISS.load_local(
                faiss_path,
                self.embeddings,
                allow_dangerous_deserialization=True
            )

        if not os.path.exists(file_path):
            st.error(f"fewshots.json not found at: {file_path}")
            st.stop()

        with open(file_path, 'r', encoding='utf-8') as f:
            shots = json.load(f)

        docs = [
            Document(
                page_content=s['naturalQuestion'],
                metadata={'sql': s['sqlQuery']}
            )
            for s in shots
        ]

        vector_db = FAISS.from_documents(docs, self.embeddings)
        vector_db.save_local(faiss_path)

        return vector_db

    # --------------------------
    # Unified Process Query (1 Call ✅)
    # --------------------------
    def process_query(self, question: str, examples: str = ""):
        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant with access to a PostgreSQL database.

        Analyze the user's message and respond ONLY with a valid JSON object. No markdown, no explanation, just JSON.

        If it's a greeting or general conversation:
        {{"intent": "GENERAL_CHAT", "language": "ar", "sql_query": null, "relevant_tables": null, "chat_response": "ردك هنا بنفس لغة المستخدم"}}

        If it's a database question:
        {{"intent": "DB_QUERY", "language": "en", "sql_query": "SELECT ...", "relevant_tables": ["Table1", "Table2"], "chat_response": null}}

        If it contains DROP/DELETE/UPDATE:
        {{"intent": "SECURITY_VIOLATION", "language": "en", "sql_query": null, "relevant_tables": null, "chat_response": null}}

        PostgreSQL Rules (for DB_QUERY only):
        1. Only use SELECT queries.
        2. Always use double quotes for table and column names.
        3. Date columns like "InvoiceDate" are stored as TEXT, so always CAST them before any date operation:
           - For DATE_TRUNC: DATE_TRUNC('month', "InvoiceDate"::TIMESTAMP)
           - For EXTRACT: EXTRACT(YEAR FROM "InvoiceDate"::TIMESTAMP)
        4. For grouping by month use the full expression, not the alias:
           GROUP BY DATE_TRUNC('month', "InvoiceDate"::TIMESTAMP)
        5. Always GROUP BY the full expression, not the alias.

        Few-shot examples for guidance:
        {examples}

        User message: {question}
        """)

        chain = prompt | self.llm | PydanticOutputParser(
            pydantic_object=UnifiedResponse
        )

        return chain.invoke({
            "question": question,
            "examples": examples
        })

    # --------------------------
    # Save Message to History
    # --------------------------
    def save_message(self, session_id, role, content, df=None):
        df_json = df.to_json(orient="records", force_ascii=False) if df is not None else None
        self.hist_conn.execute(
            "INSERT INTO messages VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, df_json, datetime.now())
        )
        self.hist_conn.commit()


# ==============================
# 4️⃣ Streamlit UI
# ==============================
st.set_page_config(
    page_title="Ultra SQL Agent",
    layout="wide"
)

agent = UltraSQLAgent()

# Sidebar
with st.sidebar:
    st.title("Chat History")

    if st.button("New Chat"):
        st.session_state.session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        agent.hist_conn.execute(
            "INSERT INTO sessions VALUES (?, ?)",
            (st.session_state.session_id,
             f"Chat {st.session_state.session_id[-4:]}")
        )
        agent.hist_conn.commit()
        st.rerun()

    sessions = pd.read_sql(
        "SELECT * FROM sessions ORDER BY session_id DESC",
        agent.hist_conn
    )

    for _, s in sessions.iterrows():
        if st.button(s['title'], key=s['session_id']):
            st.session_state.session_id = s['session_id']
            st.rerun()

# ==============================
# 5️⃣ Chat Logic
# ==============================
if "session_id" not in st.session_state:
    st.info("👈 Start a New Chat from the sidebar.")
    st.stop()

# تحميل الـ history وعرضه مع الجداول لو موجودة
history = pd.read_sql(
    """
    SELECT * FROM messages
    WHERE session_id = ?
    ORDER BY timestamp
    """,
    agent.hist_conn,
    params=[st.session_state.session_id]
)

for _, msg in history.iterrows():
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])
        if msg['dataframe_json'] and msg['dataframe_json'] != 'null':
            try:
                df_restored = pd.read_json(msg['dataframe_json'], orient="records")
                st.caption(f"Showing {len(df_restored)} rows")
                st.dataframe(df_restored, use_container_width=True)
            except Exception:
                pass

if prompt := st.chat_input("Ask your data..."):

    agent.save_message(st.session_state.session_id, "user", prompt)

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):

        # جيب الـ examples من RAG
        examples = ""
        if agent.vector_db:
            similar_docs = agent.vector_db.similarity_search(prompt, k=3)
            examples = "\n".join([
                f"Q: {d.page_content} -> SQL: {d.metadata['sql']}"
                for d in similar_docs
            ])

        # Call واحدة بس 🎉
        result = agent.process_query(prompt, examples)

        if result.intent == "SECURITY_VIOLATION":
            response = "I don't have permission to modify data."
            agent.save_message(st.session_state.session_id, "assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)

        elif result.intent == "GENERAL_CHAT":
            response = result.chat_response
            agent.save_message(st.session_state.session_id, "assistant", response)
            with st.chat_message("assistant"):
                st.markdown(response)

        else:
            try:
                with agent.engine.connect() as conn:
                    df = pd.read_sql(text(result.sql_query), con=conn)

                # Call تانية للـ Summary بس (مختصر ✅)
                summary_prompt = f"""
                Answer the question: '{prompt}'
                based on this data:
                {df.to_string()}
                in {result.language}.
                Be concise - write 2-3 sentences maximum summarizing the key insights only.
                Do NOT list all the data rows in your answer, the data table is already shown separately.
                """
                response = agent.llm.invoke(summary_prompt).content

                # احفظ الرسالة مع الجدول
                agent.save_message(st.session_state.session_id, "assistant", response, df=df)

                with st.chat_message("assistant"):
                    st.markdown(response)
                    st.caption(f"Showing {len(df)} rows")
                    st.dataframe(df, use_container_width=True)

            except Exception as e:
                response = f"Query Error: {str(e)}\n\nSQL attempted: {result.sql_query}"
                agent.save_message(st.session_state.session_id, "assistant", response)
                with st.chat_message("assistant"):
                    st.markdown(response)
###  streamlit run "c:/Users/hussein aly/Downloads/Chinhook/Chinhook/APP.py"   ###


### streamlit run .\Chinhook\APP.py