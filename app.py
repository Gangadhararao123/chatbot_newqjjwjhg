import io
import httpx
import streamlit as st
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# ================= CONFIG =================
OPENROUTER_API_KEY = "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"
MODEL_NAME = "mistralai/mistral-7b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# =========================================

st.set_page_config(page_title="Document RAG Chat", layout="centered")

# ---------- SESSION STATE ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "history" not in st.session_state:
    st.session_state.history = []

# ---------- RAG HELPERS ----------

def extract_text(file, name):
    data = file.read()

    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "".join(p.extract_text() or "" for p in reader.pages)

    elif name.endswith(".txt"):
        return data.decode()

    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        return "\n".join(p.text for p in doc.paragraphs)

    elif name.endswith((".png", ".jpg", ".jpeg")):
        return pytesseract.image_to_string(
            Image.open(io.BytesIO(data))
        )

    return ""


def chunk_text(text, size=400):
    return [text[i:i+size] for i in range(0, len(text), size)]


def retrieve(chunks, question, top_k=3):
    q_words = set(question.lower().split())
    scored = []

    for c in chunks:
        score = len(q_words & set(c.lower().split()))
        scored.append((score, c))

    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]


def ask_llm(context, question, history):
    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY from the provided context. "
                "If not found say: Not found in document."
            )
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{question}"
        }
    ] + history

    payload = {"model": MODEL_NAME, "messages": messages}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    res = httpx.post(
        OPENROUTER_URL,
        json=payload,
        headers=headers,
        timeout=60
    )

    return res.json()["choices"][0]["message"]["content"]

# ---------- UI ----------

st.title("📄 Document Q&A Chatbot (RAG)")

uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT / Image",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    text = extract_text(uploaded_file, uploaded_file.name.lower())

    if text.strip():
        st.session_state.chunks = chunk_text(text[:6000])
        st.session_state.history = []
        st.success("File uploaded successfully!")

# ---------- CHAT ----------
if st.session_state.chunks:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    question = st.chat_input("Ask a question...")

    if question:
        st.session_state.history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                context = "\n\n".join(
                    retrieve(st.session_state.chunks, question)
                )
                answer = ask_llm(
                    context,
                    question,
                    st.session_state.history
                )
                st.write(answer)

        st.session_state.history.append(
            {"role": "assistant", "content": answer}
        )

    if st.button("🔄 Reset Chat"):
        st.session_state.history = []
        st.experimental_rerun()
