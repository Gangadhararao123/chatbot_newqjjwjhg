#OPENROUTER_API_KEY = "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"
import streamlit as st
import io
import httpx
from pypdf import PdfReader
from docx import Document
from PIL import Image
import pytesseract

# ================= CONFIG =================
# Get API key from Streamlit secrets
OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
MODEL_NAME = "mistralai/mistral-7b-instruct:free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
# =========================================

# Check API key
if not OPENROUTER_API_KEY:
    st.error("⚠️ OpenRouter API key is missing. Add it to Streamlit secrets!")
    st.stop()

# ---------- Page Config ----------
st.set_page_config(page_title="Document RAG Chatbot", layout="wide")
st.title("📄 Document Q&A Chatbot (RAG)")
st.caption("Upload a document and ask multiple questions")

# ---------- Session State ----------
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "history" not in st.session_state:
    st.session_state.history = []

# ---------- Helper Functions ----------
def extract_text(file, name):
    """Extract text from PDF, TXT, DOCX, or Image"""
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
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    return ""

def chunk_text(text, size=400):
    """Split text into chunks"""
    return [text[i:i+size] for i in range(0, len(text), size)]

def retrieve(chunks, question, top_k=3):
    """Simple keyword-based retrieval"""
    q_words = set(question.lower().split())
    scored = []
    for c in chunks:
        score = len(q_words & set(c.lower().split()))
        scored.append((score, c))
    scored.sort(reverse=True)
    return [c for _, c in scored[:top_k]]

def ask_llm(context, question, history):
    """Send context + question to OpenRouter LLM"""
    messages = [
        {
            "role": "system",
            "content": (
                "Answer ONLY using the provided context. "
                "If answer not present, say: 'Not found in document.'\n\n"
                + context
            )
        }
    ]
    messages.extend(history)
    messages.append({"role": "user", "content": question})

    payload = {"model": MODEL_NAME, "messages": messages}
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }

    try:
        with httpx.Client(timeout=60) as client:
            res = client.post(OPENROUTER_URL, json=payload, headers=headers)
        if res.status_code != 200:
            return f"❌ API Error {res.status_code}: {res.text}"
        data = res.json()
        if "choices" not in data:
            return f"❌ LLM Error: {data}"
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"❌ Exception: {e}"

# ---------- File Upload ----------
uploaded_file = st.file_uploader(
    "Upload PDF / DOCX / TXT / Image",
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg"]
)

if uploaded_file:
    text = extract_text(uploaded_file, uploaded_file.name.lower())
    if text.strip():
        st.session_state.chunks = chunk_text(text[:6000])  # limit for API
        st.session_state.history = []
        st.success("✅ Document processed successfully!")

# ---------- Chat UI ----------
st.divider()
st.subheader("💬 Ask Questions")

question = st.text_input("Type your question here")

col1, col2 = st.columns([1, 1])
with col1:
    ask_btn = st.button("Ask")
with col2:
    reset_btn = st.button("Reset Chat")

if reset_btn:
    st.session_state.history = []
    st.success("🔄 Chat reset")

if ask_btn and question:
    if not st.session_state.chunks:
        st.warning("⚠️ Upload a document first")
    else:
        with st.spinner("Thinking..."):
            context = "\n\n".join(retrieve(st.session_state.chunks, question))
            answer = ask_llm(context, question, st.session_state.history)
        st.session_state.history.append({"role": "user", "content": question})
        st.session_state.history.append({"role": "assistant", "content": answer})

# ---------- Display chat ----------
st.divider()
for msg in st.session_state.history:
    if msg["role"] == "user":
        st.markdown(f"🧑 **You:** {msg['content']}")
    else:
        st.markdown(f"🤖 **Bot:** {msg['content']}")
