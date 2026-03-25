"""
app.py — Day 6: Ask Your Tax PDF
One screen: upload PDF → ask question → see answer + source chunks

Run:     streamlit run app.py
Install: pip install streamlit openai pymupdf numpy
"""

import json
import tempfile
import numpy as np
import streamlit as st
from openai import AzureOpenAI
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env file

endpoint    = os.getenv("AZURE_OPENAI_ENDPOINT", "")
api_key     = os.getenv("AZURE_OPENAI_API_KEY", "")
chat_model  = os.getenv("AZURE_OPENAI_CHAT_MODEL", "gpt-4o")
embed_model = os.getenv("AZURE_OPENAI_EMBED_MODEL", "text-embedding-ada-002")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Day 6 — Ask Your Tax PDF", page_icon="📄", layout="wide")
st.title("📄 Day 6 — Ask Your Tax PDF")
st.caption("Upload a tax circular or notification · Ask a question · Get a cited answer")

# ── Guard: credentials required ───────────────────────────────────────────────
if not all([endpoint, api_key, chat_model, embed_model]):
    st.info("👈  Fill in your Azure OpenAI credentials in the sidebar to begin.")
    st.stop()

client = AzureOpenAI(
    azure_endpoint = endpoint,
    api_key        = api_key,
    api_version    = "2024-08-01-preview",
)

# ── Helper functions ──────────────────────────────────────────────────────────

def load_and_chunk(pdf_bytes, chunk_size=400, overlap=50):
    """Load PDF bytes, extract text, split into overlapping word chunks."""
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name
    doc   = fitz.open(tmp_path)
    text  = "\n".join(page.get_text() for page in doc)
    doc.close()
    words  = text.split()
    step   = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        w = words[i : i + chunk_size]
        if len(w) >= 30:
            chunks.append({"id": len(chunks), "text": " ".join(w)})
    return chunks


def embed_all(chunks):
    """Embed every chunk and add a 'vector' field to each."""
    for chunk in chunks:
        chunk["vector"] = client.embeddings.create(
            model=embed_model, input=chunk["text"]
        ).data[0].embedding
    return chunks


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def search(question, chunks, top_k=3):
    """Embed the question, score every chunk, return top_k."""
    q_vec = client.embeddings.create(
        model=embed_model, input=question
    ).data[0].embedding
    for chunk in chunks:
        chunk["score"] = cosine_similarity(q_vec, chunk["vector"])
    return sorted(chunks, key=lambda c: c["score"], reverse=True)[:top_k]


def answer_question(question, relevant_chunks):
    """Send question + chunks to GPT. Ask it to answer from context only."""
    context = "\n\n---\n\n".join(
        f"[Chunk {c['id']}]\n{c['text']}" for c in relevant_chunks
    )
    system = (
        "You are a tax expert. Answer using ONLY the context below. "
        "Do not use your training knowledge. "
        "End your answer with: Source: Chunk N  (the chunk you used). "
        "If the answer is not in the context, say so.\n\n"
        f"CONTEXT:\n{context}"
    )
    response = client.chat.completions.create(
        model=chat_model,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": question}],
        temperature=0.0,
        max_tokens=400,
    )
    return response.choices[0].message.content

# ── Main layout: two columns ──────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

# ── LEFT: Upload and Index ─────────────────────────────────────────────────────
with left:
    st.subheader("Step 1 — Upload & Index")

    uploaded = st.file_uploader("Choose a tax circular PDF", type="pdf")

    if uploaded and st.session_state.get("pdf_name") != uploaded.name:
        with st.spinner(f"Reading and embedding {uploaded.name}..."):
            chunks = load_and_chunk(uploaded.read())
            chunks = embed_all(chunks)
        st.session_state.chunks   = chunks
        st.session_state.pdf_name = uploaded.name

    if "chunks" in st.session_state:
        chunks = st.session_state.chunks
        st.success(f"✅ **{st.session_state.pdf_name}** — {len(chunks)} chunks indexed")

        with st.expander("Preview chunks"):
            for i, c in enumerate(chunks[:3]):
                st.markdown(f"**Chunk {c['id']}**")
                st.text(c["text"][:300] + "...")
                if i < 2:
                    st.divider()
            if len(chunks) > 3:
                st.caption(f"...and {len(chunks) - 3} more chunks")

# ── RIGHT: Ask and Answer ──────────────────────────────────────────────────────
with right:
    st.subheader("Step 2 — Ask a Question")

    if "chunks" not in st.session_state:
        st.info("Upload a PDF on the left first.")
        st.stop()

    example = st.selectbox("Try an example:", [
        "",
        "Who can obtain a temporary registration to file a refund?",
        "What is the minimum refund amount that can be claimed?",
        "What is considered the \"relevant date\" for filing a refund when a long-term construction contract is cancelled?",
        "When should an unregistered person file a refund claim instead of the supplier issuing a credit note?",
    ])
    question = st.text_input("Your question:", value=example)

    if st.button("🔍 Ask", disabled=not question.strip(), use_container_width=True):
        with st.spinner("Searching and generating..."):
            top_chunks = search(question, st.session_state.chunks)
            st.session_state.answer      = answer_question(question, top_chunks)
            st.session_state.used_chunks = top_chunks

    if "answer" in st.session_state:
        st.markdown("### Answer")
        st.markdown(
            f'<div style="background:#0F1F3D;color:#E2E8F0;padding:1rem 1.2rem;'
            f'border-radius:8px;border-left:4px solid #D97706;'
            f'font-size:0.95rem;line-height:1.7">'
            f'{st.session_state.answer}</div>',
            unsafe_allow_html=True,
        )

        st.markdown("### What GPT read to answer this")
        for c in st.session_state.used_chunks:
            with st.expander(f"Chunk {c['id']} — similarity: {c['score']:.3f}"):
                st.text(c["text"][:400] + ("..." if len(c["text"]) > 400 else ""))
