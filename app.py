
import os
import re
import numpy as np
import streamlit as st
import faiss

from sentence_transformers import SentenceTransformer
from groq import Groq


# =========================================================
# 1. ตั้งค่าหน้าเว็บ
# =========================================================

st.set_page_config(
    page_title="Prachinburi Travel RAG",
    page_icon="🗺️",
    layout="centered"
)

st.title("🗺️ Prachinburi Travel RAG")
st.subheader("ผู้ช่วยท่องเที่ยวจังหวัดปราจีนบุรี")

st.write(
    "ถามข้อมูลเกี่ยวกับสถานที่ท่องเที่ยว ประวัติศาสตร์ "
    "ธรรมชาติ การเดินทาง อาหาร และผลิตภัณฑ์ท้องถิ่นได้เลย"
)


# =========================================================
# 2. ตั้งค่า Path
# =========================================================

DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    "data"
)


# =========================================================
# 3. โหลดและทำความสะอาดเอกสาร
# =========================================================

def clean_text(text):

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = text.strip()

    return text


@st.cache_data
def load_documents():

    documents = []

    for filename in sorted(os.listdir(DATA_PATH)):

        if filename.endswith(".txt"):

            filepath = os.path.join(
                DATA_PATH,
                filename
            )

            with open(
                filepath,
                "r",
                encoding="utf-8"
            ) as f:

                text = f.read()

            text = clean_text(text)

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


# =========================================================
# 4. Chunking
# =========================================================

def create_chunks(
    text,
    chunk_size=700,
    overlap=100
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


@st.cache_data
def build_chunks():

    documents = load_documents()

    chunks = []

    for doc in documents:

        doc_chunks = create_chunks(
            doc["text"],
            chunk_size=700,
            overlap=100
        )

        for i, chunk in enumerate(doc_chunks):

            chunks.append({
                "filename": doc["filename"],
                "chunk_id": i,
                "text": chunk
            })

    return chunks


# =========================================================
# 5. โหลด Embedding Model
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "paraphrase-multilingual-MiniLM-L12-v2"
    )


# =========================================================
# 6. สร้าง FAISS Index
# =========================================================

@st.cache_resource
def build_faiss():

    model = load_embedding_model()
    chunks = build_chunks()

    chunk_texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        chunk_texts,
        normalize_embeddings=True
    )

    embedding_matrix = np.array(
        embeddings,
        dtype="float32"
    )

    dimension = embedding_matrix.shape[1]

    index = faiss.IndexFlatIP(dimension)

    index.add(embedding_matrix)

    return index


# =========================================================
# 7. Retrieval
# =========================================================

def retrieve_context(
    question,
    top_k=5
):

    model = load_embedding_model()
    index = build_faiss()
    chunks = build_chunks()

    question_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    question_embedding = np.array(
        question_embedding,
        dtype="float32"
    )

    scores, indices = index.search(
        question_embedding,
        top_k
    )

    results = []

    for score, idx in zip(
        scores[0],
        indices[0]
    ):

        if idx < 0:
            continue

        results.append({
            "filename": chunks[idx]["filename"],
            "text": chunks[idx]["text"],
            "score": float(score)
        })

    return results


# =========================================================
# 8. RAG + Groq
# =========================================================

def ask_rag(question):

    results = retrieve_context(question)

    context_parts = []

    for item in results:

        context_parts.append(
            f"แหล่งข้อมูล: {item['filename']}\n"
            f"{item['text']}"
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    prompt = f"""
คุณคือผู้ช่วยท่องเที่ยวจังหวัดปราจีนบุรี

กฎการตอบ:

1. ใช้ข้อมูลใน CONTEXT เท่านั้น
2. ห้ามแต่งข้อมูลหรือเดาข้อมูลที่ไม่มีในเอกสาร
3. หาก CONTEXT ไม่มีข้อมูลที่สามารถตอบคำถามได้
   ให้ตอบว่า "ไม่พบข้อมูลในคลังเอกสาร"
4. ตอบเป็นภาษาไทยให้เข้าใจง่าย
5. ระบุชื่อไฟล์เอกสารที่ใช้เป็นแหล่งอ้างอิง
6. หากข้อมูลไม่เพียงพอ
   ห้ามใช้ความรู้จากภายนอกมาเติม

CONTEXT:

{context}

คำถาม:

{question}

คำตอบ:
"""

    api_key = st.secrets["GROQ_API_KEY"]

    client = Groq(
        api_key=api_key
    )

    response = client.chat.completions.create(

        model="openai/gpt-oss-120b",

        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.1
    )

    answer = response.choices[0].message.content.strip()

    if "ไม่พบข้อมูลในคลังเอกสาร" in answer:

        return answer, []

    sources = list(
        dict.fromkeys(
            item["filename"]
            for item in results
        )
    )

    return answer, sources


# =========================================================
# 9. Chat History
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# แสดงประวัติการสนทนา

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            st.markdown(
                "**📚 แหล่งอ้างอิง:**"
            )

            for source in message["sources"]:

                st.markdown(
                    f"- `{source}`"
                )


# =========================================================
# 10. ช่องถามคำถาม
# =========================================================

question = st.chat_input(
    "พิมพ์คำถามเกี่ยวกับจังหวัดปราจีนบุรี..."
)


if question:

    # แสดงคำถามของผู้ใช้

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):

        st.markdown(question)


    # สร้างคำตอบ

    with st.chat_message("assistant"):

        with st.spinner(
            "กำลังค้นข้อมูล..."
        ):

            try:

                answer, sources = ask_rag(
                    question
                )

                st.markdown(answer)

                if sources:

                    st.markdown(
                        "**📚 แหล่งอ้างอิง:**"
                    )

                    for source in sources:

                        st.markdown(
                            f"- `{source}`"
                        )

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })

            except Exception as e:

                st.error(
                    f"เกิดข้อผิดพลาด: {e}"
                )
