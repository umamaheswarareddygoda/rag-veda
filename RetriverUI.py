import os
import re
import openai
import tempfile
import pytesseract
import pinecone
import streamlit as st
import nltk
from pdf2image import convert_from_path
from pinecone import Pinecone
from pinecone_text.sparse import BM25Encoder
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import PineconeHybridSearchRetriever

nltk.download("punkt")

def extract_text_using_ocr(pdf_path):
    images = convert_from_path(pdf_path)
    return "\n".join(clean_text(pytesseract.image_to_string(img)) for img in images)

# --- API Keys and Configuration ---
LLM_API_KEY = "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"
PINECONE_API_KEY = "pcsk_2fR64n_HgEDAC4i3JjwKfJciWvxhoxLj2Vs2cJ4SCskfdg4mLh4ZUW1bBoKNY9P98qRzZp"
PINECONE_INDEX_NAME = "rag-veda"
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN

# --- Initialize Clients ---
LLM = openai.OpenAI(base_url="https://api.groq.com/openai/v1", api_key=LLM_API_KEY)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- Load or Create BM25 Encoder ---
bm25_path = "rag-veda.json"
bm25_encoder = BM25Encoder().load(bm25_path) if os.path.exists(bm25_path) else BM25Encoder()

# --- Initialize Retriever ---
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)

# --- Helper Functions ---
def clean_text(text):
    return re.sub(r"\s+", " ", text).strip()

def extract_text_using_ocr(pdf_path):
    images = convert_from_path(pdf_path, poppler_path=r"C:\poppler-24.08.0\Library\bin")
    return "\n".join(clean_text(pytesseract.image_to_string(img)) for img in images)

def split_text_into_chunks(text, chunk_size=300):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def prepare_rag_prompt(query, top_n=3):
    top_docs = retriever.invoke(query)
    top_contents = [doc.page_content for doc in top_docs[:top_n]]
    return f"""
You are an expert assistant. Use the following retrieved context to answer the user's question concisely and accurately.

### Question:
{query}

### Retrieved Context:
1. {top_contents[0] if len(top_contents) > 0 else ''}
2. {top_contents[1] if len(top_contents) > 1 else ''}
3. {top_contents[2] if len(top_contents) > 2 else ''}

### Instructions:
- Provide a detailed and accurate answer based on the retrieved context.
- Do not include unrelated information.
- If the context is unclear, rely on general knowledge and don't mention it anywhere.
"""

def get_llm_response(query):
    history = ""
    for msg in st.session_state["messages"][-5:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"

    rag_prompt = prepare_rag_prompt(query)
    full_prompt = f"""
You are a helpful AI assistant. Below is a conversation between a user and you.
Use the past conversation and the retrieved context to answer the current query.

### Chat History:
{history}

{rag_prompt}
"""

    response = LLM.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": full_prompt}],
        max_tokens=3000,
        temperature=0.8,
    )
    return "\n".join(choice.message.content for choice in response.choices)

# --- Streamlit UI Setup ---
st.set_page_config(page_title="ChatBot", layout="wide")

if "messages" not in st.session_state:
    st.session_state["messages"] = []

st.title("🌿 RAG Veda ChatBot")

# --- Display Chat History with Styled Dark Bubbles ---
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(
            f"""
            <div style="text-align: right;">
                <div style="display: inline-block; background-color: #1E3A8A; color: white;
                            padding: 10px 15px; border-radius: 15px; margin: 5px 0;
                            max-width: 70%; word-wrap: break-word;">
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style="text-align: left;">
                <div style="display: inline-block; background-color: #374151; color: white;
                            padding: 10px 15px; border-radius: 15px; margin: 5px 0;
                            max-width: 70%; word-wrap: break-word;">
                    {message["content"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )



# --- Chat Input ---
with st.form(key="user_input_form", clear_on_submit=True):
    query = st.text_input("Type your message here:", key="query_input", placeholder="Ask me about Ayurveda...")
    submitted = st.form_submit_button("Send")

if submitted and query:
    st.session_state["messages"].append({"role": "user", "content": query})
    with st.spinner("Thinking..."):
        response = get_llm_response(query)
    st.session_state["messages"].append({"role": "bot", "content": response})
    st.rerun()

# --- Sidebar PDF Upload ---
st.sidebar.header("📄 Upload PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_pdf:
    with st.spinner("Extracting text from PDF..."):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_pdf.read())
            pdf_path = tmp_file.name

        extracted_text = extract_text_using_ocr(pdf_path)
        chunks = split_text_into_chunks(extracted_text)

        bm25_encoder.fit(chunks)
        bm25_encoder.dump(bm25_path)

        retriever = PineconeHybridSearchRetriever(
            embeddings=embeddings,
            sparse_encoder=bm25_encoder,
            index=index
        )
        retriever.add_texts(chunks)

        st.success("PDF processeds and content added to memory.")
