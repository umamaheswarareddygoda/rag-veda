import os
import openai
import pinecone
from pinecone import Pinecone
from langchain_community.retrievers import PineconeHybridSearchRetriever
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone_text.sparse import BM25Encoder
import streamlit as st

# API Keys and Config
LLM_API_KEY = "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"
PINECONE_API_KEY = "pcsk_2fR64n_HgEDAC4i3JjwKfJciWvxhoxLj2Vs2cJ4SCskfdg4mLh4ZUW1bBoKNY9P98qRzZp"
PINECONE_INDEX_NAME = "rag-veda"

# Initialize OpenAI and Pinecone
LLM = openai.OpenAI(
    base_url=f"https://api.groq.com/openai/v1",
    api_key=LLM_API_KEY
)
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

# Load BM25 Encoder and Embeddings
import os
HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().load("rag-veda.json")

# Create Retriever
retriever = PineconeHybridSearchRetriever(
    embeddings=embeddings, sparse_encoder=bm25_encoder, index=index
)

# Function to Prepare RAG Prompt
def prepare_rag_prompt(query, top_n=3):
    top_k_documents = retriever.invoke(query)
    top_k_contents = [doc.page_content for doc in top_k_documents[:top_n]]

    template = f"""
You are an expert assistant. Use the following retrieved context to answer the user's question concisely and accurately.

### Question:
{query}

### Retrieved Context:
1. {top_k_contents[0] if len(top_k_contents) > 0 else ''}
2. {top_k_contents[1] if len(top_k_contents) > 1 else ''}
3. {top_k_contents[2] if len(top_k_contents) > 2 else ''}

### Instructions:
- Provide a detailed and accurate answer based on the retrieved context.
- Do not include unrelated information.
- If the context is unclear, rely on general knowledge and don't mention it anywhere.
"""
    return template

# Function to Get LLM Response
def get_llm_response(query):
    rag_prompt = prepare_rag_prompt(query, 3)
    response = LLM.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": rag_prompt}],
        max_tokens=3000,
        temperature=0.8
    )
    all_responses = "\n".join(choice.message.content for choice in response.choices)
    return all_responses

# Streamlit UI Setup
st.set_page_config(page_title="ChatBot", layout="wide")

# Chat history in Session State
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display Title
st.title("🌿 RAG Veda ChatBot")

# Chat Interface: Maintain consistent UI with fixed alternating styles
for message in st.session_state["messages"]:
    if message["role"] == "user":
        st.markdown(f"""
        <div style="padding:10px;border-radius:10px;margin:10px 0;">
        <b>You:</b> {message["content"]}
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="padding:10px;border-radius:10px;margin:10px 0;">
        <b>Bot:</b> {message["content"]}
        </div>
        """, unsafe_allow_html=True)

# Input box
with st.form(key="user_input_form", clear_on_submit=True):
    query = st.text_input("Type your message here:", key="query_input", placeholder="Ask me about Ayurveda...")
    submitted = st.form_submit_button("Send")

# Handle Input and Generate Response
if submitted and query:
    # Add user query to chat history
    st.session_state["messages"].append({"role": "user", "content": query})

    # Get Response
    with st.spinner("Thinking..."):
        response = get_llm_response(query)

    # Add bot response to chat history
    st.session_state["messages"].append({"role": "bot", "content": response})

    # Rerun to refresh UI with new messages
    st.rerun()
