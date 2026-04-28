import streamlit as st
import os
from PyPDF2 import PdfReader
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_astradb import AstraDBVectorStore

load_dotenv()

# --- Configuration ---
# Ensure your Astra DB Endpoint is correct
ASTRA_ENDPOINT = "https://b67b6e26-0bf8-41fc-abc1-6432137008cf-us-east-2.apps.astra.datastax.com"
ASTRA_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")

st.set_page_config(page_title="Enterprise RAG", page_icon="🧠")
st.title("Enterprise RAG Knowledge Base 🧠")

# Initialize AI models
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Connect to Astra DB Vector Store
vstore = AstraDBVectorStore(
    embedding=embeddings,
    collection_name="pdf_knowledge_base",
    api_endpoint=ASTRA_ENDPOINT,
    token=ASTRA_TOKEN
)

# --- UI Layout ---
with st.sidebar:
    st.header("Document Management")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if st.button("Process Document") and pdf is not None:
        with st.spinner("Analyzing PDF..."):
            pdf_reader = PdfReader(pdf)
            raw_text = ""
            for page in pdf_reader.pages:
                raw_text += page.extract_text() or ""

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=800, 
                chunk_overlap=200
            )
            chunks = text_splitter.split_text(text=raw_text)

            # Add to database
            vstore.add_texts(chunks)
            st.success("Document Ingested!")

# --- Chat Interface ---
query = st.text_input("Ask a question about your document:")

if query:
    with st.spinner("Searching knowledge base..."):
        # 1. Retrieve context
        docs = vstore.similarity_search(query, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        # 2. Generate response with a structured prompt
        prompt = (
            f"Use the following context to answer the question.\n\n"
            f"Context: {context}\n\n"
            f"Question: {query}\n\n"
            f"Answer:"
        )
        
        response = llm.invoke(prompt)
        
        st.subheader("Answer:")
        st.write(response.content)