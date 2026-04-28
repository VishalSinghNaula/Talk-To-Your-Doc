# 📄 Talk To Your Doc (Enterprise RAG Chatbot)

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content in real-time. This project leverages modern LangChain architecture, Google's Gemini models, and DataStax Astra DB for fast and accurate vector retrieval.

## 🚀 Features
* **PDF Ingestion:** Upload any PDF document directly through the UI.
* **Smart Chunking:** Processes documents using `RecursiveCharacterTextSplitter` to maintain context.
* **Vector Search:** Uses Astra DB to perform similarity searches, ensuring the AI only answers based on the provided document.
* **Modern LLM Integration:** Powered by Google's Gemini Flash models for quick, grounded, and accurate responses.

## 🛠️ Tech Stack
* **Frontend:** Streamlit
* **Orchestration:** LangChain
* **Embeddings:** Google Generative AI (`gemini-embedding-001`)
* **LLM:** Google Generative AI (`gemini-2.0-flash`)
* **Vector Database:** DataStax Astra DB
* **Document Processing:** PyPDF2

## ⚙️ Installation & Setup

**1. Clone the repository**
```bash
git clone [https://github.com/VishalSinghNaula/Talk-To-Your-Doc.git](https://github.com/VishalSinghNaula/Talk-To-Your-Doc.git)
cd Talk-To-Your-Doc
