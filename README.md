# 🧠 Gemini RAG Chatbot

A high-performance Retrieval-Augmented Generation (RAG) chatbot powered by **Google Gemini**, **FAISS**, and **FastAPI**. This application allows you to upload multiple PDF or TXT documents and intelligently chat with your personal knowledge base using a beautiful, human-centric web interface.

## ✨ Features

- **Document Processing**: Upload `.pdf` and `.txt` files easily through the drag-and-drop interface.
- **Fast Retrieval**: Uses `FAISS` (Facebook AI Similarity Search) for blazingly fast and efficient vector similarity search.
- **Powered by Gemini**: Leverages Google's state-of-the-art `gemini-2.5-flash` model for generating accurate, context-aware answers.
- **Source Citations**: Every answer provided by the AI is strictly grounded in your documents, and it tells you exactly which document (and page) the information was pulled from.
- **Modern UI**: A fully responsive, warm, and user-friendly interface that feels like a native chat application.
- **Session Tracking**: Maintains chat history per session for context-aware follow-up questions.

## 🛠️ Technology Stack

- **Backend**: FastAPI (Python)
- **AI & Orchestration**: LangChain, Google Gemini API
- **Vector Database**: FAISS
- **Frontend**: Vanilla HTML5, CSS3, JavaScript
- **Environment**: python-dotenv

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.10+ installed.

### 1. Clone the repository

```bash
git clone https://github.com/sahilpatel2310/gemini-rag-chatbot.git
cd gemini-rag-chatbot
```

### 2. Set up the virtual environment

```bash
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the root directory and add your Google Gemini API key:

```env
GEMINI_API_KEY=your_google_api_key_here
GEMINI_CHAT_MODEL=gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
GEMINI_TEMPERATURE=0.3
RAG_CHUNK_SIZE=500
RAG_CHUNK_OVERLAP=50
```

### 5. Run the Application

Start the FastAPI server using Uvicorn:

```bash
uvicorn app:app --reload
```

The application will be available at [http://127.0.0.1:8000](http://127.0.0.1:8000).

## 💡 How to Use

1. Open the web interface.
2. Drag and drop your target PDFs or Text files into the "Knowledge Base" panel on the left.
3. Click **Upload & Learn**. The app will extract the text, split it into chunks, embed it, and store it in the local FAISS vector database.
4. Type a question in the chat box on the right. The AI will retrieve the most relevant chunks from your documents and synthesize an answer!
