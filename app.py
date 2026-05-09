from __future__ import annotations

import os
import re
import shutil
import time
import uuid
import threading
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
STATIC_DIR = BASE_DIR / "static"
SUPPORTED_EXTENSIONS = {".pdf", ".txt"}


class AskRequest(BaseModel):
    question: str = Field(min_length=3)
    session_id: str | None = None


class SourceDocument(BaseModel):
    source: str
    page: int | None = None
    snippet: str


class AskResponse(BaseModel):
    session_id: str
    answer: str
    sources: list[SourceDocument]


class UploadResponse(BaseModel):
    ingested_files: list[str]
    chunks_added: int
    total_indexed_documents: int


class HealthResponse(BaseModel):
    status: str
    service_ready: bool
    vectorstore_loaded: bool
    indexed_documents: int
    error: str | None = None


class RAGService:
    def __init__(self) -> None:
        self.api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set GOOGLE_API_KEY or GEMINI_API_KEY in your environment or .env file."
            )

        DATA_DIR.mkdir(exist_ok=True)
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001"),
            google_api_key=self.api_key,
        )
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("GEMINI_CHAT_MODEL", "gemini-2.5-flash"),
            temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.3")),
            google_api_key=self.api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=int(os.getenv("RAG_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("RAG_CHUNK_OVERLAP", "50")),
        )
        self.vectorstore: FAISS | None = None
        self.sessions: dict[str, list[BaseMessage]] = {}
        self.lock = threading.Lock()

        if (VECTORSTORE_DIR / "index.faiss").exists():
            try:
                self.vectorstore = FAISS.load_local(
                    str(VECTORSTORE_DIR),
                    self.embeddings,
                    allow_dangerous_deserialization=True,
                )
            except Exception as exc:
                print(f"Warning: failed to load existing FAISS index from {VECTORSTORE_DIR}: {exc}")
                self.vectorstore = None

    def ingest_files(self, files: list[UploadFile]) -> UploadResponse:
        if not files:
            raise HTTPException(status_code=400, detail="Upload at least one PDF or TXT file.")

        ingested_filenames: list[str] = []
        all_documents = []

        for file in files:
            suffix = Path(file.filename or "").suffix.lower()
            if suffix not in SUPPORTED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type for {file.filename}. Use PDF or TXT.",
                )

            target_path = UPLOAD_DIR / f"{uuid.uuid4().hex}_{Path(file.filename).name}"
            with target_path.open("wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            ingested_filenames.append(Path(file.filename or target_path.name).name)

            try:
                documents = self._load_documents(target_path)
            except Exception as e:
                target_path.unlink(missing_ok=True)
                # This could happen with corrupted or password-protected PDFs
                raise HTTPException(
                    status_code=400,
                    detail=f"Failed to load content from {file.filename}: {e}",
                )

            for document in documents:
                document.metadata["source"] = Path(document.metadata.get("source", target_path.name)).name
                document.metadata["uploaded_name"] = Path(file.filename or target_path.name).name
            all_documents.extend(documents)

        if not all_documents:
            raise HTTPException(status_code=400, detail="No readable content found in the uploaded files.")

        chunks = self.splitter.split_documents(all_documents)
        if not chunks:
            raise HTTPException(status_code=400, detail="No chunks were created from the uploaded files.")

        try:
            batch_size = 50
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i : i + batch_size]
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        with self.lock:
                            if self.vectorstore is None:
                                self.vectorstore = FAISS.from_documents(batch, self.embeddings)
                            else:
                                self.vectorstore.add_documents(batch)
                        break  # Batch processed successfully
                    except Exception as e:
                        error_msg = str(e).lower()
                        if ("429" in error_msg or "quota" in error_msg or "exhausted" in error_msg) and attempt < max_retries - 1:
                            match = re.search(r"retry in (\d+(?:\.\d+)?)s", error_msg)
                            wait_time = float(match.group(1)) + 1.0 if match else 60.0
                            print(f"Rate limit hit. Waiting {wait_time}s before retrying (Attempt {attempt + 1}/{max_retries})...")
                            time.sleep(wait_time)
                        else:
                            raise
        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                raise HTTPException(status_code=429, detail="Google API rate limit or quota exceeded. Please wait a minute and try again.") from exc
            raise HTTPException(status_code=502, detail=f"Embedding/indexing failed: {exc}") from exc

        with self.lock:
            self.vectorstore.save_local(str(VECTORSTORE_DIR))
            total_documents = self.vectorstore.index.ntotal

        return UploadResponse(
            ingested_files=ingested_filenames,
            chunks_added=len(chunks),
            total_indexed_documents=total_documents,
        )

    def ask(self, question: str, session_id: str | None = None) -> AskResponse:
        if self.vectorstore is None:
            raise HTTPException(
                status_code=400,
                detail="No vector index found. Upload documents before asking questions.",
            )

        session_id = session_id or uuid.uuid4().hex
        history = self.sessions.setdefault(session_id, [])
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 10},
        )
        try:
            documents = retriever.invoke(question)
            context = self._format_context(documents)
            prompt_messages = self._build_prompt(question=question, context=context, history=history)
            response = self.llm.invoke(prompt_messages)
        except Exception as exc:
            error_msg = str(exc).lower()
            if "429" in error_msg or "quota" in error_msg or "exhausted" in error_msg:
                raise HTTPException(status_code=429, detail="Google API rate limit or quota exceeded. Please wait a minute and try again.") from exc
            raise HTTPException(status_code=502, detail=f"Gemini generation failed: {exc}") from exc
        answer = self._extract_answer_text(response)

        history.extend([HumanMessage(content=question), AIMessage(content=answer)])
        if len(history) > 10:
            self.sessions[session_id] = history[-10:]

        return AskResponse(
            session_id=session_id,
            answer=answer,
            sources=self._format_sources(documents),
        )

    def _build_prompt(
        self,
        *,
        question: str,
        context: str,
        history: list[BaseMessage],
    ) -> list[BaseMessage]:
        system_text = (
            "You are a document-grounded assistant. Use only the retrieved context to answer. "
            "If the answer is not in the context, say that clearly. Keep the answer concise and factual."
        )
        trimmed_history = history[-6:]
        user_text = (
            f"Retrieved context:\n{context}\n\n"
            f"Question: {question}"
        )
        return [SystemMessage(content=system_text), *trimmed_history, HumanMessage(content=user_text)]

    def _load_documents(self, path: Path) -> list[Any]:
        if path.suffix.lower() == ".pdf":
            return PyPDFLoader(str(path)).load()
        if path.suffix.lower() == ".txt":
            return TextLoader(str(path), encoding="utf-8").load()
        return []

    @staticmethod
    def _format_context(documents: list[Any]) -> str:
        if not documents:
            return "No relevant context retrieved."

        parts = []
        for index, document in enumerate(documents, start=1):
            source = document.metadata.get("uploaded_name") or document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            page_label = f", page {page + 1}" if isinstance(page, int) else ""
            parts.append(f"[Chunk {index} | {source}{page_label}]\n{document.page_content}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_history(history: list[BaseMessage]) -> str:
        if not history:
            return "No previous conversation."

        lines = []
        for message in history:
            role = "User" if isinstance(message, HumanMessage) else "Assistant"
            lines.append(f"{role}: {message.content}")
        return "\n".join(lines)

    @staticmethod
    def _format_sources(documents: list[Any]) -> list[SourceDocument]:
        sources: list[SourceDocument] = []
        seen: set[tuple[str, int | None, str]] = set()

        for document in documents:
            source = document.metadata.get("uploaded_name") or document.metadata.get("source", "unknown")
            page = document.metadata.get("page")
            snippet = " ".join(document.page_content.split())[:220]
            key = (source, page, snippet)
            if key in seen:
                continue
            seen.add(key)
            sources.append(SourceDocument(source=source, page=page + 1 if isinstance(page, int) else None, snippet=snippet))
        return sources

    @staticmethod
    def _extract_answer_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict) and "text" in item:
                    text_parts.append(str(item["text"]))
                else:
                    text_parts.append(str(item))
            return "\n".join(text_parts).strip()
        return str(content)


app = FastAPI(title="Gemini RAG Chatbot", version="1.0.0")
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
service: RAGService | None = None
service_init_error: str | None = None


def get_service() -> RAGService:
    global service
    global service_init_error

    if service is not None:
        return service

    try:
        service = RAGService()
        service_init_error = None
        return service
    except Exception as exc:
        service_init_error = str(exc)
        raise HTTPException(status_code=503, detail=f"RAG service unavailable: {exc}") from exc


@app.on_event("startup")
async def startup_event() -> None:
    try:
        get_service()
    except HTTPException:
        pass


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    current_service = service
    if current_service is None and service_init_error is None:
        try:
            current_service = get_service()
        except HTTPException:
            current_service = None

    indexed_documents = 0
    vectorstore_loaded = False
    if current_service and current_service.vectorstore is not None:
        vectorstore_loaded = True
        indexed_documents = current_service.vectorstore.index.ntotal

    if service_init_error:
        return HealthResponse(
            status="degraded",
            service_ready=False,
            vectorstore_loaded=vectorstore_loaded,
            indexed_documents=indexed_documents,
            error=service_init_error,
        )

    return HealthResponse(
        status="ok",
        service_ready=current_service is not None,
        vectorstore_loaded=vectorstore_loaded,
        indexed_documents=indexed_documents,
    )


@app.post("/documents/upload", response_model=UploadResponse)
async def upload_documents(files: list[UploadFile] = File(...)) -> UploadResponse:
    current_service = get_service()
    return await run_in_threadpool(current_service.ingest_files, files)


@app.post("/ask", response_model=AskResponse)
async def ask_question(payload: AskRequest) -> AskResponse:
    current_service = get_service()
    return await run_in_threadpool(current_service.ask, payload.question, payload.session_id)
