import os
import shutil
from pypdf import PdfReader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
VECTOR_STORE_DIR = "vector_store"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


def load_all_pdfs(data_dir: str):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data folder not found at '{data_dir}'.")

    pdf_files = sorted(
        filename
        for filename in os.listdir(data_dir)
        if filename.lower().endswith(".pdf")
    )

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found inside '{data_dir}'.")

    all_pages = []

    for filename in pdf_files:
        pdf_path = os.path.join(data_dir, filename)
        reader = PdfReader(pdf_path)

        pages = [
            Document(
                page_content=page.extract_text() or "",
                metadata={"source": filename, "page": page_number},
            )
            for page_number, page in enumerate(reader.pages, start=1)
        ]
        all_pages.extend(pages)
        print(f"Loaded {len(pages)} pages from: {pdf_path}")

    print(f"Loaded {len(pdf_files)} PDF file(s) in total")
    return all_pages


def split_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    chunks = splitter.split_documents(pages)
    unique_chunks = []
    seen_chunks = set()

    for chunk in chunks:
        normalized_content = " ".join(chunk.page_content.split()).casefold()
        key = (
            chunk.metadata.get("source", "unknown"),
            chunk.metadata.get("page", "unknown"),
            normalized_content,
        )
        if key not in seen_chunks:
            seen_chunks.add(key)
            unique_chunks.append(chunk)

    chunks = unique_chunks
    print(f"Split into {len(chunks)} chunks (chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

    if os.path.exists(VECTOR_STORE_DIR):
        shutil.rmtree(VECTOR_STORE_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=VECTOR_STORE_DIR,
    )
    print(f"Vector store created and saved to: {VECTOR_STORE_DIR}")
    return vector_store


def run_test_query(vector_store, question: str, top_k: int = 3):
    # results = vector_store.similarity_search(question, k=top_k, search_type="mmr")
    # MMR (max marginal relevance) picks diverse chunks instead of near-duplicate top matches
    results = vector_store.max_marginal_relevance_search(question, k=top_k, lambda_mult=0.5)

    print(f"\nTop {top_k} chunks for question: '{question}'\n")
    for i, doc in enumerate(results, start=1):
        page = doc.metadata.get("page", "unknown")
        preview = doc.page_content[:500].replace("\n", " ")
        print(f"{i}. (page {page}) {preview}...")


if __name__ == "__main__":
    pages = load_all_pdfs(DATA_DIR)
    
    chunks = split_into_chunks(pages)

    vector_store = build_vector_store(chunks)

    run_test_query(vector_store, "What is the Transformer architecture?")