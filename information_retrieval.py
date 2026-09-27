import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from groq import Groq

from document_processor import VECTOR_STORE_DIR, EMBEDDING_MODEL_NAME

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"


def load_vector_store():
    if not os.path.exists(VECTOR_STORE_DIR):
        raise FileNotFoundError(
            f"No vector store found at '{VECTOR_STORE_DIR}'. "
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = Chroma(
        persist_directory=VECTOR_STORE_DIR,
        embedding_function=embeddings,
    )
    return vector_store


def search_vector_store(vector_store, question: str, top_k: int = 3):
    results = vector_store.similarity_search_with_score(question, k=top_k * 3)

    formatted_results = [
        {
            "source": "local_pdf",
            "document": doc.metadata.get("source", "unknown"),
            "content": doc.page_content,
            "page": doc.metadata.get("page", "unknown"),
            "score": score,
        }
        for doc, score in results
    ]

    return deduplicate_results(formatted_results)[:top_k]


def deduplicate_results(results: list[dict]) -> list[dict]:
    unique_results = []
    seen = set()

    for result in results:
        source = result.get("source", "unknown")
        content = " ".join(result.get("content", "").split()).casefold()

        if source == "web" and result.get("url"):
            key = (source, result["url"].rstrip("/").casefold())
        else:
            key = (
                source,
                result.get("document", "unknown"),
                result.get("page", "unknown"),
                content,
            )

        if key not in seen:
            seen.add(key)
            unique_results.append(result)

    return unique_results


def search_web(question: str, max_results: int = 3):
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is missing. Add it to your .env file "
        )

    client = TavilyClient(api_key=api_key)
    response = client.search(query=question, max_results=max_results)

    formatted_results = [
        {
            "source": "web",
            "content": result.get("content", ""),
            "url": result.get("url"),
            "score": result.get("score"),
        }
        for result in response.get("results", [])
    ]
    return deduplicate_results(formatted_results)[:max_results]


def get_combined_context(vector_store, question: str):
    local_results = search_vector_store(vector_store, question)
    web_results = search_web(question)

    combined = deduplicate_results(local_results + web_results)
    print(
        f"Retrieved {len(local_results)} local chunk(s) and "
        f"{len(web_results)} web result(s) for: '{question}'"
    )
    return combined


if __name__ == "__main__":
    store = load_vector_store()

    test_question = "What is the Transformer architecture?"
    context_items = get_combined_context(store, test_question)

    print(f"\n--- Combined context for: '{test_question}' ---\n")
    for i, item in enumerate(context_items, start=1):
        preview = item["content"][:200].replace("\n", " ")
        print(f"{i}. [{item['source']}] {preview}...")
