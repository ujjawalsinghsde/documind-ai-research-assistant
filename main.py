import os

from document_processor import (
    DATA_DIR,
    VECTOR_STORE_DIR,
    load_all_pdfs,
    split_into_chunks,
    build_vector_store,
)
from information_retrieval import load_vector_store
from response_generator import answer_question

DEMO_QUESTIONS = [
    "What is multi-head attention in the Transformer model?",
    "Why did the authors move away from recurrence and convolutions?",
    "What is the latest GPT model released by OpenAI?",
]


def print_header(text: str):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def get_or_build_vector_store():
    if os.path.exists(VECTOR_STORE_DIR):
        print_header("STEP 1: Document Processing & Storage")
        print(f"Found existing vector store at '{VECTOR_STORE_DIR}' - reusing it.")
        return load_vector_store()

    print_header("STEP 1: Document Processing & Storage")
    print("No vector store found yet - building it from all PDFs now...")
    pages = load_all_pdfs(DATA_DIR)
    chunks = split_into_chunks(pages)
    return build_vector_store(chunks)


def ask_and_show(question: str, vector_store):
    print_header(f"QUESTION: {question}")

    result = answer_question(question, vector_store)

    print(f"\n[Routing]      -> {result['routed_to']}")
    print(f"[Retrieval]    -> {result['sources_used']} source(s) used")
    print(f"[Answer]\n{result['answer']}")
    print(f"[Validation]   -> {result['validation']['verdict']} ({result['validation']['reason']})")


def run_demo(vector_store):
    print_header("STEPS 2-4: Retrieval + Routing + Response Generation (Demo)")
    for question in DEMO_QUESTIONS:
        ask_and_show(question, vector_store)


def run_interactive(vector_store):
    print_header("YOUR TURN: Ask your own questions (type 'exit' to quit)")
    while True:
        question = input("\nYour question: ").strip()
        if question.lower() in ("exit", "quit", ""):
            print("Goodbye!")
            break
        ask_and_show(question, vector_store)


if __name__ == "__main__":
    print_header("Agentic Research Assistant - Full Pipeline")

    vector_store = get_or_build_vector_store()
    run_demo(vector_store)
    run_interactive(vector_store)
