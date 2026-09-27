import os
from dotenv import load_dotenv
from groq import Groq

from information_retrieval import load_vector_store
from query_router import route_and_retrieve
from validation import validate_answer

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file "
        )
    return Groq(api_key=api_key)


def build_prompt(question: str, context_items: list) -> str:
    if not context_items:
        context_text = "No relevant context was found."
    else:
        labeled_chunks = []
        for i, item in enumerate(context_items, start=1):
            if item["source"] == "local_pdf":
                source_label = (
                    f"Document: {item.get('document', 'unknown')}, "
                    f"Page: {item.get('page', 'unknown')}"
                )
            else:
                source_label = "Web"
            labeled_chunks.append(f"[{i}] ({source_label}) {item['content']}")
        context_text = "\n\n".join(labeled_chunks)

    return (
        "Answer the question using ONLY the context below. "
        "If the context does not contain enough information, say so honestly "
        "instead of guessing.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        "Answer in simple language so a beginner can understand it. Give a "
        "short but complete answer: answer the question directly, explain "
        "important unfamiliar terms briefly, and include the key details "
        "needed to avoid a half answer. Do not add unrelated information. "
        "Mention which source number(s) (like [1], [2]) you used. If the "
        "context lists indexed PDF files, list those filenames clearly. Do "
        "not call indexed documents 'training data'; they are documents "
        "used for retrieval."
    )


def generate_response(question: str, context_items: list, client=None) -> str:
    client = client or get_groq_client()
    prompt = build_prompt(question, context_items)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a precise, honest research assistant. "
                    "Never make up information that isn't in the given context. "
                    "Explain answers in simple language and keep them short but "
                    "complete. Answer the exact question directly and do not "
                    "leave out the key details needed to understand it. "
                    "Your answer is displayed in a plain-text terminal, so do not use "
                    "LaTeX (no \\[ \\], \\( \\), or $...$) or Markdown formatting - "
                    "write any formulas or symbols in plain text instead."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return completion.choices[0].message.content.strip()


def answer_question(question: str, vector_store) -> dict:
    decision, context_items = route_and_retrieve(question, vector_store)
    answer = generate_response(question, context_items)
    validation = validate_answer(question, context_items, answer)

    return {
        "question": question,
        "routed_to": decision,
        "sources_used": len(context_items),
        "answer": answer,
        "validation": validation,
    }


if __name__ == "__main__":
    store = load_vector_store()

    test_question = "What is multi-head attention in the Transformer model?"
    result = answer_question(test_question, store)

    print(f"\nQuestion: {result['question']}")
    print(f"Routed to: {result['routed_to']}")
    print(f"Sources used: {result['sources_used']}")
    print(f"\nAnswer:\n{result['answer']}")
    print(f"\nValidation: {result['validation']['verdict']} - {result['validation']['reason']}")
