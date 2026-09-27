import os
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM

from information_retrieval import (
    deduplicate_results,
    load_vector_store,
    search_vector_store,
    search_web,
)

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"   
LOCAL_RELEVANCE_THRESHOLD = 0.55
AMBIGUOUS_RELEVANCE_THRESHOLD = 0.70
CURRENT_WEB_TERMS = (
    "latest",
    "current",
    "today",
    "now",
    "news",
    "recent",
    "recently",
    "this year",
    "up to date",
    "up-to-date",
    "price",
    "weather",
)
COMPARISON_TERMS = (
    "compare",
    "comparison",
    "versus",
    " vs ",
    "difference between",
)


def is_current_web_question(question: str) -> bool:
    question_lower = f" {question.lower()} "
    return any(term in question_lower for term in CURRENT_WEB_TERMS)


def is_comparison_question(question: str) -> bool:
    question_lower = f" {question.lower()} "
    return any(term in question_lower for term in COMPARISON_TERMS)


def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file "
        )

    return LLM(model=f"groq/{GROQ_MODEL}", api_key=api_key, temperature=0)


def build_router_agent(llm):
    return Agent(
        role="Query Router",
        goal=(
            "Decide whether a user's question should be answered using the "
            "uploaded PDF documents, a live web search, or both."
        ),
        backstory=(
            "You are an expert research assistant who decides whether a "
            "question can be answered from the user's uploaded documents "
            "or requires information from the wider internet."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
    )


def build_routing_task(agent, question: str):
    return Task(
        description=(
            f'Question: "{question}"\n\n'
            "The local document store contains one or more PDF documents "
            "uploaded by the user. Use the local documents when the question "
            "can be answered from their content.\n\n"
            "Decide where this question should be answered from:\n"
            "- 'local' if it can be answered using the uploaded PDF documents\n"
            "- 'web' if it needs current information or information outside "
            "the uploaded documents\n"
            "- 'both' if it compares the uploaded documents with outside or "
            "current information\n\n"
            "Respond with EXACTLY one word: local, web, or both. "
            "Do not explain your answer."
        ),
        expected_output="One word: local, web, or both.",
        agent=agent,
    )


def classify_query(question: str, vector_store=None, llm=None) -> str:
    if is_current_web_question(question):
        return "both" if is_comparison_question(question) else "web"

    if vector_store is not None:
        local_matches = vector_store.similarity_search_with_score(question, k=1)
        if local_matches:
            relevance_score = local_matches[0][1]
            if relevance_score <= LOCAL_RELEVANCE_THRESHOLD:
                return "both" if is_comparison_question(question) else "local"
            if relevance_score <= AMBIGUOUS_RELEVANCE_THRESHOLD:
                return "both"

    llm = llm or get_llm()
    agent = build_router_agent(llm)
    task = build_routing_task(agent, question)

    crew = Crew(agents=[agent], tasks=[task], process=Process.sequential, verbose=False)
    reply = str(crew.kickoff()).strip().lower()

    mentions_local = "local" in reply
    mentions_web = "web" in reply

    if mentions_local and not mentions_web:
        return "local"
    if mentions_web and not mentions_local:
        return "web"
    return "both"


def route_and_retrieve(question: str, vector_store):
    decision = classify_query(question, vector_store=vector_store)
    print(f"Router decision for '{question}': {decision}")

    results = []
    if decision in ("local", "both"):
        results += search_vector_store(vector_store, question)
    if decision in ("web", "both"):
        results += search_web(question)

    return decision, deduplicate_results(results)


if __name__ == "__main__":
    store = load_vector_store()

    test_questions = [
        "What is multi-head attention in the Transformer model?",
        "What is the latest GPT model released by OpenAI?",
    ]

    for question in test_questions:
        decision, results = route_and_retrieve(question, store)
        print(f"\n--- Results for: '{question}' (routed to: {decision}) ---")
        for i, item in enumerate(results, start=1):
            preview = item["content"][:200].replace("\n", " ")
            print(f"{i}. [{item['source']}] {preview}...")
        print()
