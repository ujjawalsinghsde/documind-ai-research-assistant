import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"


def get_groq_client():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GROQ_API_KEY is missing. Add it to your .env file "
        )
    return Groq(api_key=api_key)


def build_validation_prompt(question: str, context_items: list, answer: str) -> str:
    if not context_items:
        context_text = "No context was provided."
    else:
        labeled_chunks = [
            f"[{i}] {item['content']}" for i, item in enumerate(context_items, start=1)
        ]
        context_text = "\n\n".join(labeled_chunks)

    return (
        "You are fact-checking an AI-generated answer against the context "
        "it was supposed to use. You are NOT answering the question - you "
        "are only judging whether the answer is supported by the context.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\n\n"
        f"Answer to check:\n{answer}\n\n"
        "On the first line, respond with EXACTLY one of these verdicts: "
        "GROUNDED, PARTIALLY GROUNDED, or NOT GROUNDED.\n"
        "GROUNDED means every claim in the answer is backed by the context.\n"
        "PARTIALLY GROUNDED means some claims are backed, but others are not "
        "in the context.\n"
        "NOT GROUNDED means the answer is mostly made up or unrelated to "
        "the context.\n"
        "On the second line, give a one-sentence reason for your verdict."
    )


def validate_answer(question: str, context_items: list, answer: str, client=None) -> dict:
    client = client or get_groq_client()
    prompt = build_validation_prompt(question, context_items, answer)

    completion = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a strict, impartial fact-checker. You only "
                    "judge whether an answer is supported by given context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0,
    )

    reply = completion.choices[0].message.content.strip()
    lines = [line.strip() for line in reply.splitlines() if line.strip()]

    verdict_line = lines[0].upper() if lines else ""
    reason = lines[1] if len(lines) > 1 else "No reason given."

    if "NOT GROUNDED" in verdict_line:
        verdict = "Not Grounded"
    elif "PARTIALLY GROUNDED" in verdict_line:
        verdict = "Partially Grounded"
    elif "GROUNDED" in verdict_line:
        verdict = "Grounded"
    else:
        # If the model didn't follow the format, don't silently claim success.
        verdict = "Unknown"

    return {"verdict": verdict, "reason": reason}


if __name__ == "__main__":
    test_question = "What is multi-head attention?"
    test_context = [
        {
            "source": "local_pdf",
            "content": (
                "Multi-head attention allows the model to jointly attend to "
                "information from different representation subspaces at "
                "different positions."
            ),
        }
    ]

    good_answer = (
        "Multi-head attention lets the model attend to information from "
        "different representation subspaces at different positions [1]."
    )
    made_up_answer = (
        "Multi-head attention was invented in 2023 by OpenAI to replace "
        "convolutional networks entirely."
    )

    for label, answer in [("Good answer", good_answer), ("Made-up answer", made_up_answer)]:
        result = validate_answer(test_question, test_context, answer)
        print(f"\n{label} : {answer}")
        print(f"Verdict : {result['verdict']}")
        print(f"Reason : {result['reason']}")
