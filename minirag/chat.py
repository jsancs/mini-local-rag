from typing import Generator, Optional
import ollama


SYS_PROMPT = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
CONVERSATION_HISTORY = [
    {"role": "assistant", "content": SYS_PROMPT},
]


def chat_streaming(query: str, model: str, contenx: Optional[str] = None) -> Generator[str, None, None]:
    stream = ollama.chat(
        model=model,
        messages=[
            *CONVERSATION_HISTORY,
            {"role": "user", "content": query if contenx is None else f"{contenx} {query}"},
        ],
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]


def add_msg_to_memory(user_query: str, model_response: str) -> None:
    CONVERSATION_HISTORY.extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": model_response},
    ])


def clear_conversation() -> None:
    print("Clearing conversation history...")
    global CONVERSATION_HISTORY
    CONVERSATION_HISTORY = [
        {"role": "assistant", "content": SYS_PROMPT},
    ]
