from typing import Generator
import ollama


SYS_PROMPT = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."
CONVERSATION_HISTORY = [
    {"role": "assistant", "content": SYS_PROMPT},
]


def check_model_exists(model_name: str) -> bool:
    all_models = ollama.list()
    ollama_model_names = [model["name"] for model in all_models["models"]]

    if model_name in ollama_model_names:
        return True
    return False


def pull_model(model_name: str) -> None:
    print(f"Model {model_name} not found. Pulling model...")
    try:
        ollama.pull(model_name)
        print("Model pulled successfully.")
    except ollama.ResponseError as e:
        print(f"Error: {e}")
        raise ValueError("Model not found. Please provide a valid model name.")


def chat_streaming(query: str, model: str) -> Generator[str, None, None]:
    stream = ollama.chat(
        model=model,
        messages=[
            *CONVERSATION_HISTORY,
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    for chunk in stream:
        yield chunk["message"]["content"]


def handle_model(model_name: str) -> None:
    model_exists = check_model_exists(model_name)
    if not model_exists:
        pull_model(model_name)


def add_msg_to_memory(user_query: str, model_response: str) -> None:
    CONVERSATION_HISTORY.extend([
        {"role": "user", "content": user_query},
        {"role": "assistant", "content": model_response},
    ])
