import argparse
import ollama


SYS_PROMPT = "You are a knowledgeable, efficient, and direct AI assistant. Provide concise answers, focusing on the key information needed. Offer suggestions tactfully when appropriate to improve outcomes. Engage in productive collaboration with the user."


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


def chat_streaming(query: str, model: str) -> None:
    stream = ollama.chat(
        model=model,
        messages=[
            {"role": "assistant", "content": SYS_PROMPT},
            {"role": "user", "content": query},
        ],
        stream=True,
    )

    for chunk in stream:
        print(chunk["message"]["content"], end="", flush=True)


def handle_model(model_name):
    model_exists = check_model_exists(model_name)
    if not model_exists:
        pull_model(model_name)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat with an AI assistant.")

    parser.add_argument("-q", "--query", type=str, help="Query to chat about.", required=True)
    parser.add_argument("-m", "--model", type=str, default="llama3.2:1b", help="Model to use.")
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    model_name = args.model
    user_query = args.query
    
    handle_model(model_name)
    
    if user_query:
        chat_streaming(user_query, model_name)
    else:
        raise ValueError("Please provide a query to chat about.")
    

if __name__ == "__main__":
    main()