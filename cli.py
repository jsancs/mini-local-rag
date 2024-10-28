import argparse
from typing import List, Optional
from prompt_toolkit.shortcuts import prompt

from chat import chat_streaming, add_msg_to_memory, clear_conversation
from embeddings import create_collection, load_collection, list_collections, similarity_search
from utils import handle_model


COLLECTION_ACTIVE = None


def show_help():
    print("Commands:")
    print("/clear - Clear the chat history and starts a new conversation")
    print("/add - Add document to the model's memory")
    print("/activate <collection_name> - Activate a collection")
    print("/deactivate - Deactivate the active collection")
    print("/list - List available collections")
    print("/help - Show this help message")
    print("/bye - Exit the chat")
    print("ctrl + c - Stop the model from responding")
    print("ctrl + d - Exit the chat")
    print()


def get_user_input() -> str:
    user_query = prompt(
        ">>> ",
        placeholder='Send a message (/? for help)',
    )
    return user_query


def generate_response(
        user_query: str,
        model_name: str,
        context: Optional[str] = None,
    ) -> None:
    model_response = "" 

    for chunk in chat_streaming(user_query, model_name, context):
        model_response += chunk
        print(chunk, end="", flush=True)
    print()

    add_msg_to_memory(user_query, model_response)


def get_documents() -> List[str]:
    doc_paths = []
    while True:
        doc_path = prompt(
            "Local path to document document (file or dir) (/done to finish): ",
        )
        if doc_path == "/done":
            break

        if doc_path:
            doc_paths.append(doc_path)

    return doc_paths


def handle_user_query(user_query: str, model_name: str) -> None:
    global COLLECTION_ACTIVE

    if user_query == "/bye":
        print("Goodbye!")
        exit()
    elif user_query == "/clear":
        clear_conversation()
    elif user_query == "/help" or user_query == "/?":
        show_help()
    elif user_query.startswith("/activate"):
        collection_name = user_query.split(" ")[1]
        COLLECTION_ACTIVE = load_collection(collection_name)
        if COLLECTION_ACTIVE:
            print(f"Collection {collection_name} activated.")
        else:
            print(f"Collection {collection_name} not found.")
    elif user_query == "/list":
        list_collections()
    elif user_query == "/deactivate":
        COLLECTION_ACTIVE = None
        print("Collection deactivated.")
    elif user_query == "/add":
        collection_name = ""
        documents = get_documents()
        if documents:
            collection_name = prompt(
                "Enter a name   for the collection: ",
            )

        print(f"Adding documents to collection: {collection_name}")
        print(f"Docs selected: {documents}")

        create_collection(documents, collection_name)
    else:
        context = similarity_search(user_query, COLLECTION_ACTIVE) if COLLECTION_ACTIVE else None
        generate_response(user_query, model_name, context)


def chat_cli(model_name: str) -> None:
    while True:
        try:
            user_query = get_user_input()
            handle_user_query(user_query, model_name)           

        except KeyboardInterrupt:
            # Ctrl-C to stop the model from responding
            print("\nUse Ctrl + d or /bye to exit.")
        except EOFError:
            # Ctrl-D to exit
            break


def parse_arguments():
    parser = argparse.ArgumentParser(description="Chat with an AI assistant.")
    parser.add_argument("-m", "--model", type=str, default="llama3.2:1b", help="Model to use.")
    return parser.parse_args()


def main():
    args = parse_arguments()
    model_name = args.model
    
    handle_model(model_name)
    chat_cli(model_name)
    

if __name__ == "__main__":
    main()
