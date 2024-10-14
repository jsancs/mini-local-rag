import argparse
from typing import List
from prompt_toolkit.shortcuts import prompt

from chat import chat_streaming, add_msg_to_memory
from embeddings import create_collection
from utils import handle_model


def show_help():
    print("Commands:")
    print("/bye - Exit the chat")
    print("/add - Add document to the model's memory")
    print("/help - Show this help message")
    print("ctrl + c - Stop the model from responding")
    print("ctrl + d - Exit the chat")
    print()


def generate_response(user_query: str, model_name: str) -> None:
    model_response = "" 

    for chunk in chat_streaming(user_query, model_name):
        model_response += chunk
        print(chunk, end="", flush=True)
    print()

    add_msg_to_memory(user_query, model_response)

def add_document() -> List[str]:
    doc_paths = []
    while True:
        doc_path = prompt(
            "Local path to document document (/done to finish): ",
        )
        if doc_path == "/done":
            break

        if doc_path:
            doc_paths.append(doc_path)
        
    return doc_paths

def handle_user_query(user_query: str, model_name: str) -> None:
    if user_query == "/bye":
        print("Goodbye!")
        exit()
    elif user_query == "/help" or user_query == "/?":
        show_help()
    elif user_query == "/add":
        collection_name = ""
        documents = add_document()
        if documents:
            collection_name = prompt(
                "Enter a name for the collection: ",
            )

        print(f"Adding documents to collection: {collection_name}")
        print(f"Docs selected: {documents}")

        create_collection(documents, collection_name)


    else:
        generate_response(user_query, model_name)


def chat_cli(model_name: str) -> None:
    while True:
        try:
            user_query = prompt(
                ">>> ",
                placeholder='Send a message (/? for help)',
            )

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