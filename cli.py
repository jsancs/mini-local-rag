import argparse
from prompt_toolkit.shortcuts import prompt

from chat import chat_streaming, handle_model, add_msg_to_memory


def show_help():
    print("Commands:")
    print("/bye - Exit the chat")
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

def handle_user_query(user_query: str, model_name: str) -> None:
    if user_query == "/bye":
        print("Goodbye!")
        exit()
    elif user_query == "/help" or user_query == "/?":
        show_help()
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