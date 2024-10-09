import argparse
from prompt_toolkit.shortcuts import prompt

from chat import chat_streaming, handle_model, add_msg_to_memory


def chat_cli(model_name: str) -> None:
    while True:
        try:
            user_query = prompt(
                ">>> ",
                placeholder='Send a message (/? for help)',
            )

            model_response = "" 

            for chunk in chat_streaming(user_query, model_name):
                model_response += chunk
                print(chunk, end="", flush=True)
            print()

            add_msg_to_memory(user_query, model_response)

        except KeyboardInterrupt: 
            # Ctrl-C interrupt
            break
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