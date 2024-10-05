# mini-local-rag

A tiny implementation of a RAG system that runs entirely on your computer!

### Usage

To send a chat completion request: <br>
`python chat.py -q <your_query>`

For now, there are 2 configurable params: <br>
* `-q --query`: user query to send to the chat (required)
* `-m --model`: model to use (llama3.2:1b by default). You can check the full list of available models [here](https://ollama.com/library)