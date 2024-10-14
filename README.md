# mini-local-rag

A tiny implementation of a RAG system that runs entirely on your computer!

> [!IMPORTANT]  
> This repo is in development. It can have important changes between commits

### Start

1. Start Ollama engine.
2. Run `python cli.py`. It will run a Llama3.2:1b model.
3. Chat with the model

There is 1 configurable param: <br>
* `-m --model`: model to use (llama3.2:1b by default). You can check the full list of available models [here](https://ollama.com/library)


### Usage
* Type a message to chat with the model. All the conversation will be remembered by the model.
* Type `/bye` to stop the chat.
* Type `/help` to show all the commands.
* Type `/add` to create a collection.
    * You'll be asked to enter the paths for all the documents for the collection.
    * Once you have entered all the docs, type `/stop` to interrupt the process.
    * You will be asked to introduce a coleccion name.
    * Then the embeddings will be generated and stored in a .npy file for future reference. The embeddings will be stored in memory with numpy.

> [!Note]  
> For the moment, only .txt files are supported. More extensions will be available soon :)