# Mars Exploration Chatbot using Llama 3.1 and Ollama

## Overview

This project holds the power of Meta's Llama 3.1 8B model, integrated with the Ollama framework, to create an AI-driven chatbot capable of answering queries related to NASA's Mars exploration missions. 
The chatbot is implemented using advanced retrieval techniques, making it a robust tool for engaging with complex questions about Mars.

## Key Features

- **Powered by Llama 3.1**: Utilizes the 8B parameter version of Llama 3.1, one of the latest open-source large language models.
- **Ollama Integration**: Seamlessly integrates with the Ollama framework, enhancing the deployment and interaction capabilities of the Llama 3.1 model.
- **Contextual Understanding**: Capable of handling extensive context, making it ideal for detailed discussions about Mars exploration.
- **Multilingual Support**: The model supports eight languages, expanding its usability for a global audience.
- **Interactive Gradio Interface**: The project includes a user-friendly interface built with Gradio, allowing seamless interaction with the chatbot.

## Comparison with GPT Series

- **Model Size**: Llama 3.1 surpasses GPT-3.5 in scale with its 405B parameter model, providing deeper computational capabilities.
- **Context Length**: Handles up to 128K tokens, significantly more than GPT-3.5 and GPT-4, making it suitable for tasks requiring long-term contextual understanding.
- **Open-Source Advantage**: Unlike proprietary GPT models, Llama 3.1 is open-source, allowing for extensive customization and deployment across various platforms.



## Implementation


### 1. Load Data
The project loads data from a NASA webpage about Mars exploration using the `WebBaseLoader` class.

### 2. Set Up Models
The model setup includes generating embeddings and configuring the `ChatOllama` language model with the Llama 3.1 8B version for processing natural language queries.

### 3. Initialize RetrievalQA
The `RetrievalQA` chain is configured using the Llama 3.1 8B model and a Chroma vector store containing the data. A specific prompt from the hub is used for this purpose.

### 4. Query Processing
When a user asks a question, the `RetrievalQA` chain processes it to return the relevant information. If no suitable answer is found, the chatbot responds with "I don't know."

### 5. Gradio Interface
The project includes a Gradio-based user interface that allows users to interact with the chatbot, ask questions, and receive answers. The interface is styled with a description and an image related to Mars exploration.

https://github.com/riya-chougule/Llama-3.1-exploration/commit/8a5a9545b40c1feb0fb1da65cf1cfb54769fefc5#diff-8a96d00282d5b3219f43bf47cb78fb6507376c491c86426cdeeedb06a258bf69

