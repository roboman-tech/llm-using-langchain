# Gemini + LangChain PDF RAG Utility

This project demonstrates how to use Google Gemini (via the `google-generativeai` API) and LangChain for both general content generation and Retrieval-Augmented Generation (RAG) from PDF documents.

## Features

- **General content generation** using Gemini.
- **Chat history** management with Gemini.
- **PDF ingestion** and chunking using LangChain.
- **Vector store creation** with Chroma and Google embeddings.
- **RetrievalQA**: Ask questions based on PDF content.
- **Customizable generation parameters** (temperature, top_p, etc).

## Requirements

- Python 3.8+
- [google-generativeai](https://pypi.org/project/google-generativeai/)
- [langchain](https://pypi.org/project/langchain/)
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- [chromadb](https://pypi.org/project/chromadb/)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [IPython](https://pypi.org/project/ipython/)

Install dependencies:
```sh
pip install google-generativeai langchain langchain-google-genai chromadb python-dotenv ipython
```

## Setup

1. **Get a Gemini API key** from Google AI Studio.
2. **Create a `.env` file** in the project directory:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

Run the script:
```sh
python gemini-utilize.py
```

### What it does

- Loads a PDF from a URL, splits it into chunks, and creates a vector index.
- Uses Gemini to answer questions about the PDF content via RetrievalQA.
- Demonstrates general content generation and chat history.
- Allows you to customize generation parameters.

### Example output

```
Explain me about the United States of America.
...
Answer: [retrieved answer from PDF]
...
Paris is the capital of France.
```

## Customization

You can change the PDF URL, chunk size, or prompt in the script as needed.

---
