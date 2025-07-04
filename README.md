# Gemini + LangChain PDF QA Utility

This project shows how to use Google Gemini and LangChain to answer questions about a PDF file.

## What it does

- Loads a PDF from a web link.
- Splits the PDF into chunks and creates a searchable index.
- Lets you ask questions about the PDF using Gemini.
- Shows how to generate general content and keep chat history.

## Requirements

- Python 3.8 or newer
- google-generativeai
- langchain
- langchain-google-genai
- chromadb
- python-dotenv
- ipython

Install everything with:
```sh
pip install google-generativeai langchain langchain-google-genai chromadb python-dotenv ipython
```

or

```sh
pip install -r requirements.txt
```

## Setup

1. Get a Gemini API key from Google AI Studio.
2. Make a `.env` file in this folder with:
    ```
    GEMINI_API_KEY=your_api_key_here
    ```

## Usage

Run the script:
```sh
python gemini-utilize.py
```

## Customization

You can change the PDF link, chunk size, or prompts in the script.

- `temperature` : Influences the randomness in token selection.
- `top_k` : Measure of how many of the most probable tokens are considered at each step.
- `max_output_tokens` : Upper limit of tokens generated in a response.
- `top_p` : Controls how the AI model chooses words when generating text.
---
