import google.generativeai as genai
from IPython.display import display
from IPython.display import Markdown
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import os

load_dotenv()
print(os.environ["GEMINI_API_KEY"])
api_key = os.environ["GEMINI_API_KEY"]
genai.configure(api_key = api_key)
model = genai.GenerativeModel('gemini-1.5-flash-latest')

## General content generation
response = model.generate_content("Explain me about the United States of America.")
print(response.text)
Markdown(response.text)

## Keep chat history
hist = model.start_chat()
response = hist.send_message("Hi! Tell me how to make America great again.")
Markdown(response.text)

for i in hist.history:
    print(i)
    print('\n\n')
i.parts[0].text 

model.count_tokens("Use your API keys securely. Do not share them or embed them in code the public can view.")

## RAG implementation
CHUNK_SIZE = 700
CHUNK_OVERLAP = 100
pdf_path = "http://jmc.stanford.edu/articles/whatisai/whatisai.pdf"

pdf_loader = PyPDFLoader(pdf_path)
split_pdf_document = pdf_loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
context = "\n\n".join(str(p.page_content) for p in split_pdf_document)
texts = text_splitter.split_text(context)

gemini_model = ChatGoogleGenerativeAI(model='gemini-1.5-flash-latest', google_api_key=api_key , temperature=0.8)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
vector_index = Chroma.from_texts(texts, embeddings)
retriever = vector_index.as_retriever(search_kwargs={"k" : 5})

# Create a RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(gemini_model, retriever=retriever, return_source_documents=True)
question = "What is AI?"
result = qa_chain.invoke({"query": question})
print("Answer:", result["result"])