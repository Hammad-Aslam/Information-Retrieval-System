import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.llms import GooglePalm
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# Custom configuration for pydantic to avoid protected namespace conflicts
class CustomGooglePalmEmbeddings(GooglePalmEmbeddings):
    model_config = {"protected_namespaces": ()}


# Function to read PDF and extract text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Function to split text into smaller chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    chunks = text_splitter.split_text(text)
    return chunks


# Function to generate vector embeddings for text chunks
def get_vector_store(text_chunks):
    embeddings = (
        CustomGooglePalmEmbeddings()
    )  # Using custom class to handle namespace issues
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vector_store


# Function to set up the conversational chain
def get_conversational_chain(vector_store):
    llm = GooglePalm()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(), memory=memory
    )
    return conversation_chain


# Main function to run the process
def main():
    # List of PDF documents
    pdf_docs = ["example.pdf"]  # Replace with actual PDF file path

    # Step 1: Extract text from PDF
    text = get_pdf_text(pdf_docs)

    # Step 2: Split the text into chunks
    text_chunks = get_text_chunks(text)

    # Step 3: Generate vector embeddings for the chunks
    vector_store = get_vector_store(text_chunks)

    # Step 4: Set up the conversational retrieval chain
    conversation_chain = get_conversational_chain(vector_store)

    # Step 5: Ask a question to the conversational chain
    question = "What is the content of the document?"  # Example query
    response = conversation_chain({"question": question})

    # Print the response
    print(response)


if __name__ == "__main__":
    main()
