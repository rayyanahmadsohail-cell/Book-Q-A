"""
rag_setup.py - Load documents and build the vector store

This script does 4 things:
1. Reads all PDF and TXT files from the 'docs/' folder
2. Splits them into small chunks (500 characters each)
3. Converts each chunk into an embedding (a list of numbers)
4. Stores everything in ChromaDB (our vector database)

Run this ONCE before asking questions:
    python rag_setup.py
"""

import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma


load_dotenv()


def build_vector_store(docs_folder="docs"):
    """
    Load all documents from docs/ folder and store them in ChromaDB.

    This is the RAG "indexing" phase - we only need to run this once.
    After this, the vector store is saved to disk and ready for searching.
    """

    
    all_documents = []

    
    for filename in os.listdir(docs_folder):
        file_path = os.path.join(docs_folder, filename)

        if filename.endswith(".pdf"):
            print(f"  Loading PDF: {filename}")
            loader = PyPDFLoader(file_path)
            all_documents.extend(loader.load())

        elif filename.endswith(".txt"):
            print(f"  Loading TXT: {filename}")
            loader = TextLoader(file_path, encoding="utf-8")
            all_documents.extend(loader.load())

    
    if not all_documents:
        print("No documents found!")
        print("Please add PDF or TXT files to the 'docs/' folder first.")
        return None

    print(f"  Loaded {len(all_documents)} pages total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      
        chunk_overlap=50     
    )
    chunks = splitter.split_documents(all_documents)
    print(f"  Split into {len(chunks)} chunks")

    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"   # Save to disk so we can reuse it
    )

    print(f"  Vector store ready! Indexed {len(chunks)} chunks.")
    print("  You can now run: python main.py")
    return vector_store


if __name__ == "__main__":
   
    os.makedirs("docs", exist_ok=True)

    print("=" * 50)
    print("  Smart Book Q&A Crew - Document Indexer")
    print("=" * 50)
    print()
    build_vector_store()
