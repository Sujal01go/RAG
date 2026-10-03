import os
from langchain_community.document_loaders import TextLoader,DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

def load_documents(docs_path="docs"):

    #checking if directory exists 
    if not os.path.exists(docs_path):
        raise FileNotFoundError("This directory does not exist,please create one")
    
    loader=DirectoryLoader(
        path=docs_path,
        glob="*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"}
    )

    documents=loader.load()

    if len(documents)==0:
        raise FileNotFoundError("No text File Found")
    
    for i,doc in enumerate(documents[:2]):
        print(f"\nDocument {i+1}:")
        print(f" Source: {doc.metadata['source']}")
        print(f" Content length : {len(doc.page_content)} characters")
        print(f" Content preview: {doc.page_content[:100]}....")
        print(f" metadata: {doc.metadata}")

    return documents

def split_documents(documents, chunk_size=800, chunk_overlap=0):
    #Split documents into smaller chunks with overlap

    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_documents(documents)

    if chunks:
        for i, chunk in enumerate(chunks[:5]):
            print(f"\n--- Chunk {i+1} ---")
            print(f"Source: {chunk.metadata['source']}")
            print(f"Length: {len(chunk.page_content)} characters")
            print("Content:")
            print(chunk.page_content)
            print("-" * 50)

        if len(chunks) > 5:
            print(f"\n... and {len(chunks) - 5} more chunks")

    return chunks

def create_vector_store(chunks,persist_directory="db/chroma_db"):
    #create and persist ChromaDB vector store

    embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

    vectorstore=Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space":"cosine"}
    )

    return vectorstore


def main():
    
    documents=load_documents(docs_path="docs")
    chunks=split_documents(documents)
    vectorstore=create_vector_store(chunks)



if __name__=="__main__":
    main()
