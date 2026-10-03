from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

persistent_directory="db/chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db=Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space":"cosine"}
)

query="What wass NVIDIA's first graphics accelerator called?"

retriever=db.as_retriever(search_kwargs={"k":5})

# retriever=db.as_retriever(
#     search_type="similarity_score_threshold",
#     search_kwargs={
#         "k":5, #returns 5 chunks with similarity
#         "score_threshold":0.3 #Only return chunks with cosine similarity = 0.3
#     }
# )

relevant_docs=retriever.invoke(query)

# print(f"user query : {query}")

# for i,doc in enumerate(relevant_docs,1):
#     print(f"Document {i}:\n{doc.page_content}\n")


combined_input = f"""
Based on the following documents, please answer this question:

Question:
{query}

Documents:
{"\n\n".join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear and helpful answer using only the information
from these documents.

If you cannot find the answer in the documents, say:
"I don't have enough information in the provided documents."
"""


model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer only using the provided documents."
    ),
    HumanMessage(
        content=combined_input
    )
]

result = model.invoke(messages)


# Display answer
print("\n--- Generated Response ---")
print(result.content)