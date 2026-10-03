from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage,AIMessage

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

model = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)

chat_history=[]

def ask_question(user_question):
    print(f"\n--- You asked: {user_question}---")

    if chat_history:
        message= [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable.Just return the rewritten question")
        ]+chat_history+[
            HumanMessage(content=f"New question: {user_question}")
        ]
        result=model.invoke(message)
        search_question=result.content.strip()
        print(f"Searching for {search_question}")
    else:
        search_question=user_question

    
    retriever=db.as_retriever(search_kwargs={"k":3})
    docs=retriever.invoke(search_question)

    print(f"found {len(docs)} relevent documents : ")
    for i,doc in enumerate(docs,1):
        lines=doc.page_content.split('\n')[:2]
        preview='\n'.join(lines)
        print(f" Doc{i}: {preview}...")

    combined_input = f"""
        Based on the following documents, please answer this question:

        Question:
        {search_question}

        Documents:
        {"\n\n".join([f"- {doc.page_content}" for doc in docs])}

        Please provide a clear and helpful answer using only the information
        from these documents.

        If you cannot find the answer in the documents, say:
        "I don't have enough information in the provided documents."
        """
    messages = [
    SystemMessage(
        content="You are a helpful assistant. Answer only using the provided documents."
    )]+chat_history+[
    HumanMessage(
        content=combined_input
    )
    ]
    result=model.invoke(messages)
    answer=result.content

    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    print(f"\n\nanswer: {answer}")
    return answer


def start_chat():
    print("Ask me questions: Type 'quit' to exit.")

    while True:
        question=input("\nYour Question : ")

        if question.lower()=="quit":
            print("goodbye")
            break

        ask_question(question)


if __name__=="__main__":
    start_chat()
        

