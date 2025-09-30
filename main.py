import os
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain import hub

load_dotenv()

def main():
    print("Vectorstore-in-memory")
    pdf_path ="/Users/srini/Desktop/Work/Development/AILearning/LangChain/vectorstore-in-memory/2210.03629v3.pdf"

    loader = PyPDFLoader(file_path=pdf_path)
    documents = loader.load()
    llm = ChatOpenAI()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30, separator="\n")
    docs = text_splitter.split_documents(documents=documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    vectorstore.save_local("faiss_index_react")

    new_vectorstore = FAISS.load_local("faiss_index_react", embeddings, allow_dangerous_deserialization=True)

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    retrieval_chain = create_retrieval_chain(
        retriever=vectorstore.as_retriever(), combine_docs_chain=combine_docs_chain
    )

    result = retrieval_chain.invoke(input={"input": "Give me the gist ReAct in 3 sentences"})

    print(result["answer"])

if __name__ == "__main__":
    print("Vectorstore-in-memory-1")



if __name__ == "__main__":
    main()
