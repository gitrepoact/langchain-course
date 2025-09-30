import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv



load_dotenv()

def main():
    print("Hello from intro-to-vector-dbs!")
    loader = TextLoader("/Users/srini/Desktop/Work/Development/AILearning/LangChain/Intro-to-vector-dbs/mediumblog1.txt")
    document = loader.load()
    print("Splitter")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"Created {len(texts)} chunks")

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    print("Ingesting...")

    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ["INDEX_NAME"])

    print("Finish")

if __name__ == "__main__":
    main()