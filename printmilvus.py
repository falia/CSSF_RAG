from langchain_community.vectorstores import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import torch

def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Milvus(
        collection_name="cssf_documents",
        embedding_function=embedding_model,
        connection_args={"host": "18.201.3.155", "port": "19530"},
        auto_id=True
    )

    return vectorstore

def search(query: str, k: int = 5):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source URL: {doc.metadata.get('source_url')}")
        print(doc.page_content[:500])  # print first 500 chars

if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)
    