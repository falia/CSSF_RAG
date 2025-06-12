from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from pymilvus import connections


def get_vectorstore():
    # Explicitly connect to the remote Milvus server
    connections.connect(alias="default", host="3.252.104.166", port="19530")

    # Log the actual connection being used
    addr = connections.get_connection_addr("default")
    print(f"Connected to Milvus at {addr['host']}:{addr['port']}")

    # Initialize the embedding model
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    # Initialize the Milvus vector store
    vectorstore = Milvus(
        collection_name="cssf_documents",
        embedding_function=embedding_model,
        connection_args={"host": "3.252.104.166", "port": "19530"},
        auto_id=True
    )

    return vectorstore


def search(query: str, k: int = 5):
    vectorstore = get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)

    for i, doc in enumerate(results):
        print(f"\n--- Result {i+1} ---")
        print(f"Source URL: {doc.metadata.get('source_url', 'N/A')}")
        print(doc.page_content[:500])  # show only the first 500 characters


if __name__ == "__main__":
    user_query = input("Enter your question: ")
    search(user_query)
