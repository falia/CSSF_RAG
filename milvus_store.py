from langchain_milvus import Milvus
from langchain_huggingface import HuggingFaceEmbeddings
import torch
from pymilvus import connections

def get_vectorstore():
    connections.connect(alias="default", host="52.31.135.91", port="19530")

    addr = connections.get_connection_addr("default")
    print(f"Milvus connection info: {addr}")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    return Milvus(
        collection_name="cssf_documents",
        embedding_function=embedding_model,
        connection_args={"host": "52.31.135.91", "port": "19530"},
        auto_id=True
    )