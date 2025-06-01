from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.docstore.document import Document

import json
import os

# === CONFIG ===
model_name = "BAAI/bge-large-en-v1.5"
pdf_file_paths = [
"C:\\Users\\faton\\Downloads\\Macroprudential_measures_for_GBP_Liability_Driven_Investment_Funds.pdf", 
"C:\\Users\\faton\\Downloads\\Answer-to-the-public-consultation-Abrdn-Investments-Luxembourg-S.A.pdf"]  
output_jsonl = "bge_embeddings_from_pdfs.jsonl"

# === Load embedding model ===
embedding_model = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs={"device": "cpu"},  # Change to "cpu" if no GPU
    encode_kwargs={"normalize_embeddings": True}
)

# === Load PDF documents ===
documents = []
for path in pdf_file_paths:
    loader = PyPDFLoader(path)
    pdf_docs = loader.load()  # Returns a list of Document objects (one per page)
    for doc in pdf_docs:
        doc.metadata["source"] = os.path.basename(path)
    documents.extend(pdf_docs)

# === Split documents into semantically coherent chunks ===
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = text_splitter.split_documents(documents)
print(f"✅ Loaded {len(documents)} pages → {len(chunks)} chunks.")

# === Prepare and embed chunks ===
with open(output_jsonl, "w", encoding="utf-8") as out_file:
    texts = ["passage: " + doc.page_content for doc in chunks]
    embeddings = embedding_model.embed_documents(texts)

    for i, (doc, embedding) in enumerate(zip(chunks, embeddings)):
        record = {
            "embedding": embedding,
            "text": doc.page_content,
            "metadata": {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
                "chunk_id": i
            }
        }
        out_file.write(json.dumps(record) + "\n")

print(f"✅ Saved {len(chunks)} embeddings to: {output_jsonl}")
