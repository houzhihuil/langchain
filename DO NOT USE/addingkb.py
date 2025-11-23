# modified code: add_to_kb.py
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
import os

INDEX_PATH = "faiss_index"

embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Load new docs only
loader = DirectoryLoader("./docs_new", glob="**/*.pdf")
new_docs = loader.load()
print(f"Loaded {len(new_docs)} new documents.")

# Case 1: index exists → load and append
if os.path.exists(INDEX_PATH):
    print("Loading existing FAISS index...")
    vectorstore = FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)

    print("Adding new documents...")
    vectorstore.add_documents(new_docs)

# Case 2: index does NOT exist → create new one
else:
    print("Creating new FAISS index...")
    vectorstore = FAISS.from_documents(new_docs, embeddings)

# Save back
vectorstore.save_local(INDEX_PATH)
print("Updated FAISS index saved.")
 