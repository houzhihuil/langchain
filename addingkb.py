# add_to_kb.py

from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

# --- Load and embed documents ---
print("Loading documents...")
loader = DirectoryLoader("./docs", glob="**/*.pdf")  # adjust file types if needed
docs = loader.load()
print(f"Loaded {len(docs)} documents.")

# Create embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Build FAISS vector store
print("Creating vector store...")
vectorstore = FAISS.from_documents(docs, embeddings)

# Save the index
vectorstore.save_local("faiss_index")
print("Vector store saved to ./faiss_index")
print("Knowledge base created and saved.")