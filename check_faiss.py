import faiss

# Use the correct filename and safe path formatting
index = faiss.read_index(r"faiss_index/index.faiss")   # <-- fixed
print("FAISS index loaded successfully!")
# check how many vectors are in the index
print(f"vectors: {index.ntotal}" )