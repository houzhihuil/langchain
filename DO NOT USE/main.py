from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS 

# --- 1️⃣ Load and embed local documents ---
print("Loading documents...")
#loader = DirectoryLoader("./docs", glob="**/*.txt")  # adjust extensions as needed
loader = DirectoryLoader("./docs", glob="**/*.pdf")  # adjust extensions as needed
docs = loader.load()

print(f"Loaded {len(docs)} documents.")

# Create embeddings using Ollama
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# Create or load FAISS vector store
print("Creating vector store...")
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local("faiss_index")


# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- 2️⃣ Define your LLM ---
llm = OllamaLLM(model="minimax-m2:cloud")

# --- 3️⃣ Define your prompt template ---
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question:

{context}

Question: {question}
""")

# --- 4️⃣ Test the vector store and LLM ---

# Ask a sample question related to your documents
question = "What is this document about?"  # <-- change as needed

# Retrieve top matches from the vector store
retrieved_docs = retriever.invoke(question) 

# Combine the retrieved content
context = "\n\n".join([doc.page_content for doc in retrieved_docs])
#print("\n--- Retrieved Context ---")
#print(context[:50] + "...\n")  # show first 500 chars

# Format the full prompt
final_prompt = prompt.format(context=context, question=question)

# Get LLM response
response = llm.invoke(final_prompt)



print("--- LLM Response ---")
print(response)


 
 