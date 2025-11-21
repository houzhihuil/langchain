# ask_kb.py

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

# --- Load vector store ---
print("Loading vector store...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Define LLM ---
llm = OllamaLLM(model="minimax-m2:cloud")

# --- Prompt Template ---
prompt = ChatPromptTemplate.from_template("""
Use the following context to answer the question:

{context}

Question: {question}
""")

# --- Ask a question ---
question = input("Ask a question: ")

print("\n--- Retrieving relevant documents... ---")
retrieved_docs = retriever.invoke(question)
context = "\n\n".join([doc.page_content for doc in retrieved_docs])

final_prompt = prompt.format(context=context, question=question)

response = llm.invoke(final_prompt)

print("\n--- Answer ---")
print(response)
