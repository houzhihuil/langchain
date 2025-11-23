# adding UI for question answering using Ollama LLM and FAISS vector store

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
import streamlit as st

st.title("📚 Local RAG with Ollama")

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
# question = input("Ask a question: ")
question = st.text_input("Ask a question:")
if question:
    docs = retriever.invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    final_prompt = prompt.format(context=context, question=question)
    answer = llm.invoke(final_prompt)
    st.write("### Answer")
    st.write(answer)

 

 
