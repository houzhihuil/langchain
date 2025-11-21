""" # main.py - use minimax-m2:cloud to answer a simple question
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import DirectoryLoader
 
# --- 1️⃣ Define your cloud  
llm = OllamaLLM(model="minimax-m2:cloud")

# --- 2️⃣ Define a simple Q&A prompt ---
prompt = ChatPromptTemplate.from_template("""
#You are a helpful assistant. Answer the following question clearly:

#Question: {question}
""")

# --- 3️⃣ Ask a question ---
question = "Explain Accounting job market in Montreal."

# --- 4️⃣ Format prompt and get response ---
final_prompt = prompt.format(question=question)
response = llm.invoke(final_prompt)

print("--- LLM Response ---")
print(response)
 """
# main.py - use minimax-m2:cloud to answer a simple question
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# --- 1️⃣ Define your cloud  
llm = OllamaLLM(model="minimax-m2:cloud")

# --- 2️⃣ Define a simple Q&A prompt ---
prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the following question clearly:

Question: {question}
""")

# --- 3️⃣ Ask a question ---
question = "Can you tell me a joke?"

# --- 4️⃣ Format prompt and get response ---
final_prompt = prompt.format(question=question)
response = llm.invoke(final_prompt)

print("--- LLM Response ---")
print(response)
