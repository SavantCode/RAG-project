import os
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from filereader import read_multiple_pdfs
from prompts import RAG_WORKFLOW_INSTRUCTIONS

# Load environment variables
load_dotenv()

# 1. Initialize free local embeddings & vector store
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vector_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/climate_change.pdf",
    "pdfs/global_warming.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
vector_store.add_documents(chunks)
print(f"Indexed {len(chunks)} chunks into vector store.")

# 2. Define the search tool
@tool
def search_documentation(query: str) -> str:
    """Search uploaded documents and return matching text chunks."""
    retrieved_docs = vector_store.similarity_search(query, k=min(4, len(chunks)))
    results = []
    for i, doc in enumerate(retrieved_docs, start=1):
        source = doc.metadata.get('source', 'unknown')
        results.append(f"--- Chunk {i} (Source: {source}) ---\n{doc.page_content}")
    return "\n\n".join(results)

# 3. Initialize Groq Model
model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
llm = ChatGroq(model=model_name, temperature=0.1)

# 4. Test run query
query = "what are the cause of climate change"
context = search_documentation.invoke(query)

prompt = f"""{RAG_WORKFLOW_INSTRUCTIONS}

Context documents:
{context}

User Question: {query}
"""

response = llm.invoke([HumanMessage(content=prompt)])
print("\n--- AI Response ---")
print(response.content)