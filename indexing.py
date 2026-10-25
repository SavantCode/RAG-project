from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings  # <--- Changed
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from filereader import read_multiple_pdfs

load_dotenv()

# Use a free, high-speed local embedding model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # <--- Changed

vectore_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/climate_change.pdf",
    "pdfs/global_warming.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

all_splits = text_splitter.split_documents(docs)

print(f"Split documentation into {len(all_splits)} chunks")

vectore_store.add_documents(documents=all_splits)
print(f"Indexed {len(all_splits)} chunks")