import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DB_DIR = "./college_db"
TEXT_FILE = "../data/college.txt"

# 1. Delete old DB to avoid duplicate chunks
if os.path.exists(DB_DIR):
    shutil.rmtree(DB_DIR)

# 2. Load document
loader = TextLoader(
    TEXT_FILE,
    encoding="utf-8"
)

docs = loader.load()

# 3. Split document into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

chunks = splitter.split_documents(docs)

print(f"Total chunks created: {len(chunks)}")

# 4. Create embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 5. Create Chroma Vector DB
db = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory=DB_DIR,
    collection_name="college_collection"
)

# 6. User Query
query = "What is attendance fee?"

# 7. Search similar chunks
results = db.similarity_search(
    query,
    k=3
)

# 8. Print retrieved knowledge
print("\nQuestion:")
print(query)

print("\nRelevant Information:")

for i, result in enumerate(results, start=1):
    print("----------------")
    print(f"Result {i}")
    print(result.page_content)