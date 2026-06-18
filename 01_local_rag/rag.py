#Load Document
from langchain_community.document_loaders import TextLoader

loader = TextLoader("../data/college.txt")

docs = loader.load()

#Split Document

from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=60,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)
for i, chunk in enumerate(chunks):
    print("\nCHUNK", i)
    print(chunk.page_content)
#Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Vector Store
from langchain_community.vectorstores import Chroma

db = Chroma.from_documents(
    chunks,
    embeddings,
persist_directory="./college_db"
)