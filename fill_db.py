from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb
import os

DATA_PATH = r"data"  
CHROMA_PATH = r"chroma_db"  
MAX_BATCH_SIZE = 5000

chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name="Choreo-RAG")

raw_documents = []
for root, dirs, files in os.walk(DATA_PATH):
    for file_name in files:
        if file_name.endswith(".md"):
            file_path = os.path.join(root, file_name)
            loader = TextLoader(file_path, encoding='utf-8')
            try:
                raw_documents.extend(loader.load())
            except Exception as e:
                print(f"Error loading file: {file_path}. Error: {e}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

documents = [chunk.page_content for chunk in chunks]
metadata = [chunk.metadata for chunk in chunks]
ids = [f"ID{i}" for i in range(len(chunks))]

def batch_data(data, batch_size):
    """Yield successive batches from a list."""
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]


try:
    for doc_batch, meta_batch, id_batch in zip(
        batch_data(documents, MAX_BATCH_SIZE),
        batch_data(metadata, MAX_BATCH_SIZE),
        batch_data(ids, MAX_BATCH_SIZE),
    ):
        collection.upsert(
            documents=doc_batch,
            metadatas=meta_batch,
            ids=id_batch
        )
    print("Data successfully added to ChromaDB in batches!")
except Exception as e:
    print(f"Error during ChromaDB upsert: {e}")
