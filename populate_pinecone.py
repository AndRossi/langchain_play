"""Initialize and populate the Pinecone vector store with local data"""
import os

from pinecone import Pinecone, ServerlessSpec
from keys import PINECONE_KEY, OPENAI_KEY
from constants import PINECONE_INDEX_NAME
import time
from langchain_pinecone import PineconeVectorStore

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

DATA_FOLDER = os.path.abspath("data")

# Endpoint to Pinecone API
pc = Pinecone(api_key=PINECONE_KEY)

# Set the PINECONE_KEY in the environment.
# This is needed for PineconeVectorStore operations that upsert chunked docs into the vector store.
os.environ["PINECONE_API_KEY"] = PINECONE_KEY

# Load all documents and split them into 1000-sized chunks.
documents = []
print(f"Scanning {DATA_FOLDER} for docs...")
for document_name in os.listdir(DATA_FOLDER):
    document_path = os.path.join(DATA_FOLDER, document_name)
    if document_path.endswith(".txt"):
        print(f"\tLoading document {document_path}")
        loader = TextLoader(document_path)
        documents += loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(documents)
print(f"Done.\n")

# Create "empty" openAI embeddings for the obtained chunks.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Create a new
print(f"Connecting to Pinecone...")
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if PINECONE_INDEX_NAME not in existing_indexes:
    print(f"\tNo index {PINECONE_INDEX_NAME} found. Creating it...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1"),
    )
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        time.sleep(1)
    print(f"\tDone.\n")

    index = pc.Index(PINECONE_INDEX_NAME)

    # Upload into the pinecone vector store
    print(f"\tUploading docs into Pinecone...")
    docsearch = PineconeVectorStore.from_documents(chunked_documents, embeddings, index_name=PINECONE_INDEX_NAME)
    print(f"\tDone.\n")
