"""Initialize and populate the Pinecone vector store with local data"""
import os
import time
import base64
from keys import PINECONE_KEY, OPENAI_KEY
from constants import PINECONE_INDEX_NAME

from pinecone import Pinecone, ServerlessSpec

from langchain_core.messages.human import HumanMessage
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter


DATA_FOLDER = os.path.abspath("data")
DATA_PROCESSING_FOLDER = os.path.abspath("data_processing")

# Endpoint to Pinecone API
pc = Pinecone(api_key=PINECONE_KEY)

# Set the PINECONE_KEY in the environment.
# This is needed for PineconeVectorStore operations that upsert chunked docs into the vector store.
os.environ["PINECONE_API_KEY"] = PINECONE_KEY

# We need gpt-4o to get text descriptions for images.
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
)


def _txt_to_documents(filepath: str) -> list[Document]:
    print(f"\tLoading .txt document {filepath}...")
    loader = TextLoader(filepath)
    return loader.load()


def _png_to_document(filepath: str) -> list[Document]:
    print(f"\tLoading .png document {filepath}...")

    name, extension = filepath.split("/")[-1].split(".")
    with open(filepath, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    message = HumanMessage(
        content=[
            {"type": "text", "text": " Please give me the most detailed description of this image."},
            {"type": "image_url", "image_url": {"url": f"data:image/{extension};base64,{image_data}"}},
        ],
    )
    response = llm.invoke([message])
    print(response.content)
    return [Document(page_content=f"{name}: {response.content}")]


# Load all documents and split them into 1000-sized chunks.
documents = []
print(f"Scanning {DATA_FOLDER} for docs...")
for filename in os.listdir(DATA_FOLDER):
    filepath = os.path.join(DATA_FOLDER, filename)
    if filepath.endswith(".txt"):
        documents += _txt_to_documents(filepath)
    elif filepath.endswith(".png"):
        documents += _png_to_document(filepath)
    else:
        print(f"\tWARNING: Document of unsupported type: {filepath}")

# Split all the obtained docs into chunks.
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunked_documents = text_splitter.split_documents(documents)
print(f"Done.\n")

# Create "empty" openAI embeddings for the obtained chunks.
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)

# Create a new pinecone index
print(f"Connecting to Pinecone...")

# For the purpose of this playground we want to reinitialize the index every time.
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if PINECONE_INDEX_NAME in existing_indexes:
    print(f"\tIndex {PINECONE_INDEX_NAME} already existing. Deleting it...")
    pc.delete_index(name=PINECONE_INDEX_NAME)

# Recreate and repopulate the index.
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
