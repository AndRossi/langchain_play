"""Initialize and populate the Pinecone vector store with local data"""
import os
from typing import Any
import time
import base64
from keys import PINECONE_KEY, OPENAI_KEY
from constants import PINECONE_INDEX_NAME, MAX_CHUNK_SIZE

from pinecone import Pinecone, ServerlessSpec

from langchain_core.messages.human import HumanMessage
from langchain_core.documents.base import Document
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

from unstructured.partition.pdf import partition_pdf
from pydantic import BaseModel


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


def _get_image_caption(filepath: str) -> str:
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
    return response.content


def _txt_to_documents(filepath: str) -> list[Document]:
    print(f"\tLoading .txt document {filepath}...")
    loader = TextLoader(filepath)
    return loader.load()


def _image_to_documents(filepath: str) -> list[Document]:
    name, extension = filepath.split("/")[-1].split(".")
    print(f"\tLoading .{extension} document {filepath}...")

    caption = _get_image_caption(filepath)
    return [Document(page_content=f"{name}: {caption}")]


def _pdf_to_documents(filepath: str) -> list[Document]:
    print(f"\tLoading .pdf document {filepath}...")

    # Use unstructured to partition the PDF in elements
    raw_pdf_elements = partition_pdf(
        filename=os.path.join(DATA_FOLDER, "hyrule_crests.pdf"),
        extract_images_in_pdf=True,  # use pdf format to find embedded image blocks
        infer_table_structure=True,  # use YOLOX to find tables and titles
        # chunking_strategy="by_title",  # if enabled, it does not extract images :(
        max_characters=MAX_CHUNK_SIZE,  # max chunk size.
        new_after_n_chars=int(MAX_CHUNK_SIZE*0.9),  # start a new chunk asap when you exceed 90% of the max chunk size.
        combine_text_under_n_chars=int(MAX_CHUNK_SIZE*0.45),  # merge chunks smaller than 45% of max chunk size.
        image_output_dir_path=DATA_PROCESSING_FOLDER,
    )

    # Split the obtained raw_pdf_elements into groups using the title elements as separators.
    all_documents = []
    cur_document_contents = []
    for el in raw_pdf_elements:

        # We assume the first element in a PDF is always a title.
        if el.category == "Title":
            if cur_document_contents != []:
                new_doc = Document(page_content=f"{cur_document_contents[0]}: {' '.join(cur_document_contents[1:])}")
                all_documents.append(new_doc)
                cur_document_contents = []
            cur_document_contents.append(el.text)

        elif el.category == "Image":
            img_filepath = el.metadata.image_path
            img_caption = _get_image_caption(img_filepath)
            cur_document_contents.append(img_caption)

        else:
            cur_document_contents.append(el.text)

    if cur_document_contents != []:
        final_doc = Document(page_content=f"{cur_document_contents[0]}: {' '.join(cur_document_contents[1:])}")
        all_documents.append(final_doc)

    print(all_documents)
    return all_documents


# Load all documents and split them into 1000-sized chunks.
documents = []
print(f"Scanning {DATA_FOLDER} for docs...")
for filename in os.listdir(DATA_FOLDER):
    filepath = os.path.join(DATA_FOLDER, filename)
    if filepath.endswith(".txt"):
        documents += _txt_to_documents(filepath)
    elif filepath.endswith(".png"):
        documents += _image_to_documents(filepath)
    elif filepath.endswith(".pdf"):
        documents += _pdf_to_documents(filepath)
    else:
        print(f"\tWARNING: Document of unsupported type: {filepath}")

# Split all the obtained docs into chunks.
text_splitter = CharacterTextSplitter(chunk_size=MAX_CHUNK_SIZE, chunk_overlap=0)
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
