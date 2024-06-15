"""Script to create a Langchain chain that performs RAG accessing docs in pinecone."""

import os

from keys import OPENAI_KEY, PINECONE_KEY
from constants import PINECONE_INDEX_NAME

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

# Set the PINECONE_KEY in the environment so we can connect to our PineconeVectorStore.
os.environ['PINECONE_API_KEY'] = PINECONE_KEY

# Endpoint to the Pinecone API for our vector store and index. It is already populated.
embedding = OpenAIEmbeddings(openai_api_key=OPENAI_KEY)
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embedding
)

# Endpoint to the OpenAI APIs for the LLM.
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=OPENAI_KEY,
    # base_url="...",
    # organization="...",
    # other params...
)

# Code to *manually* get the chunks that match the query the most.
# query = "What is the religion of Hyrule?"
# out = vectorstore.similarity_search(query, k=3)
# Let's not do this manually, though - we will use a document chain instead.

# Prompt with context and input to fill.
prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context\n:
<context>\n{context}\n</context>\n
Question: {input}""")

# Chain to pass a list of documents (chunks, in our case) as context to an LLM while asking a question.
# The prompt MUST contain placeholder {context} - that is where the documents are injected.
document_chain = create_stuff_documents_chain(llm, prompt)

# Create a larger retrieval chain that:
#  - when receiving a query, sends it to the vectorstore to retrieve the top 3 documents (chunks) matching it;
#  - injects in the prompt the query as input and the top docs as context, and sends it to the LLM as a query;
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})  # I love that the vectorstore can act as a retriever <3
retrieval_chain = create_retrieval_chain(retriever, document_chain)

question = "What is the Dark Plague in Hyrule?"
print(f"Asking the following question:\n\t{question}")

answer_without_retrieval = llm.invoke(question)
print(f"Answer without additional context:\n\t{answer_without_retrieval.content}")

answer_with_retrieval = retrieval_chain.invoke({"input": question})
print(f"Answer with additional context:\n\t{answer_with_retrieval['answer']}")
