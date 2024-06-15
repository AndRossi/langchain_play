"""Script to create a Langchain Agent that can perform RAG in a REACT fashion."""

import os

from keys import OPENAI_KEY, PINECONE_KEY, TAVILY_KEY
from constants import PINECONE_INDEX_NAME

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.tools.retriever import create_retriever_tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

# Set the PINECONE_KEY in the environment so we can query our PineconeVectorStore in the retrieval tool.
os.environ['PINECONE_API_KEY'] = PINECONE_KEY
# Set the TAVILY_KEY in the environment so we can connect to Tavily APIs in the search tool.
os.environ['TAVILY_API_KEY'] = TAVILY_KEY

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


# Use prompt hwchase17/openai-functions-agent.
# It has two placeholders, 'agent_scratchpad' and 'input', and supports chat history.
# Prompt details:
#   input_variables=['agent_scratchpad', 'input']
#   input_types={
#       'chat_history': a list of all sorts of langchain_core.messages,
#       'agent_scratchpad': a list of all sorts of langchain_core.messages
#   }
prompt = hub.pull("hwchase17/openai-functions-agent")

# Let's build the two tools.

# Retriever tool: use this for Hyrule-themed questions only.
# We will use as retriever our usual vectorstore full of chunks of Hyrule docs.
retriever_tool = create_retriever_tool(
    retriever,
    "hyrule_search",
    "Search for information about Hyrule. For any questions about Hyrule, you must use this tool!",
)

# Search tool: use this for any other question.
# We will use Tavily as a search tool.
search_tool = TavilySearchResults()

tools = [retriever_tool, search_tool]

# Create the agent.
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Ask a question about Hyrule - make sure we are using the retrieval tool.
question_about_hyrule = "What is the Dark Plague of Hyrule?"
print(f"Asking the following question about Hyrule:\n\t{question_about_hyrule}")
answer_about_hyrule = agent_executor.invoke({"input": question_about_hyrule})
print(f"Answer (should use the retrieval agent):\n\t{answer_about_hyrule['output']}")

print("\n##########################################################################################\n")

# Ask a question about something else - make sure we are using the search tool.
question_about_something_else = "What is the weather like in Zurich right now?"
print(f"Asking the following question about something else:\n\t{question_about_something_else}")
answer_about_something_else = agent_executor.invoke({"input": question_about_something_else})
print(f"Answer (should use the search agent):\n\t{answer_about_something_else['output']}")

print("\n##########################################################################################\n")

# Note: we can also use chat history with agents:
chat_human_question_1 = "Are multiple deities in worshipped in Hyrule"
chat_ai_answer_1 = "Yes!"
chat_human_question_2 = "Tell me which ones please."
print(
    "Asking the following question with the given chat:"
    f"\n\t{chat_human_question_1}\n\t{chat_ai_answer_1}\n\t{chat_human_question_2}"
)

chat_history = [HumanMessage(content=chat_human_question_1), AIMessage(content=chat_ai_answer_1)]
chat_ai_answer_2 = agent_executor.invoke({
    "chat_history": chat_history,
    "input": chat_human_question_2
})
print(f"Answer (should use the search agent):\n\t{chat_ai_answer_2['output']}")
