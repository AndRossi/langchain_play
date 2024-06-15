import json

# Get all API keys from a (gitignored) file.
ALL_KEYS = json.load(open("keys.json"))

OPENAI_KEY = ALL_KEYS["OPENAI_KEY"]
LANGSMITH_KEY = ALL_KEYS["LANGSMITH_KEY"]
PINECONE_KEY = ALL_KEYS["PINECONE_KEY"]
TAVILY_KEY = ALL_KEYS["TAVILY_KEY"]
