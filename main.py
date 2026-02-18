import sys
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from rag import index_documents
from memory import save_conversation

from tools import (
    get_flight_schedule,
    get_hotel_schedule,
    convert_currency,
    query_internal_knowledge
)



load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_API_KEY = os.getenv("HF_API_KEY")

if not OPENROUTER_API_KEY:
    raise RuntimeError("OPENROUTER_API_KEY not set")
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"



# Only index if vector DB doesn't already exist
if not os.path.exists("chroma_db"):
    print("Vector store not found. Indexing documents...")
    index_documents()
else:
    print("Vector store already exists. Skipping indexing.")


prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
if not prompt:
    raise RuntimeError("Please provide a prompt")


tools = [
    get_flight_schedule,
    get_hotel_schedule,
    convert_currency,
    query_internal_knowledge
]


agent = create_agent(
    model="openai:nvidia/nemotron-3-nano-30b-a3b:free",
    tools=tools,
    system_prompt="You are a helpful AI assistant.",
    # debug=True
)


# Save user message
save_conversation("user", prompt)

response = agent.invoke({
    "messages": [
        {"role": "user", "content": prompt}
    ]
})

final_output = response["messages"][-1].content

# Save assistant message
save_conversation("assistant", final_output)

print("\n=== Conversation ===")
print(f"User: {prompt}")
print(f"Assistant: {final_output}")