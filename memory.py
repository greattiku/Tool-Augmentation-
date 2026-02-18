from langchain_core.documents import Document
from rag import get_vectorstore

def save_conversation(role: str, content: str):
    vectorstore = get_vectorstore()

    doc = Document(
        page_content=f"{role.upper()}: {content}",
        metadata={"role": role, "type": "conversation"}
    )

    vectorstore.add_documents([doc])

