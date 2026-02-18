
import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"

# Load embeddings once
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_vectorstore():
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings
    )



def index_documents():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    loaders = [
        DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader),
        DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyMuPDFLoader),
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    if not documents:
        return

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)

    print("Indexing done")

vectorstore = get_vectorstore()


def get_retriever():
    return vectorstore.as_retriever(search_kwargs={"k": 4})
