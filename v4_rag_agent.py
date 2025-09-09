from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from dotenv import load_dotenv
from langchain_core.documents import Document

load_dotenv()

# Initialize the chat model
from langchain.chat_models import init_chat_model
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

# Initialize the embedding model
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small") 

# Initialize the vector store
from langchain_core.vectorstores import InMemoryVectorStore
vector_store = InMemoryVectorStore(embeddings)

file_path = "documents/manual_procedimientos.pdf"

#  ==== 1) Indexing ====

def split_documents(file_path: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[Document]:
    print("Splitting...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits


def vector_store_documents(all_splits: list[Document]) -> list[str]:
    print("Vector Storing...")
    return vector_store.add_documents(documents=all_splits)


def index_documents(file_path: str) -> list[Document]:
    print("Indexing...")
    all_splits = split_documents(file_path)
    vector_store = vector_store_documents(all_splits)

    return vector_store

index_documents(file_path)

# #  ==== 2) Retrieval ====

@tool(response_format="content_and_artifact")
def retrieve(query: str):
    """Retrieve information related to a query."""
    retrieved_docs = vector_store.similarity_search(query, k=3)
    serialized = "\n\n".join(
        (f"Source: {doc.metadata}\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )
    return serialized, retrieved_docs

# ==== 3) Agent ====

prompt = """
Eres un asistente que se encarga de responder preguntas al usuario sobre el material adjunto.
para acceder al material adjunto, debes usar la herramienta retrieve..
Cuando uses la herramienta retrieve, puedes reformular la pregunta del usuario para obtener mejores resultados.
Tambien puedes realizar mas de una busqueda para llegara a la respuesta.
Si no encuentras la respuesta en el material adjunto, debes decir que no sabes la respuesta.
"""

memory = MemorySaver()
agent = create_react_agent(llm, [retrieve], checkpointer=memory, prompt=prompt)

# config = {"configurable": {"thread_id": "def234"}}
# input_message = (
#     "What is the standard method for Task Decomposition?\n\n"
#     "Once you get the answer, look up common extensions of that method."
# )
# for event in agent.stream(
#     {"messages": [{"role": "user", "content": input_message}]},
#     stream_mode="values",
#     config=config,
# ):
#     event["messages"][-1].pretty_print()