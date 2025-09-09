from dotenv import load_dotenv
load_dotenv()


# Loading the documents
from langchain_community.document_loaders import PyPDFLoader

file_path = "constitucion_nacional.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(len(docs))


# Splitting the documents into chunks

from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200,
    add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))


# Embedding the chunks
from langchain_openai import OpenAIEmbeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")


# Creating the vector store
from langchain_core.vectorstores import InMemoryVectorStore

vector_store = InMemoryVectorStore(embedding=embeddings)
ids = vector_store.add_documents(documents=all_splits)


# Similarity search
query = "En que consiste el preambulo?"
results = vector_store.similarity_search(query)
print(results)
