## RAG Agent API (FastAPI + LangChain + LangGraph)

Build and run a Retrieval-Augmented Generation (RAG) agent over PDFs. The API indexes PDF documents into an in-memory vector store and serves a ReAct-style agent that retrieves relevant chunks to answer questions.

### Features
- **PDF indexing**: Splits PDFs into chunks and embeds them
- **In-memory vector store**: Fast similarity search (reset on restart)
- **ReAct agent**: Uses a retrieval tool to ground answers
- **Streaming responses**: Server-Sent Events (SSE) endpoint

## Requirements
- Python 3.10+
- An OpenAI API key

### Python dependencies
Install via pip (recommended in a virtual environment):

```bash
pip install -U fastapi uvicorn python-dotenv langchain langchain-community langchain-openai langgraph langchain-text-splitters
```

## Environment variables
Create a `.env` file in the project root with:

```bash
OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

## Run the API
You can run the server either via `python` (uses the embedded `uvicorn.run`) or directly with `uvicorn`:

```bash
# Option A: Python entrypoint
python /Users/spedemonte/Coding/RAG-Agent/main.py

# Option B: Uvicorn (reload for development)
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Once running, visit `http://localhost:8000/` to verify.

## How it works
- `v4_rag_agent.py`:
  - Initializes a chat model: `gpt-4o-mini` via `langchain.chat_models.init_chat_model` with the OpenAI provider
  - Initializes embeddings: `text-embedding-3-small` via `langchain_openai.OpenAIEmbeddings`
  - Uses `InMemoryVectorStore` as the vector store (cleared on process restart)
  - Splits PDFs using `RecursiveCharacterTextSplitter` with `chunk_size=1000`, `chunk_overlap=200`
  - Defines a `retrieve` tool that performs `similarity_search(..., k=3)` and returns sources+content
  - Builds a ReAct agent with `langgraph.prebuilt.create_react_agent` and `MemorySaver`
  - On import, indexes `documents/manual_procedimientos.pdf` by default
- `main.py`:
  - Exposes FastAPI endpoints to index more PDFs and query the agent (plain and streaming)

## API Endpoints

### GET `/`
Health check.

```bash
curl http://localhost:8000/
```

### POST `/index_documents`
Upload and index one or more PDF files. Files are saved to `documents/` and immediately added to the in-memory vector store.

Form field name: `documents`

```bash
# Single PDF
curl -X POST "http://localhost:8000/index_documents" \
  -H "Accept: application/json" \
  -F "documents=@/absolute/path/to/your.pdf"

# Multiple PDFs
curl -X POST "http://localhost:8000/index_documents" \
  -H "Accept: application/json" \
  -F "documents=@/absolute/path/one.pdf" \
  -F "documents=@/absolute/path/two.pdf"
```

Response:
```json
{"message": "Documents indexed successfully"}
```

### GET `/rag_agent`
Query the RAG agent. Provide your question via the `question` query param.

```bash
curl "http://localhost:8000/rag_agent?question=¿Cuál%20es%20el%20procedimiento%20estándar?"
```

Returns the final answer as plain text. The agent prompt is in Spanish and will attempt to answer using only the indexed documents; if it cannot find the answer, it will say it does not know.

### GET `/rag_agent/stream`
Stream the agent’s answer via Server-Sent Events (SSE). Useful for progressive display.

```bash
curl -N "http://localhost:8000/rag_agent/stream?question=Explica%20el%20flujo%20de%20aprobación"
```

The stream sends lines in the form `data: <content>` as they are generated.

## Notes
- The vector store is in-memory; restart will clear the index. Re-index your PDFs after restarting.
- On startup, `v4_rag_agent.py` automatically indexes `documents/manual_procedimientos.pdf`. Change `file_path` there to adjust the default.
- The agent uses a `retrieve` tool internally; it may reformulate your question and perform multiple searches to answer.
- The agent’s system prompt is in Spanish.

## Project structure (relevant files)
- `main.py`: FastAPI app and endpoints
- `v4_rag_agent.py`: Indexing, retrieval tool, and agent creation
- `documents/`: Place your PDFs here (uploaded files are saved here automatically)

## Troubleshooting
- Missing OpenAI key: ensure `.env` contains `OPENAI_API_KEY` and the shell has access to it
- No results found: verify your PDFs were successfully uploaded and indexed
- Import errors: ensure all packages in the dependencies section are installed
