from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import uvicorn
from dotenv import load_dotenv
from v4_rag_agent import agent, index_documents as index_documents_v4


load_dotenv()

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, RAG Agent!"}

@app.post("/index_documents")
def index_documents(documents: list[UploadFile] = File(...)):
    for document in documents:
        document_content = document.file.read()
        document_name = document.filename
        document_path = f"documents/{document_name}"
        with open(document_path, "wb") as f:
            f.write(document_content)
        index_documents_v4(document_path)
    return {"message": "Documents indexed successfully"}

@app.get("/rag_agent")
def query_documents(question: str):
    input_message = question
    config = {"configurable": {"thread_id": "def234"}}

    response = agent.invoke({"messages": [{"role": "user", "content": input_message}]}, config=config)
    return response["messages"][-1].content

@app.get("/rag_agent/stream")
def query_documents_stream(question: str):
    input_message = question
    config = {"configurable": {"thread_id": "def234"}}

    def stream_response():
        for event in agent.stream(
            {"messages": [{"role": "user", "content": input_message}]},
            stream_mode="values",
            config=config,
        ):
            event["messages"][-1].pretty_print()
            message = event["messages"][-1]
            content = getattr(message, "content", None)
            if content:
                yield f"data: {content}\n\n"

    return StreamingResponse(stream_response(), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)