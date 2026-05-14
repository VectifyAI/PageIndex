import os
import tempfile
import anthropic
from fastapi import FastAPI, File, HTTPException, Header, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pageindex.client import PageIndexClient

app = FastAPI(title="PageIndex API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

WORKSPACE_DIR = os.environ.get("WORKSPACE_DIR", "./workspace")
API_SECRET = os.environ.get("API_SECRET", "")
os.makedirs(WORKSPACE_DIR, exist_ok=True)

pi_client = PageIndexClient(
    model=os.environ.get("PAGEINDEX_MODEL", "claude-sonnet-4-6"),
    workspace=WORKSPACE_DIR,
)

anthropic_client = anthropic.Anthropic()


def verify_secret(x_api_secret: str = Header(default="")):
    if API_SECRET and x_api_secret != API_SECRET:
        raise HTTPException(status_code=401, detail="Invalid API secret.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
async def index_document(
    file: UploadFile = File(...),
    x_api_secret: str = Header(default=""),
):
    verify_secret(x_api_secret)
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        doc_id = pi_client.index(tmp_path)
        return {"doc_id": doc_id, "filename": file.filename}
    finally:
        os.unlink(tmp_path)


class QueryRequest(BaseModel):
    doc_id: str
    question: str


@app.post("/query")
def query_document(
    req: QueryRequest,
    x_api_secret: str = Header(default=""),
):
    verify_secret(x_api_secret)
    structure = pi_client.get_document_structure(req.doc_id)

    tools = [
        {
            "name": "get_page_content",
            "description": "Retrieve the text of specific pages from the document.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "pages": {
                        "type": "string",
                        "description": "Pages to fetch. Examples: '5-7', '3,8', '12'.",
                    }
                },
                "required": ["pages"],
            },
        }
    ]

    messages = [
        {
            "role": "user",
            "content": f"Document structure:\n{structure}\n\nQuestion: {req.question}",
        }
    ]

    for _ in range(10):
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2048,
            system=(
                "You are a document analysis assistant. "
                "Use the document structure to navigate the document and answer the question. "
                "Always cite the page numbers where you found the information."
            ),
            tools=tools,
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            answer = next(
                (b.text for b in response.content if hasattr(b, "text")), ""
            )
            return {"answer": answer}

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use" and block.name == "get_page_content":
                    content = pi_client.get_page_content(req.doc_id, block.input["pages"])
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": content}
                    )
            messages.append({"role": "assistant", "content": response.content})
            messages.append({"role": "user", "content": tool_results})

    raise HTTPException(status_code=500, detail="Could not answer the question.")
