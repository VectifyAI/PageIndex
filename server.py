import os
import tempfile
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse

from pageindex import PageIndexClient

app = FastAPI(title="PageIndex API", version="1.0.0")

WORKSPACE = os.environ.get("PAGEINDEX_WORKSPACE", "/tmp/pageindex_workspace")
MODEL = os.environ.get("PAGEINDEX_MODEL", None)
RETRIEVE_MODEL = os.environ.get("PAGEINDEX_RETRIEVE_MODEL", None)

client = PageIndexClient(
    model=MODEL,
    retrieve_model=RETRIEVE_MODEL,
    workspace=WORKSPACE,
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents")
def list_documents():
    """List all indexed documents."""
    docs = []
    for doc_id, doc in client.documents.items():
        docs.append({
            "doc_id": doc_id,
            "doc_name": doc.get("doc_name", ""),
            "type": doc.get("type", ""),
        })
    return docs


@app.post("/index")
async def index_document(file: UploadFile = File(...)):
    """Upload and index a PDF or Markdown file."""
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in (".pdf", ".md", ".markdown"):
        raise HTTPException(status_code=400, detail="Only .pdf, .md, and .markdown files are supported")

    tmp_dir = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, file.filename)
    try:
        with open(tmp_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        doc_id = client.index(tmp_path)
    except Exception as e:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))

    return {"doc_id": doc_id}


@app.get("/documents/{doc_id}")
def get_document(doc_id: str):
    """Get document metadata."""
    import json
    result = json.loads(client.get_document(doc_id))
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/documents/{doc_id}/structure")
def get_structure(doc_id: str):
    """Get document tree structure."""
    import json
    result = json.loads(client.get_document_structure(doc_id))
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/documents/{doc_id}/pages")
def get_pages(doc_id: str, pages: str):
    """Get page content. Use pages param like '5-7', '3,8', or '12'."""
    import json
    result = json.loads(client.get_page_content(doc_id, pages))
    if isinstance(result, dict) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
    return result
