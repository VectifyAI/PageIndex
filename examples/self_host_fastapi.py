import os
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile

from pageindex.page_index import page_index_main
from pageindex.utils import config


app = FastAPI(title="PageIndex (Self-Hosted Example)")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/index")
def index_pdf(
    file: UploadFile = File(...),
    model: str = "gpt-4o-2024-11-20",
    toc_check_pages: int = 20,
    max_pages_per_node: int = 10,
    max_tokens_per_node: int = 20000,
    if_add_node_id: str = "yes",
    if_add_node_summary: str = "yes",
    if_add_doc_description: str = "no",
    if_add_node_text: str = "no",
):
    filename = file.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf uploads are supported")

    pdf_bytes = file.file.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty upload")

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_bytes)
            tmp_path = tmp.name

        opt = config(
            model=model,
            toc_check_page_num=toc_check_pages,
            max_page_num_each_node=max_pages_per_node,
            max_token_num_each_node=max_tokens_per_node,
            if_add_node_id=if_add_node_id,
            if_add_node_summary=if_add_node_summary,
            if_add_doc_description=if_add_doc_description,
            if_add_node_text=if_add_node_text,
        )

        return page_index_main(tmp_path, opt)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
