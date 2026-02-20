import asyncio
import json
import os
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from pageindex import page_index
from pageindex.page_index_md import md_to_tree


RESULTS_DIR = Path("./results")
UPLOADS_DIR = Path("./uploads")
TASKS_DIR = Path("./tasks")
for folder in [RESULTS_DIR, UPLOADS_DIR, TASKS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)


TASK_LOCK = threading.Lock()
TASKS: dict[str, dict] = {}
MAX_WORKERS = max(1, int(os.getenv("PAGEINDEX_MAX_WORKERS", "2")))
EXECUTOR = ThreadPoolExecutor(max_workers=MAX_WORKERS)

app = FastAPI(title="PageIndex API", version="2.0.0")


def _to_yes_no(flag: bool) -> str:
    return "yes" if flag else "no"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sanitize_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def _task_file(task_id: str) -> Path:
    return TASKS_DIR / f"{task_id}.json"


def _persist_task(task_id: str) -> None:
    task_path = _task_file(task_id)
    with task_path.open("w", encoding="utf-8") as f:
        json.dump(TASKS[task_id], f, indent=2, ensure_ascii=False)


def _set_task(task_id: str, **fields) -> None:
    with TASK_LOCK:
        if task_id not in TASKS:
            return
        TASKS[task_id].update(fields)
        _persist_task(task_id)


def _get_task(task_id: str) -> dict:
    with TASK_LOCK:
        if task_id in TASKS:
            return dict(TASKS[task_id])

    task_path = _task_file(task_id)
    if not task_path.exists():
        raise HTTPException(status_code=404, detail="Task not found.")
    with task_path.open("r", encoding="utf-8") as f:
        task = json.load(f)
    with TASK_LOCK:
        TASKS[task_id] = task
    return dict(task)


def _load_tasks_from_disk() -> None:
    for task_path in TASKS_DIR.glob("*.json"):
        with task_path.open("r", encoding="utf-8") as f:
            task = json.load(f)
        # A restarted worker cannot continue in-flight tasks.
        if task.get("status") in {"queued", "running"}:
            task["status"] = "failed"
            task["error"] = "Server restarted before task completion."
            task["completed_at"] = _now_iso()
            with task_path.open("w", encoding="utf-8") as f:
                json.dump(task, f, indent=2, ensure_ascii=False)
        task_id = task.get("task_id")
        if task_id:
            TASKS[task_id] = task


def _save_upload(file: UploadFile, task_id: str) -> Path:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in [".pdf", ".md", ".markdown"]:
        raise HTTPException(status_code=400, detail="Only .pdf, .md, .markdown are supported.")

    safe_name = _sanitize_name(Path(file.filename or "input").name)
    target = UPLOADS_DIR / f"{task_id}_{safe_name}"
    with target.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    return target


def _build_output_path(task_id: str, input_path: Path, output_name: str | None) -> Path:
    if output_name:
        safe_output = _sanitize_name(output_name)
    else:
        safe_output = f"{input_path.stem}_structure.json"
    if not safe_output.endswith(".json"):
        safe_output = f"{safe_output}.json"
    return RESULTS_DIR / f"{task_id}_{safe_output}"


def _run_generation(input_path: Path, ext: str, params: dict) -> dict:
    if ext == ".pdf":
        return page_index(
            str(input_path),
            model=params["model"],
            toc_check_page_num=params["toc_check_pages"],
            max_page_num_each_node=params["max_pages_per_node"],
            max_token_num_each_node=params["max_tokens_per_node"],
            if_add_node_id=_to_yes_no(params["if_add_node_id"]),
            if_add_node_summary=_to_yes_no(params["if_add_node_summary"]),
            if_add_doc_description=_to_yes_no(params["if_add_doc_description"]),
            if_add_node_text=_to_yes_no(params["if_add_node_text"]),
        )

    return asyncio.run(
        md_to_tree(
            md_path=str(input_path),
            if_thinning=params["if_thinning"],
            min_token_threshold=params["thinning_threshold"],
            if_add_node_summary=_to_yes_no(params["if_add_node_summary"]),
            summary_token_threshold=params["summary_token_threshold"],
            model=params["model"],
            if_add_doc_description=_to_yes_no(params["if_add_doc_description"]),
            if_add_node_text=_to_yes_no(params["if_add_node_text"]),
            if_add_node_id=_to_yes_no(params["if_add_node_id"]),
        )
    )


def _worker(task_id: str, input_path: Path, ext: str, params: dict, output_path: Path) -> None:
    _set_task(task_id, status="running", started_at=_now_iso())
    try:
        result = _run_generation(input_path, ext, params)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        _set_task(
            task_id,
            status="completed",
            completed_at=_now_iso(),
            output_path=str(output_path.resolve()),
            download_url=f"/tasks/{task_id}/download",
            result_url=f"/tasks/{task_id}/result",
        )
    except Exception as exc:
        _set_task(
            task_id,
            status="failed",
            completed_at=_now_iso(),
            error=str(exc),
        )


_load_tasks_from_disk()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/tasks")
def create_task(
    file: UploadFile = File(...),
    model: str = Form("gpt-4o-2024-11-20"),
    toc_check_pages: int = Form(20),
    max_pages_per_node: int = Form(10),
    max_tokens_per_node: int = Form(20000),
    if_add_node_id: bool = Form(True),
    if_add_node_summary: bool = Form(True),
    if_add_doc_description: bool = Form(False),
    if_add_node_text: bool = Form(False),
    if_thinning: bool = Form(False),
    thinning_threshold: int = Form(5000),
    summary_token_threshold: int = Form(200),
    output_name: str | None = Form(None),
):
    if not (os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")):
        raise HTTPException(status_code=400, detail="CHATGPT_API_KEY or OPENAI_API_KEY is not set.")

    task_id = uuid4().hex
    input_path = _save_upload(file, task_id)
    ext = input_path.suffix.lower()
    output_path = _build_output_path(task_id, input_path, output_name)

    task = {
        "task_id": task_id,
        "status": "queued",
        "created_at": _now_iso(),
        "started_at": None,
        "completed_at": None,
        "input_file": str(input_path.resolve()),
        "input_type": ext.lstrip("."),
        "output_path": None,
        "error": None,
        "status_url": f"/tasks/{task_id}",
        "download_url": None,
        "result_url": None,
    }

    with TASK_LOCK:
        TASKS[task_id] = task
        _persist_task(task_id)

    params = {
        "model": model,
        "toc_check_pages": toc_check_pages,
        "max_pages_per_node": max_pages_per_node,
        "max_tokens_per_node": max_tokens_per_node,
        "if_add_node_id": if_add_node_id,
        "if_add_node_summary": if_add_node_summary,
        "if_add_doc_description": if_add_doc_description,
        "if_add_node_text": if_add_node_text,
        "if_thinning": if_thinning,
        "thinning_threshold": thinning_threshold,
        "summary_token_threshold": summary_token_threshold,
    }
    EXECUTOR.submit(_worker, task_id, input_path, ext, params, output_path)

    return {
        "task_id": task_id,
        "status": "queued",
        "status_url": f"/tasks/{task_id}",
        "download_url": f"/tasks/{task_id}/download",
        "result_url": f"/tasks/{task_id}/result",
    }


@app.get("/tasks/{task_id}")
def get_task(task_id: str) -> dict:
    return _get_task(task_id)


@app.get("/tasks/{task_id}/result")
def get_task_result(task_id: str) -> dict:
    task = _get_task(task_id)
    if task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Task is not completed yet.")
    output_path = task.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found.")
    with Path(output_path).open("r", encoding="utf-8") as f:
        return json.load(f)


@app.get("/tasks/{task_id}/download")
def download_task_result(task_id: str):
    task = _get_task(task_id)
    if task.get("status") != "completed":
        raise HTTPException(status_code=409, detail="Task is not completed yet.")
    output_path = task.get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Result file not found.")
    return FileResponse(path=output_path, media_type="application/json", filename=Path(output_path).name)
