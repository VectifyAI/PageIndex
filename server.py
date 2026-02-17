from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import glob
from typing import List, Optional
import json
import logging
from dotenv import load_dotenv

# Import PageIndex logic
from pageindex.page_index import page_index_main
from types import SimpleNamespace as config
import tkinter as tk
from tkinter import filedialog

# Load environment variables
load_dotenv()

# Configure logging
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/server.log"),
        logging.StreamHandler()
    ]
)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for local development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request Models
class ScanRequest(BaseModel):
    path: str
    
class ProcessRequest(BaseModel):
    files: List[str]
    model: str = "deepseek-chat"  # Default model
    max_pages: int = 10
    max_tokens: int = 20000

# API Endpoints

@app.get("/api/health")
async def health_check():
    return {"status": "ok"}

@app.get("/api/choose-path")
async def choose_path():
    """Open a native folder selection dialog."""
    try:
        # Create hidden root window
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1) # Bring to front
        
        # Open directory dialog
        path = filedialog.askdirectory(title="选择 PDF 文件夹")
        
        # Destroy root window
        root.destroy()
        
        return {"path": path}
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return {"path": "", "error": str(e)}

@app.post("/api/scan")
async def scan_directory(request: ScanRequest):
    """Scan a directory for PDF files."""
    if not os.path.exists(request.path):
        raise HTTPException(status_code=404, detail="Directory not found")
    
    if os.path.isfile(request.path):
         if request.path.lower().endswith(".pdf"):
             return {"files": [request.path]}
         else:
             return {"files": []}

    pdf_files = []
    # Support both absolute paths and relative to CWD
    search_path = os.path.join(request.path, "*.pdf")
    files = glob.glob(search_path)
    
    # Also try recursive scan if no files found in top level? 
    # For now, stick to top level for simplicity, or we can use glob.glob(..., recursive=True)
    
    return {"files": sorted(files)}

@app.post("/api/process")
async def process_files(request: ProcessRequest):
    """Process a list of PDF files."""
    results = []
    
    # Configure options
    opt = config(
        model=request.model,
        toc_check_page_num=20,
        max_page_num_each_node=request.max_pages,
        max_token_num_each_node=request.max_tokens,
        if_add_node_id="yes",
        if_add_node_summary="yes",
        if_add_doc_description="no",
        if_add_node_text="no"
    )

    for pdf_path in request.files:
        if not os.path.exists(pdf_path):
            results.append({"file": pdf_path, "status": "error", "error": "File not found"})
            continue
            
        try:
            # Run PageIndex logic
            result_data = await page_index_main(pdf_path, opt)
            
            # Save result to file
            pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]
            output_dir = './results'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{pdf_name}_structure.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)
                
            results.append({"file": pdf_path, "status": "success"})
        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {e}")
            results.append({"file": pdf_path, "status": "error", "error": str(e)})

    return {"results": results}

@app.get("/api/results")
async def list_results():
    """List generated JSON result files."""
    results_dir = "./results"
    if not os.path.exists(results_dir):
        return {"files": []}
        
    files = glob.glob(os.path.join(results_dir, "*.json"))
    # Sort by modification time, newest first
    files.sort(key=os.path.getmtime, reverse=True)
    
    return {"files": [{"name": os.path.basename(f), "path": f} for f in files]}

@app.get("/api/results/{filename}")
async def get_result_content(filename: str):
    """Get the content of a specific result file."""
    filepath = os.path.join("./results", filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="File not found")
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = json.load(f)
        return content
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
