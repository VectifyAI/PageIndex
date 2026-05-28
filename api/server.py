from fastapi import FastAPI

from api.routers import api_router

def create_app() -> FastAPI:
    app = FastAPI(
        title="PageIndex API",
        version="1.0.0",
        description="HTTP API for indexing markdown documents using the PageIndex PDF pipeline.",
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
