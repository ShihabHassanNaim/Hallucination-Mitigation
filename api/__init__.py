"""CRISP FastAPI web service.

Exports create_app() so tests can build isolated instances.
Run the dev server with:

    cd E:\\Thesis\\Hallucination
    .venv\\Scripts\\python.exe -m uvicorn api.app:create_app --factory --reload
"""
from .app import create_app

__all__ = ["create_app"]
