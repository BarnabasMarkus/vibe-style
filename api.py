import os
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
import search
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
api_logger = logging.getLogger("api_logger")
api_logger.setLevel(logging.INFO)

# Avoid adding duplicate handlers
if not api_logger.handlers:
    log_file = os.getenv("API_LOG_FILE", "api_calls.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    api_logger.addHandler(file_handler)

# Load configurations from environment variables
image_folder = os.getenv("IMAGE_FOLDER", "fashion-dataset/images")
index_path = os.getenv("INDEX_PATH", "image_index.bin")
port = int(os.getenv("PORT", 8000))

# Load model, processor, image paths, and index
model, processor = search.load_clip_model()
image_paths = search.load_image_paths(image_folder)
index = search.load_faiss_index(index_path)

# Application setup
app = FastAPI()

# Custom middleware for logging API calls
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Log the incoming request with query parameters
        query_params = dict(request.query_params)
        api_logger.info(f"Request: {request.method} {request.url}, Params: {query_params}")
        response = await call_next(request)
        # Log the response status
        api_logger.info(f"Response: {response.status_code} {request.url}")
        return response

# Add the middleware to the FastAPI app
app.add_middleware(LoggingMiddleware)

@app.get("/search")
async def search_images(query: str, top_k: int = 5):
    """Search for images matching the text query."""

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    if top_k <= 0:
        raise HTTPException(status_code=400, detail="Top K must be a positive integer.")
    if top_k > 100:
        raise HTTPException(status_code=400, detail="Top K cannot exceed 100.")

    try:
        results = search.get_top_matches(query, model, processor, index, image_paths, top_k)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during the search process.") from e

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
