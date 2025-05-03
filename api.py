import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import uvicorn
import logging
from fastapi import FastAPI, HTTPException, Request, Depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response
import utils

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

# Load resources
device, model, processor, index, image_paths = utils.load_rescources()

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
        results = utils.search_images_by_vibe(query=query,
                                              processor=processor,
                                              model=model,
                                              index=index,
                                              image_paths=image_paths,
                                              top_k=top_k,
                                              device=device)
        # Convert results to a list of dicts with native Python types
        image_paths_list = list(results[0])
        scores_list = [float(s) for s in results[1]]
        return {"results": [
            {"image_path": path, "score": score}
            for path, score in zip(image_paths_list, scores_list)
        ]}
    except Exception as e:
        raise HTTPException(status_code=500, detail="An error occurred during the search process.") from e

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
