# VibeStyle

Ever wondered what "business casual with a twist of avocado toast" looks like? Or want to see "cyberpunk raincoat" or "grandma's chic knitting club" in fashion form? VibeStyle lets you search for wild, fun, or classic outfit vibes—just type your dream look and see what pops up!

VibeStyle is a fashion search engine that helps you find outfits matching any vibe you can describe—like "casual summer dress", "retro 90s street style", or "elegant evening gown"—using deep learning and vector similarity search. It provides both a web UI (Streamlit) and an API (FastAPI) for searching a large collection of fashion images.

## Features

- **Text-to-image search:** Enter a description (e.g., "casual summer dress") to find matching outfits.
- **Streamlit UI:** User-friendly web interface for interactive searching.
- **FastAPI backend:** REST API for programmatic access.
- **Efficient search:** Uses FAISS for fast vector similarity search over image embeddings.
- **Example queries:** Quickly try out popular fashion vibes.

## Project Structure

- `app.py` — Streamlit web application.
- `api.py` — FastAPI server for API access.
- `build_index.py` — Script to build the FAISS index from image embeddings.
- `config.py` — Configuration (paths, model names, etc).
- `utils.py` — Utility functions (loading models, searching, etc).
- `fashion_index.faiss` — Saved FAISS index (generated).
- `image_paths.pt` — List of image paths (generated).
- `fashion-dataset/images/` — Folder containing fashion images.

## Setup

1. **Clone the repository** and install dependencies:
    ```bash
    git clone <repo-url>
    cd vibe-style
    pip install -r requirements.txt
    ```

2. **Prepare the dataset:**
    - Place your fashion images in the folder specified by `image_folder` in `config.py`.

3. **Build the index:**
    ```bash
    python build_index.py
    ```

4. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

5. **Run the API server:**
    ```bash
    python api.py
    ```

## Usage

- **Web UI:** Open the Streamlit app in your browser. Enter a description or click an example to search for outfits.
- **API:** Send a GET request to `/search` with a `query` parameter.

    Example:
    ```
    GET http://localhost:8000/search?query=casual+summer+dress
    ```

    Or use `curl` from the command line:
    ```bash
    curl "http://localhost:8000/search?query=retro+90s+street+style"
    curl "http://localhost:8000/search?query=business+casual+with+a+twist+of+avocado+toast"
    curl "http://localhost:8000/search?query=grandma's+chic+knitting+club"
    ```

## Configuration

Edit `config.py` to change model checkpoints, batch sizes, image folder paths, and index file locations.

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [FastAPI](https://fastapi.tiangolo.com/)
- [Fashion-CLIP](https://huggingface.co/patrickjohncyh/fashion-clip)
