# VibeStyle

VibeStyle is a text-to-image search application that helps users find their perfect outfit by entering a text description. It leverages OpenAI's CLIP model for embedding generation and FAISS for efficient similarity search.

## Features
- **Text-to-Image Search**: Enter a text query to find matching images from a dataset.
- **Streamlit Web App**: User-friendly interface for searching outfits.
- **REST API**: Expose search functionality via FastAPI.
- **Batch Processing**: Efficiently process large datasets to build FAISS indices.

## Dataset
The project uses the [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset). Download the dataset and place it in the `fashion-dataset/images` folder.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd vibe-style
   ```

2. Install dependencies:
   ```bash
   pip install -r requriments.txt
   ```

3. Download the dataset and place it in the `fashion-dataset/images` folder.

4. Set up environment variables:
   - `IMAGE_FOLDER`: Path to the folder containing images (default: `fashion-dataset/images`).
   - `INDEX_PATH`: Path to save/load the FAISS index (default: `image_index.bin`).
   - `API_LOG_FILE`: Path to save API logs (default: `api_calls.log`).
   - `PORT`: Port for the FastAPI server (default: `8000`).

## Usage

### 1. Build the FAISS Index
Run the `build_index.py` script to process images and build the FAISS index:
```bash
python build_index.py --image_folder fashion-dataset/images --batch_size 100 --index_path image_index.bin
```

### 2. Start the Streamlit App
Launch the web app to search for outfits:
```bash
streamlit run app.py
```

### 3. Start the REST API
Run the FastAPI server to expose search functionality:
```bash
python api.py
```

### 4. Search for Outfits
- **Web App**: Enter a text query in the Streamlit app to find matching outfits.
- **API**: Use the `/search` endpoint to query the API:
  ```bash
  curl "http://localhost:8000/search?query=bohemian+summer+vibes&top_k=5"
  ```

#### Example Queries
Here are some creative examples of text queries you can try:
- `"bohemian summer vibes"`
- `"elegant evening gown with sequins"`
- `"casual streetwear with a pop of neon"`
- `"vintage floral dress for a garden party"`
- `"cozy winter outfit with a knitted sweater"`
- `"bold and edgy leather jacket look"`
- `"minimalist monochrome outfit"`

## Project Structure
- `search.py`: Core search logic using CLIP and FAISS.
- `build_index.py`: Script to process images and build the FAISS index.
- `app.py`: Streamlit web app for user interaction.
- `api.py`: FastAPI server for RESTful search.
- `requriments.txt`: List of dependencies.
- `.gitignore`: Files and folders to ignore in version control.

## License
This project is licensed under the MIT License.

## Acknowledgments
- [OpenAI CLIP](https://github.com/openai/CLIP)
- [FAISS](https://github.com/facebookresearch/faiss)
- [Streamlit](https://streamlit.io/)
- [Fashion Product Images Dataset](https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-dataset)