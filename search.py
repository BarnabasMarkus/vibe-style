import torch
import faiss
import os
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

# Set environment variables
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for OpenMP runtime error

def load_clip_model():
    """Load the CLIP model and processor."""
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def load_image_paths(image_folder: str):
    """Load image paths from the specified folder."""
    image_paths = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.endswith(".jpg")
    ]
    print(f"Found {len(image_paths)} images.")
    return image_paths

def load_faiss_index(index_path: str):
    """Load the FAISS index from disk."""
    index = faiss.read_index(index_path)
    print(f"FAISS index loaded from disk. Size: {index.ntotal} embeddings.")
    return index

def get_top_matches(text_query: str, model, processor, index, image_paths, top_k: int = 5):
    """Search the FAISS index for the top matches to the text query."""
    inputs = processor(text=[text_query], return_tensors="pt", padding=True)
    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        text_features /= text_features.norm(p=2, dim=-1, keepdim=True)
    query_vec = text_features.cpu().numpy()

    # Search
    scores, indices = index.search(query_vec, top_k)
    results = [
        {"image_path": image_paths[i], "score": float(scores[0][rank])}
        for rank, i in enumerate(indices[0])
    ]
    return results
