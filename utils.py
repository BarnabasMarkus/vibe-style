import os
import numpy as np
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
import config

def get_device():
    """
    Get the device to be used for PyTorch operations.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device

def load_model(model_ckpt:str, device:str):
    """
    Load the CLIP model and processor.
    """
    model = CLIPModel.from_pretrained(model_ckpt).float().to(device)
    processor = CLIPProcessor.from_pretrained(model_ckpt, use_fast=False)
    return model, processor

def load_image_paths(image_folder:str):
    """
    Load image paths from the specified folder.
    """
    image_paths = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return image_paths

def search_images_by_vibe(query,
                          processor,
                          model,
                          index,
                          image_paths,
                          top_k,
                          device):
    """
    Search for images in the index that match the text query.
    """
    # Process the text query
    inputs = processor(text=query, return_tensors="pt", padding=True)

    # Move all tensors to the same device as the model
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.to(device)

    with torch.no_grad():
        text_features = model.get_text_features(**inputs)
        # Normalize features
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Search the index
    distances, indices = index.search(text_features.cpu().numpy(), top_k)

    # Retrieve the image paths
    results = [image_paths[i] for i in indices[0]]

    return results, distances[0]

def load_rescources(config):
    """
    Load resources such as model, processor, index, and image paths.
    """
    # Load model and processor
    device = get_device()
    model, processor = load_model(model_ckpt=config.model_ckpt, device=device)

    # Load index and image paths
    index = faiss.read_index(config.index_path)
    image_paths = torch.load(config.image_paths_file)

    return device, model, processor, index, image_paths
