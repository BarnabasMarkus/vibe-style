import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import numpy as np
import torch
import faiss
from PIL import Image
from tqdm import tqdm
import utils
import config

def get_image_features(image_paths,
                       processor,
                       model,
                       batch_size,
                       device):
    # Initialize empty list to store embeddings
    embeddings = []
    model.eval()
    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i + batch_size]
        images = [Image.open(path).convert('RGB') for path in batch_paths]
        inputs = processor(images=images, return_tensors="pt", padding=True)
        # Move all tensors to device and ensure float dtype
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor):
                inputs[k] = v.float().to(device)
        with torch.no_grad():
            image_features = model.get_image_features(**inputs)
            # Check for NaNs immediately after model output
            if torch.isnan(image_features).any():
                print("NaN detected in image_features after model forward!")
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        embeddings.extend(image_features.cpu().numpy())
    # Convert list to numpy array
    embeddings = np.array(embeddings)
    return embeddings

if __name__ == "__main__":

    # Set device
    device = utils.get_device()

    # Load the model and processor
    model, processor = utils.load_model(model_ckpt=config.model_ckpt, device=device)
    print(f"Loaded model {config.model_ckpt} on {device}")

    # Load image paths
    image_paths = utils.load_image_paths(config.image_folder)
    print(f"Found {len(image_paths)} images in {config.image_folder}")

    # Get image features
    embeddings = get_image_features(image_paths[:config.image_processing_limit],
                                    processor=processor,
                                    model=model,
                                    batch_size=config.batch_size,
                                    device=device)

    # Create FAISS index
    dimension = embeddings.shape[1]  # CLIP embedding dimension
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    # Save the index and paths for later use
    faiss.write_index(index, config.index_path)
    torch.save(image_paths, config.image_paths)

    print(f"Created and saved index with {len(embeddings)} embeddings")
