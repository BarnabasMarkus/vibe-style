import os
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
from tqdm import tqdm


# Load CLIP model and processor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)


def get_image_paths(image_folder):
    """
    Get paths of all images in the specified folder.
    """
    if not os.path.exists(image_folder):
        raise FileNotFoundError(f"Image folder '{image_folder}' does not exist.")

    image_paths = [
        os.path.join(image_folder, fname)
        for fname in os.listdir(image_folder)
        if fname.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if not image_paths:
        raise ValueError(f"No valid image files found in '{image_folder}'.")

    print(f"Found {len(image_paths)} images in '{image_folder}'.")
    return image_paths


def process_images_in_batches(image_paths, batch_size=10):
    """
    Process images in batches, compute embeddings, and add them to the FAISS index.
    """
    dimension = None
    index = None

    # Initialize tqdm progress bar
    total_batches = (len(image_paths) + batch_size - 1) // batch_size  # Calculate total number of batches
    progress_bar = tqdm(total=total_batches, desc="Processing Batches", unit="batch")

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        images = []

        # Load and preprocess images
        for image_path in batch_paths:
            try:
                with Image.open(image_path) as img:
                    images.append(img.convert("RGB"))
            except Exception as e:
                print(f"Warning: Failed to process image '{image_path}'. Error: {e}")
                continue

        if not images:
            print(f"Skipping batch {i // batch_size + 1} due to no valid images.")
            continue

        # Process images with CLIP
        inputs = processor(images=images, return_tensors="pt", padding=True)
        with torch.no_grad():
            image_embeddings = model.get_image_features(**inputs)

        # Normalize embeddings
        image_embeddings = image_embeddings / image_embeddings.norm(dim=1, keepdim=True)

        # Initialize FAISS index if not already done
        if index is None:
            dimension = image_embeddings.shape[1]
            index = faiss.IndexFlatL2(dimension)

        # Add embeddings to the index
        index.add(image_embeddings.numpy())
        # print(f"Processed batch {i // batch_size + 1}, added {len(batch_paths)} embeddings to the index.")

        # Update progress bar
        progress_bar.update(1)

    if index is None:
        raise RuntimeError("No embeddings were added to the FAISS index. Check your image folder and processing logic.")

    return index


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build FAISS index for image embeddings.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the folder containing images.")
    parser.add_argument("--batch_size", type=int, default=1000, help="Batch size for processing images.")
    parser.add_argument("--index_path", type=str, default="image_index.bin", help="Path to save the FAISS index.")
    args = parser.parse_args()

    try:
        print(f"Processing images from '{args.image_folder}' in batches of {args.batch_size}.")
        image_paths = get_image_paths(args.image_folder)
        index = process_images_in_batches(image_paths, batch_size=args.batch_size)
        faiss.write_index(index, args.index_path)
        print(f"FAISS index with {index.ntotal} embeddings saved to '{args.index_path}'.")
    except Exception as e:
        print(f"Error: {e}")
