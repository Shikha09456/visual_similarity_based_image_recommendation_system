"""Image embedding and nearest-neighbor search utilities."""
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import models

ROOT = Path(__file__).resolve().parents[1]

def create_embedding_model():
    """Load pretrained ResNet-50 and remove its final classification layer."""
    weights = models.ResNet50_Weights.DEFAULT
    full_model = models.resnet50(weights=weights)
    model = torch.nn.Sequential(*list(full_model.children())[:-1])
    model.eval()
    return model, weights.transforms()

def embed_image(image: Image.Image, model, preprocess) -> np.ndarray:
    """Convert an image to a normalized 2048-dimensional ResNet-50 embedding."""
    tensor = preprocess(image).unsqueeze(0)
    with torch.no_grad():
        features = model(tensor).flatten(start_dim=1)
    vector = features.cpu().numpy().astype("float32")
    faiss.normalize_L2(vector)
    return vector[0]

def load_catalog(path: Path) -> pd.DataFrame:
    """Load and validate the generated product catalog."""
    catalog = pd.read_csv(path)
    required = {"product_id", "image_path"}
    missing = required.difference(catalog.columns)
    if missing:
        raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
    return catalog

def find_similar_products(
    query_image: Image.Image,
    catalog: pd.DataFrame,
    index_path: Path,
    embeddings_path: Path,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return the top-k visually similar catalog images by cosine similarity."""
    if catalog.empty:
        return pd.DataFrame()
    if not index_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError("Index files missing. Run `python build_index.py` first.")

    model, preprocess = create_embedding_model()
    query_vector = embed_image(query_image.convert("RGB"), model, preprocess).reshape(1, -1)
    index = faiss.read_index(str(index_path))

    k = min(top_k + 1, len(catalog))
    scores, indices = index.search(query_vector, k)

    results = []
    query_resolved = None
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        row = catalog.iloc[int(idx)]
        # Skip an exact same catalog file if the uploaded image is itself in the catalog
        results.append({
            "product_id": str(row["product_id"]),
            "image_path": str(row["image_path"]),
            "similarity": float(score),
        })

    return pd.DataFrame(results).head(top_k)
