"""Create image embeddings and a FAISS cosine-similarity index from local images."""
from pathlib import Path

import numpy as np
import pandas as pd
import faiss
from PIL import Image
from tqdm import tqdm

from src.similarity import create_embedding_model, embed_image

ROOT = Path(__file__).resolve().parent
IMAGE_DIR = ROOT / "data" / "raw" / "images"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(
        p for p in IMAGE_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(
            f"No images found in {IMAGE_DIR}. Add .jpg/.jpeg/.png/.webp files and retry."
        )

    model, preprocess = create_embedding_model()
    rows, vectors = [], []

    for image_path in tqdm(image_paths, desc="Embedding images"):
        try:
            with Image.open(image_path) as image:
                vector = embed_image(image.convert("RGB"), model, preprocess)
            vectors.append(vector)
            rows.append({
                "product_id": image_path.stem,
                "image_path": image_path.relative_to(ROOT).as_posix(),
            })
        except Exception as exc:
            print(f"Skipping {image_path}: {exc}")

    if not vectors:
        raise SystemExit("No images could be processed. Check that the image files are valid.")

    matrix = np.vstack(vectors).astype("float32")
    # Embeddings are L2-normalized, so inner product is cosine similarity.
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    faiss.write_index(index, str(ARTIFACT_DIR / "image_embeddings.faiss"))
    np.save(ARTIFACT_DIR / "image_embeddings.npy", matrix)
    pd.DataFrame(rows).to_csv(PROCESSED_DIR / "catalog.csv", index=False)

    print(f"Indexed {len(rows)} images.")
    print(f"FAISS index: {ARTIFACT_DIR / 'image_embeddings.faiss'}")
    print(f"Catalog: {PROCESSED_DIR / 'catalog.csv'}")

if __name__ == "__main__":
    main()
