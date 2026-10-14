from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image

from src.similarity import find_similar_products, load_catalog

ROOT = Path(__file__).resolve().parent
CATALOG_PATH = ROOT / "data" / "processed" / "catalog.csv"
INDEX_PATH = ROOT / "artifacts" / "image_embeddings.faiss"
EMBEDDINGS_PATH = ROOT / "artifacts" / "image_embeddings.npy"

st.set_page_config(page_title="Visual Product Finder", page_icon="🛍️", layout="wide")
st.title("🛍️ Visual Product Similarity")
st.write("Upload a product image to find visually similar products from your catalog.")

@st.cache_data
def get_catalog():
    if CATALOG_PATH.exists():
        return load_catalog(CATALOG_PATH)
    return pd.DataFrame(columns=["product_id", "image_path"])

@st.cache_resource
def get_search_assets():
    if INDEX_PATH.exists() and EMBEDDINGS_PATH.exists():
        return True
    return False

catalog = get_catalog()
assets_ready = get_search_assets()

if catalog.empty or not assets_ready:
    st.info(
        "Your product catalog/index is not ready yet. Add images to "
        "`data/raw/images/`, then run `python build_index.py`. "
        "See README.md for the expected folder structure."
    )

uploaded = st.file_uploader("Choose a query product image", type=["jpg", "jpeg", "png", "webp"])
top_k = st.slider("Number of recommendations", min_value=1, max_value=12, value=5)

if uploaded:
    query_image = Image.open(uploaded).convert("RGB")
    st.subheader("Your query image")
    st.image(query_image, width=280)

    if st.button("Find similar products", type="primary"):
        if catalog.empty or not assets_ready:
            st.error("Build the image index first. Follow the setup steps in README.md.")
        else:
            try:
                results = find_similar_products(
                    query_image=query_image,
                    catalog=catalog,
                    index_path=INDEX_PATH,
                    embeddings_path=EMBEDDINGS_PATH,
                    top_k=top_k,
                )
                if results.empty:
                    st.warning("No matching products were found.")
                else:
                    st.subheader("Most visually similar products")
                    columns = st.columns(min(3, len(results)))
                    for i, (_, row) in enumerate(results.iterrows()):
                        with columns[i % len(columns)]:
                            image_path = ROOT / str(row["image_path"])
                            if image_path.exists():
                                st.image(str(image_path), use_container_width=True)
                            st.markdown(f"**Product:** {row['product_id']}")
                            st.caption(f"Cosine similarity: {row['similarity']:.3f}")
            except Exception as exc:
                st.error(f"Search failed: {exc}")
