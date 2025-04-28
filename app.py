import os
import streamlit as st
import search

# Load configurations from environment variables
image_folder = os.getenv("IMAGE_FOLDER", "fashion-dataset/images")
index_path = os.getenv("INDEX_PATH", "image_index.bin")

model, processor = search.load_clip_model()
image_paths = search.load_image_paths(image_folder)
index = search.load_faiss_index(index_path)

st.set_page_config(page_title="VibeStyle", page_icon=":guardsman:", layout="wide")
st.title("VibeStyle")
st.write("Find your perfect outfit with VibeStyle! Just enter a text description of the outfit you're looking for, and we'll show you the best matches from our collection.")
st.write("### Search for your outfit")

query = st.text_input("Enter a text description of the outfit you're looking for:")
top_k = st.number_input("Number of top matches to return:", min_value=1, max_value=100, value=5)

if st.button("Search"):
    if query:
        with st.spinner("Searching..."):
            results = search.get_top_matches(query, model, processor, index, image_paths, top_k)
            st.write("### Search Results")
            for i in range(0, len(results), 6):  # Process results in chunks of 6
                cols = st.columns(min(6, len(results) - i))  # Create up to 6 columns
                for col, result in zip(cols, results[i:i + 6]):
                    with col:
                        st.image(result["image_path"],
                            caption=f"Score: {result['score']:.4f}",
                            use_container_width=True)
    else:
        st.error("Please enter a query to search.")
