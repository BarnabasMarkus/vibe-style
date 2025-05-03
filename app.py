import os
import torch
import faiss
import streamlit as st
import utils

# Configure PyTorch and threading before Streamlit initialization
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
torch.set_num_threads(1)

# Initialize Streamlit UI
st.set_page_config(page_title="VibeStyle", page_icon=":guardsman:", layout="wide")

@st.cache_resource
def load_cached_resources():
    return utils.load_rescources()

device, model, processor, index, image_paths = load_cached_resources()

st.title("VibeStyle")
st.write("Find your perfect outfit with VibeStyle! Just enter a text description of the outfit you're looking for, and we'll show you the best matches from our collection.")
st.write("### Search for your outfit")

# Example vibe search terms (expanded to 8)
example_terms = [
    "casual summer dress",
    "streetwear with sneakers",
    "elegant evening gown",
    "business casual outfit",
    "cozy winter sweater",
    "boho chic festival look",
    "minimalist monochrome",
    "retro 90s street style"
]

# Layout: Example buttons near the search field
st.write("Try an example vibe:")
cols = st.columns(len(example_terms))
for i, term in enumerate(example_terms):
    if cols[i].button(term):
        st.session_state["query"] = term
        st.session_state["run_search"] = True  # trigger search

# Put search field, number input, and button in one line
if "query" not in st.session_state:
    st.session_state["query"] = ""
input_col, num_col, btn_col = st.columns([4, 2, 1])
with input_col:
    query = st.text_input(
        "Enter a text description of the outfit you're looking for:",
        value=st.session_state["query"],
        key="query"
    )
with num_col:
    top_k = st.number_input("Number of top matches to return:", min_value=1, max_value=100, value=5)
with btn_col:
    st.markdown("<br>", unsafe_allow_html=True)  # Add vertical space for alignment
    search_clicked = st.button("Search", key="search_btn", use_container_width=True)

# Search button or auto-search if example pressed
run_search = st.session_state.pop("run_search", False) if "run_search" in st.session_state else False

if search_clicked or run_search:
    if query:
        with st.spinner("Searching..."):
            results = utils.search_images_by_vibe(query=query,
                                                  processor=processor,
                                                  model=model,
                                                  index=index,
                                                  image_paths=image_paths,
                                                  top_k=top_k,
                                                  device=device)
            st.write("### Search Results")
            cols = st.columns(min(len(results[0]), 6))
            for i, (img_path, score) in enumerate(zip(results[0], results[1])):
                col = cols[i % 6]
                with col:
                    st.image(img_path, caption=f"Score: {score:.4f}\n{img_path}", use_container_width=True)
                if (i + 1) % 6 == 0 and i + 1 < len(results[0]):
                    cols = st.columns(min(len(results[0]) - i - 1, 6))
    else:
        st.error("Please enter a query to search.")
