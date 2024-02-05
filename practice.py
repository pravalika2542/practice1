import streamlit as st
st.title("Image Similarity Retrieval")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
