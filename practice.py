import streamlit as st
st.set_page_config(page_title='Similarity')
st.title("Image Similarity Retrieval")
uploaded_file = st.file_uploader("Choose an image...", type=["png"])
