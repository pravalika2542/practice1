import streamlit as st
from keras.applications import vgg16
from keras.preprocessing.image import load_img, img_to_array
from keras.models import Model
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import os

st.set_page_config(page_title='Similarity')
st.title("Image Similarity Retrieval")
uploaded_file = st.file_uploader("Choose an image...", type=["png"])
