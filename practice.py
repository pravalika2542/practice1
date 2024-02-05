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

# Function to retrieve most similar products
def retrieve_most_similar_products(given_img, cos_similarities_df, nb_closest_images):
    st.write("-----------------------------------------------------------------------")
    st.write("Original Product:")

    original = load_img(given_img, target_size=(imgs_model_width, imgs_model_height))
    st.image(original, caption="Original Image", use_column_width=True)
    st.write("-----------------------------------------------------------------------")
    st.write("Most Similar Products:")

    closest_imgs = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1].index
    closest_imgs_scores = cos_similarities_df[given_img].sort_values(ascending=False)[1:nb_closest_images + 1]

    for i in range(len(closest_imgs)):
        st.image(load_img(closest_imgs[i], target_size=(imgs_model_width, imgs_model_height)),
                 caption=f"Similarity Score: {closest_imgs_scores[i]:.4f}", use_column_width=True)

# Streamlit App
st.title("Image Similarity Retrieval")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)

    # Load pre-trained VGG16 model
    vgg_model = vgg16.VGG16(weights='imagenet')
    feat_extractor = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer("fc2").output)

    # Load images and compute similarities
    files = ["C:\Users\dell\Documents\ML Project/" + x for x in os.listdir("C:\Users\dell\Documents\ML Project/") if "png" in x]
    files.append(uploaded_file)
    imported_images = []

    for f in files:
        original = load_img(f, target_size=(imgs_model_width, imgs_model_height))
        numpy_image = img_to_array(original)
        image_batch = np.expand_dims(numpy_image, axis=0)
        imported_images.append(image_batch)

    images = np.vstack(imported_images)
    processed_imgs = preprocess_input(images.copy())
    imgs_features = feat_extractor.predict(processed_imgs)
    cos_similarities = cosine_similarity(imgs_features)
    cos_similarities_df = pd.DataFrame(cos_similarities, columns=files, index=files)

    # Retrieve and display most similar products
    retrieve_most_similar_products(uploaded_file, cos_similarities_df, nb_closest_images)
