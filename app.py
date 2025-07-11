# app.py

import streamlit as st
from transformers import pipeline
from PIL import Image

# ✅ Load the image classification pipeline
@st.cache_resource
def load_pipeline():
    classifier = pipeline("image-classification", model="microsoft/resnet-50")
    return classifier

classifier = load_pipeline()

# ✅ Streamlit UI
st.title("🌿 Plant Disease Detector Demo")
st.write("Upload a leaf photo to detect (this demo uses ResNet50 trained on ImageNet).")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Perform prediction
    results = classifier(image)

    # Display top predictions
    st.subheader("🔎 Top Predictions:")
    for result in results[:5]:
        label = result['label']
        score = result['score']
        st.write(f"**{label}**: {score:.2f}")

    st.info("⚠️ Note: This demo uses a general object classification model, not plant disease-specific.")
