# app.py

import streamlit as st
from transformers import pipeline
from PIL import Image

# Load pipeline
@st.cache_resource
def load_pipeline():
    classifier = pipeline("image-classification", model="nateraw/plant-disease-classification")
    return classifier

classifier = load_pipeline()

# Treatment advice dictionary (expand as needed)
treatment_dict = {
    "Tomato___Late_blight": "Use fungicides containing chlorothalonil or copper-based sprays.",
    "Tomato___Bacterial_spot": "Remove infected leaves. Apply copper-based bactericides.",
    "Tomato___Leaf_Mold": "Improve air circulation. Use chlorothalonil fungicide.",
    # Add more diseases with treatments here
}

# Streamlit UI
st.title("🌿 Plant Disease Detector")
st.write("Upload a leaf photo to detect disease and get treatment advice.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Leaf Image", use_column_width=True)
    
    # Predict disease
    result = classifier(image)
    if len(result) > 0:
        disease_name = result[0]['label']
        confidence = result[0]['score']
        
        st.success(f"**Predicted Disease:** {disease_name} ({confidence:.2f} confidence)")
        treatment = treatment_dict.get(disease_name, "No treatment advice available for this disease.")
        st.info(f"**Treatment Advice:** {treatment}")
    else:
        st.warning("Could not detect any disease.")
