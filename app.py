# app.py

import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image

# Load model and processor
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("nateraw/plant-disease-classification")
    model = AutoModelForImageClassification.from_pretrained("nateraw/plant-disease-classification")
    return processor, model

processor, model = load_model()

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
    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    predicted_class_idx = outputs.logits.argmax(-1).item()
    disease_name = model.config.id2label[predicted_class_idx]
    
    st.success(f"**Predicted Disease:** {disease_name}")
    treatment = treatment_dict.get(disease_name, "No treatment advice available for this disease.")
    st.info(f"**Treatment Advice:** {treatment}")
