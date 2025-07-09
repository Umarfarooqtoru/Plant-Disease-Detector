# app.py

import streamlit as st
from transformers import pipeline
from PIL import Image

# Load the pre-trained plant disease classification pipeline
@st.cache_resource
def load_pipeline():
    classifier = pipeline("image-classification", model="nateraw/plant-disease-classification")
    return classifier

classifier = load_pipeline()

# Streamlit UI
st.title("🌿 Plant Disease Detector (Hugging Face API-free Pipeline)")
st.write("Upload a leaf photo to detect disease using a pre-trained model from Hugging Face.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert PIL image to RGB and predict
    result = classifier(image)

    # Display top prediction
    disease_name = result[0]['label']
    confidence = result[0]['score']

    st.success(f"**Predicted Disease:** {disease_name} ({confidence:.2f} confidence)")

    # Example treatment advice
    treatment_dict = {
        "Tomato___Bacterial_spot": "Remove affected leaves. Apply copper-based bactericides.",
        "Tomato___Early_blight": "Use fungicides with chlorothalonil. Rotate crops yearly.",
        "Tomato___Late_blight": "Apply fungicides containing chlorothalonil or copper-based sprays.",
        "Tomato___Leaf_Mold": "Ensure good air circulation. Use fungicides as needed.",
        "Tomato___Septoria_leaf_spot": "Remove infected leaves. Apply fungicides.",
        "Tomato___Spider_mites Two-spotted_spider_mite": "Use insecticidal soap or miticides.",
        "Tomato___Target_Spot": "Apply appropriate fungicides and practice crop rotation.",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Control whitefly vectors. Remove infected plants.",
        "Tomato___Tomato_mosaic_virus": "Remove infected plants. Disinfect tools.",
        "Tomato___healthy": "No disease detected. Maintain good crop practices."
    }
    advice = treatment_dict.get(disease_name, "No treatment advice available.")
    st.info(f"**Treatment Advice:** {advice}")
