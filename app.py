import streamlit as st
import requests

# Streamlit UI
st.title("🌿 Plant Disease Detector (Hugging Face API)")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prepare API call
    api_url = "https://api-inference.huggingface.co/models/akhaliq/plant-disease-model"
    headers = {"Authorization": f"Bearer {st.secrets['HF_TOKEN']}"}

    # Read image bytes
    image_bytes = uploaded_file.read()

    # Call Hugging Face Inference API
    response = requests.post(api_url, headers=headers, data=image_bytes)

    if response.status_code == 200:
        result = response.json()
        if len(result) > 0:
            disease_name = result[0]['label']
            confidence = result[0]['score']
            st.success(f"**Predicted Disease:** {disease_name} ({confidence:.2f} confidence)")
        else:
            st.warning("Could not detect any disease.")
    else:
        st.error(f"API Error: {response.status_code} {response.text}")
