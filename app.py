# app.py

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

# Class names – update as per your model
class_names = ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Healthy"]

# Function to load model
@st.cache_resource
def load_model():
    model = models.resnet50(pretrained=False)
    model.fc = torch.nn.Linear(2048, len(class_names))  # Adjust output size to number of classes
    model.load_state_dict(torch.load("plant_disease_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

# Define image transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Streamlit UI
st.title("🌿 Offline Plant Disease Detector")
st.write("Upload a leaf photo to detect disease (works fully offline).")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_tensor = transform(image).unsqueeze(0)

    # Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        _, predicted = outputs.max(1)
        disease_name = class_names[predicted.item()]
        st.success(f"**Predicted Disease:** {disease_name}")

        # Example treatment advice
        treatment_dict = {
            "Tomato___Bacterial_spot": "Remove affected leaves. Apply copper-based bactericides.",
            "Tomato___Early_blight": "Use fungicides with chlorothalonil. Rotate crops yearly.",
            "Tomato___Late_blight": "Apply fungicides containing chlorothalonil or copper-based sprays.",
            "Healthy": "No disease detected. Maintain good crop practices.",
        }
        advice = treatment_dict.get(disease_name, "No treatment advice available.")
        st.info(f"**Treatment Advice:** {advice}")
