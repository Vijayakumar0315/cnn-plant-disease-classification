import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import io
import os

# =============================================================================
# App Configuration
# =============================================================================
# --- MUST match the settings in train.py ---
MODEL_SAVE_PATH = "saved_models/best_leaf_disease_model.pth"
TRAIN_DIR = "Dataset/Train" # Path to the training data directory

# --- DYNAMIC CLASS DETECTION (THE FIX) ---
# Automatically detect class names from the training folder structure.
# This ensures the app is always in sync with the trained model.
try:
    CLASS_NAMES = sorted([d.name for d in os.scandir(TRAIN_DIR) if d.is_dir()])
except FileNotFoundError:
    st.error(f"Error: The directory '{TRAIN_DIR}' was not found. Please make sure your 'Dataset/Train' folder exists.")
    # Use a placeholder if the directory doesn't exist to prevent crashing
    CLASS_NAMES = []

IMG_SIZE = 224
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =============================================================================
# Model Loading
# =============================================================================
# Define the same model architecture as in train.py
def build_model(num_classes):
    model = models.resnet50(weights=None) # We don't need pre-trained weights here
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Function to load the trained model weights
@st.cache_resource # Cache the model to avoid reloading on every interaction
def load_model(path, num_classes):
    model = build_model(num_classes).to(DEVICE)
    # Load the state dictionary
    # map_location ensures the model loads correctly whether you're on CPU or GPU
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval() # Set model to evaluation mode
    return model

# =============================================================================
# Image Preprocessing and Prediction
# =============================================================================
# Define the same transformations as the validation set in train.py
def preprocess_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0) # Add a batch dimension

# Prediction function
def predict(image_tensor, model):
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)
    
    predicted_class = CLASS_NAMES[predicted_idx.item()]
    return predicted_class, confidence.item()

# =============================================================================
# Streamlit User Interface
# =============================================================================
st.set_page_config(page_title="Leaf Disease Classification", layout="centered")

st.title("🌿 Leaf Disease Classification")
st.write("Upload an image of a plant leaf, and the model will predict its condition based on the trained classes.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption='Uploaded Image.', use_column_width=True)
    
    # Show a spinner while processing
    with st.spinner('Analyzing the image...'):
        # Check if model file and classes exist
        if os.path.exists(MODEL_SAVE_PATH) and CLASS_NAMES:
            model = load_model(MODEL_SAVE_PATH, len(CLASS_NAMES))
            
            # Preprocess and predict
            image_tensor = preprocess_image(image_bytes)
            predicted_class, confidence = predict(image_tensor, model)
            
            # Display the result
            st.success("Analysis Complete!")
            st.markdown(f"### Predicted Class: **{predicted_class}**")
            st.markdown(f"### Confidence: **{confidence:.2%}**")

            # Add some context to the prediction
            if predicted_class == 'healthy':
                st.balloons()
                st.info("The model predicts the leaf is healthy. Great news!")
            else:
                st.warning(f"The model predicts a **{predicted_class}** infection. Further inspection may be required.")
        elif not CLASS_NAMES:
            # This handles the case where the TRAIN_DIR was not found
            pass
        else:
            st.error(f"Model file not found at {MODEL_SAVE_PATH}. Please train the model first by running train.py.")

else:
    st.info("Please upload an image to get a prediction.")

