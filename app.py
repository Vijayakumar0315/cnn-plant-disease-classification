
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import io
import json
from flask import Flask, request, jsonify, render_template

# --- DICTIONARY MAPPING: CATEGORY TO SUGGESTIONS AND EMOJIS ---
# FIX: Made all keys lowercase for consistent and robust matching.
CATEGORY_INFO = {
    'fungal': {
        'emoji': '🍄',
        'suggestion': 'Consider applying a suitable fungicide. Improve air circulation and avoid wetting the leaves when watering.'
    },
    'bacterial': {
        'emoji': '🦠',
        'suggestion': 'Use a copper-based bactericide. Prune and destroy infected plant parts. Avoid overhead watering.'
    },
    'viral': {
        'emoji': '🧬',
        'suggestion': 'There is no cure for most plant viruses. Remove and destroy the infected plant to prevent it from spreading. Control insect vectors like aphids.'
    },
    'healthy': {
        'emoji': '✅',
        'suggestion': 'The plant appears healthy. Continue with good care practices, including proper watering and fertilization.'
    }
}

# --- THIS IS THE FIX ---
# This dictionary was missing, causing the server to crash.
UNKNOWN_INFO = {
    'emoji': '❓',
    'suggestion': 'The model could not identify this category. Please try a clearer image or ensure the model is trained on this class.'
}


# --- MODEL AND CLASS NAME LOADING ---
model_path = os.path.join("saved_models", "best_leaf_disease_model.pth")
TRAIN_DIR = "Dataset/Train" # Path to the training data directory

if not os.path.exists(model_path):
    print(f"---")
    print(f"ERROR: Model file not found at '{model_path}'")
    print("Please run 'python train.py' to train and save the model first.")
    print(f"---")
    exit()

# --- DYNAMIC CLASS DETECTION ---
# Automatically detect class names from the training folder structure.
# This ensures the app is always in sync with the trained model.
try:
    # Added filter to ignore hidden/system folders
    CLASS_NAMES = sorted(
        [d.name for d in os.scandir(TRAIN_DIR) 
         if d.is_dir() and not d.name.startswith('.') and not d.name.startswith('_')]
    )
    num_classes = len(CLASS_NAMES)
    if num_classes == 0:
        raise FileNotFoundError # Trigger the error below
    print(f"✅ Found {num_classes} classes from '{TRAIN_DIR}': {CLASS_NAMES}")
except FileNotFoundError:
    print(f"---")
    print(f"ERROR: Could not find training directory at '{TRAIN_DIR}'")
    print("Please make sure your 'Dataset/Train' folder exists and is not empty.")
    print(f"---")
    exit()
# --- End of Class Detection ---


# --- MODEL DEFINITION (Must match train.py) ---
# Load the ResNet-50 structure
model = models.resnet50(weights=None) 
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)

# Load the saved weights (trained parameters)
# Using map_location='cpu' ensures it runs on any machine
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval() # Set model to evaluation mode (very important!)

print("✅ PyTorch model loaded successfully on CPU.")

# --- IMAGE TRANSFORMS (Must match train.py) ---
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# --- HELPER FUNCTION FOR PREDICTION ---
def predict_disease(image_bytes):
    """Preprocesses the image, runs prediction, and returns results."""
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Preprocess the image and add a batch dimension
        image_tensor = preprocess(image).unsqueeze(0) 
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted_index = torch.max(probabilities, 1)
        
        predicted_index = predicted_index.item()
        confidence = confidence.item() * 100
        
        # Map index to class name
        category = CLASS_NAMES[predicted_index] if predicted_index < len(CLASS_NAMES) else None
        
        return category, confidence
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, 0.0

# --- FLASK APP INITIALIZATION ---
# Flask will look for HTML files in a folder named 'templates'
app = Flask(__name__)

# --- API ENDPOINTS ---

@app.route('/')
def home():
    """Serves the main HTML page."""
    # Renders the 'index.html' file from the 'templates' folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """Handles the image upload and returns a JSON prediction."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file:
        image_bytes = file.read()
        category, confidence = predict_disease(image_bytes)
        
        if category:
            # Normalize category name to lowercase for robust matching
            category_lower = category.lower()
            # Get the info, using UNKNOWN_INFO as a safe default
            category_info = CATEGORY_INFO.get(category_lower, UNKNOWN_INFO)
            
            # Send back a complete JSON response
            return jsonify({
                'category': category, # Send the original-cased name
                'confidence': f"{confidence:.2f}%",
                'emoji': category_info['emoji'],
                'suggestion': category_info['suggestion']
            })
        else:
            return jsonify({'error': 'Could not process image'}), 500

# --- RUN THE APP ---
if __name__ == '__main__':
    # Starts the web server on http://127.0.0.1:5000
    app.run(debug=True, port=5000)

