import torch

# -- Project Configuration --
# Automatically select GPU if available, otherwise fall back to CPU
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- IMPORTANT: Set paths to your dataset ---
# These paths are relative to the main project folder (leaf-disease-project)
TRAIN_DIR = "Dataset/Train"
VAL_DIR = "Dataset/Test"

# Path to save the best trained model
MODEL_SAVE_PATH = "saved_models/best_leaf_disease_model.pth"


# -- Model Hyperparameters --
# --- Set this to the number of folders (classes) in your Train directory ---
NUM_CLASSES = 4 # (Bacterial, Fungal, healthy, viral)
LEARNING_RATE = 0.001
BATCH_SIZE = 32 # Number of images to process at once
NUM_EPOCHS = 20 # Number of times to loop over the entire training dataset
IMG_SIZE = 224 # Image size will be resized to 224x224 pixels


# -- Data Augmentation Configuration --
# These transformations are applied to the training data to increase variety
# and make the model more robust.
TRAIN_TRANSFORMS = {
    'horizontal_flip_p': 0.5, # Probability of flipping an image horizontally
    'rotation_degrees': 10,   # Max angle for random rotations
    'normalize_mean': [0.485, 0.456, 0.406], # Standard values for pre-trained models
    'normalize_std': [0.229, 0.224, 0.225],  # Standard values for pre-trained models
}

# For validation, we only apply the necessary resizing and normalization
# to test the model on data that looks like real-world examples.
VAL_TRANSFORMS = {
    'normalize_mean': [0.485, 0.456, 0.406],
    'normalize_std': [0.229, 0.224, 0.225],
}

