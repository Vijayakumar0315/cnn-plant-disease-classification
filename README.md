Leaf Disease Classification using PyTorch & ResNet-50
A deep learning image classification system to identify multiple diseases in plant leaves. This project uses a pre-trained ResNet-50 model and includes a simple web interface built with Streamlit.

Features
High Accuracy: Utilizes a pre-trained ResNet-50 model for state-of-the-art performance.

Multi-Class Classification: Capable of identifying several leaf conditions:

Bacterial

Fungal

Viral

Healthy

Interactive Web UI: A simple Streamlit interface allows users to upload an image and get an instant prediction.

GPU Accelerated: The training script is optimized to use a CUDA-enabled GPU.

Technologies Used
Backend: Python

Deep Learning Framework: PyTorch

Web Framework: Streamlit

Core Libraries: Torchvision, Pillow, NumPy

Setup and Installation
Follow these steps to set up the project on your local machine.

1. Clone the Repository
git clone [https://github.com/Shabes35/Plant-Disease-Classification-PyTorch.git](https://github.com/Shabes35/Plant-Disease-Classification-PyTorch.git)
cd Plant-Disease-Classification-PyTorch

2. Set Up a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries.

pip install -r requirements.txt

4. Download the Dataset
This project is trained on the PlantVillage Dataset. You can download it from Kaggle.

Dataset Link: New Plant Diseases Dataset on Kaggle

After downloading, unzip the file and place the Dataset folder in the root of the project directory.

5. Train the Model
The pre-trained model file (.pth) is not included in this repository. You must train the model yourself by running the provided script.

How to Run the Project
1. Train the Model
Run the training script from the main project directory. The script will create a saved_models/ folder and save the best_leaf_disease_model.pth file inside it upon completion.

python train.py

2. Run the Streamlit Application
Once you have the trained .pth model, launch the web application.

streamlit run app.py

Your web browser will open with the application, ready for you to upload leaf images for classification.

Project Structure
├── Dataset/                # Contains Train and Test image folders
├── saved_models/           # Stores the trained .pth model file
├── .gitignore              # Specifies files for Git to ignore
├── app.py                  # The Streamlit web application script
├── README.md               # This file
├── requirements.txt        # Project dependencies
└── train.py                # The all-in-one model training script
