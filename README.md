Leaf Disease Classification using PyTorch & ResNet-50
This project is a deep learning-based image classification system built to identify multiple diseases in plant leaves from images. It leverages transfer learning with a pre-trained ResNet-50 model and provides an easy-to-use web interface built with Streamlit.

<!-- It's a good idea to add a screenshot of your app here! -->

Features
High Accuracy: Utilizes a pre-trained ResNet-50 model for state-of-the-art performance.

Multi-Class Classification: Capable of identifying several leaf conditions, including:

Bacterial

Fungal

Viral

Healthy

Interactive Web UI: A simple Streamlit interface allows users to upload an image and get an instant prediction.

GPU Accelerated: The training script is optimized to use a CUDA-enabled GPU if available.

Technologies Used
Backend: Python

Deep Learning Framework: PyTorch

Web Framework: Streamlit

Core Libraries: Torchvision, Pillow, NumPy, Tqdm

Setup and Installation
Follow these steps to set up the project on your local machine.

1. Clone the Repository
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name

2. Set Up a Virtual Environment (Recommended)
# For Windows
python -m venv venv
venv\Scripts\activate

# For macOS/Linux
python3 -m venv venv
source venv/bin/activate

3. Install Dependencies
Install all the required Python libraries using the requirements.txt file.

pip install -r requirements.txt

4. Download the Dataset
This project uses the PlantVillage Dataset. You can download it from Kaggle.

Dataset Link: New Plant Diseases Dataset on Kaggle

After downloading, unzip the file and place the contents so that your Dataset folder is in the root of the project directory.

5. Download the Pre-trained Model
The trained model is not stored in this repository. You can download the best_leaf_disease_model.pth file from the link below and place it inside the saved_models/ directory.

Model Download Link: [<-- PASTE YOUR GOOGLE DRIVE / DROPBOX LINK HERE -->]

How to Run the Project
To Train Your Own Model (Optional)
If you wish to train the model from scratch on your own dataset, run the training script:

python train.py

To Run the Streamlit Application
Once you have the trained .pth model in the saved_models/ folder, launch the web application:

streamlit run app.py

Your web browser will open with the application, ready for you to upload leaf images.

Project Structure
├── Dataset/                # Contains Train and Test image folders
├── saved_models/           # Stores the trained .pth model file
├── .gitignore              # Specifies files for Git to ignore
├── app.py                  # The Streamlit web application script
├── README.md               # This file
├── requirements.txt        # Project dependencies
└── train.py                # The all-in-one model training script
