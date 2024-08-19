#Overview
This repository contains the code and resources for a project aimed at developing an AI application to classify Alzheimer's disease stages using MRI images. The project involves training a Convolutional Neural Network (CNN), deploying the model as a web service, and fine-tuning a pre-trained model for improved performance.

Project Structure
data/: Contains the dataset used for training and validation.
models/: Contains the trained model files, including the fine-tuned MobileNetV2 model.
notebooks/: Jupyter notebooks with code for model training, evaluation, and visualization.
web_service/: Contains the Flask application code for deploying the model as a web service.
requirements.txt: Lists the required Python packages for the project.
Installation
Clone the Repository

bash
Copy code
git clone https://github.com/yourusername/alzheimers-disease-classification.git
cd alzheimers-disease-classification
Install Dependencies

Create a virtual environment and install the required packages:

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
pip install -r requirements.txt
Dataset
The dataset used for this project consists of MRI images categorized into four stages of Alzheimer's disease:

Mild Demented
Moderate Demented
Non Demented
Very Mild Demented
Download the dataset and organize it into the data/ directory with train and validation subdirectories.

Model Training
To train the CNN model, run the following script:

bash
Copy code
python train_model.py
This script will load the dataset, build and train the CNN model, and save the trained model to the models/ directory.

Model Deployment
The Flask web service allows users to upload MRI images and receive classification results. To start the web service, navigate to the web_service/ directory and run:

bash
Copy code
python app.py
The service will be available at http://localhost:5000/. Use Postman or a similar tool to test the service by uploading an image and receiving the classification result.

Fine-Tuning
To fine-tune the MobileNetV2 model, run:

bash
Copy code
python fine_tune_model.py
This script will load the pre-trained MobileNetV2 model, add custom layers, and train it on the dataset. The fine-tuned model will be saved to the models/ directory.

Results
After training and fine-tuning, you can visualize the training and validation accuracy and loss using:

bash
Copy code
python plot_results.py
This script generates plots showing the performance of the trained and fine-tuned models.

Requirements
TensorFlow
Flask
Matplotlib
NumPy
OpenCV
To install the required packages, use:

bash
Copy code
pip install -r requirements.txt
References
Kaggle - Dataset source
Roboflow - Dataset source
iMerit - Dataset source
License
This project is licensed under the MIT License - see the LICENSE file for details.
