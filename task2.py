import numpy as np
import tensorflow as tf
import os
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing import image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('C:/Users/Student/.keras/datasets/Alzheimers Dataset/alzheimers_cnn_model.h5', compile=False)

# Get class names from your model
class_names = ['Mild Demented', 'Moderate Demented', 'Non Demented', 'Very Mild Demented']

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the file temporarily
    file_path = os.path.join('C:/Users/Student/.keras/datasets/Alzheimers Dataset/', file.filename)
    file.save(file_path)

    # Load and preprocess the image
    img = image.load_img(file_path, target_size=(160, 160))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0) / 255.0  # Normalize the image

    # Predict the class
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    # Get the predicted class and confidence
    predicted_class = class_names[np.argmax(score)]
    confidence = 100 * np.max(score)

    # Clean up the temporary file
    os.remove(file_path)

    return jsonify({
        'class': predicted_class,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=True)

#Example cURL command: curl -X POST -F "file=@C:/Users/Student/.keras/datasets/Alzheimers Dataset/test_image.jpg" http://127.0.0.1:5000/predict

