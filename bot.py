import os
import numpy as np
import tempfile
from flask import Flask, render_template, request, jsonify
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model  # Load your custom model
import time

# Initialize Flask app
app = Flask(__name__)

# Load your custom-trained model (load without compilation to avoid issues with training configuration)
model_path = './module/updated_model.h5'

# Load the model without compiling
model = load_model(model_path, compile=False)

# Recompile the model with a default optimizer, loss function, and metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define skin disease classes and treatment info (including additional diseases)
DISEASE_INFO = {
    'Melanoma': {'description': 'Melanoma is a type of skin cancer that develops from pigment-producing cells.',
                 'treatment': 'Treatment options include surgery, immunotherapy, and targeted therapy.'},
    'Basal Cell Carcinoma': {'description': 'Basal Cell Carcinoma is the most common form of skin cancer.',
                             'treatment': 'Treatment includes surgery, radiation, and topical medications.'},
    'Squamous Cell Carcinoma': {'description': 'Squamous Cell Carcinoma is a form of skin cancer that can grow quickly.',
                                'treatment': 'Treatment involves surgery, radiation, and topical therapies.'},
    'Benign Nevus': {'description': 'A benign nevus is a non-cancerous mole or growth on the skin.',
                     'treatment': 'Most benign nevi do not require treatment unless they change or become problematic.'},
    'Acne': {'description': 'Acne is a skin condition that causes pimples, blackheads, and cysts, often due to clogged pores.',
             'treatment': 'Treatment includes topical creams, oral medications, and in severe cases, laser therapy.'},
    'Psoriasis': {'description': 'Psoriasis is an autoimmune disease that causes skin cells to multiply rapidly, resulting in scaly patches.',
                  'treatment': 'Treatment options include topical creams, UV therapy, and systemic medications.'},
    'Eczema': {'description': 'Eczema is a chronic skin condition that causes itchy, inflamed skin.',
               'treatment': 'Treatment includes moisturizing, anti-inflammatory creams, and avoiding triggers.'},
    'Rosacea': {'description': 'Rosacea is a chronic condition that causes redness and visible blood vessels in the face.',
                'treatment': 'Treatment includes topical medications, oral antibiotics, and lifestyle changes.'}
}

# Create a prediction function
def predict_skin_disease(img_path):
    # Load and preprocess the image
    img = image.load_img(img_path, target_size=(224, 224))  # Adjust target size to match your model's expected input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image, adjust based on your model's training process

    # Make prediction using the custom model
    preds = model.predict(img_array)
    
    # Post-process prediction (assuming the model outputs probabilities for each class)
    predicted_class_index = np.argmax(preds, axis=1)[0]  # Get the index of the highest probability
    confidence = preds[0][predicted_class_index] * 100  # Confidence score (percentage)

    # Map the predicted class index to the skin disease name
    disease_classes = list(DISEASE_INFO.keys())  # List of all the diseases
    predicted_disease = disease_classes[predicted_class_index]

    return predicted_disease, confidence

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Route to handle image upload and prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Check if the image is part of the request
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    # Create a temporary file for the uploaded image
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(file.read())
        filename = temp_file.name

    # Show loading screen while processing
    time.sleep(2)  # Simulate processing time

    # Make prediction
    predicted_class, confidence = predict_skin_disease(filename)

    # Retrieve disease information
    disease_info = DISEASE_INFO.get(predicted_class, {'description': 'No description available.', 'treatment': 'No treatment available.'})

    # Return the result
    result = {
        'disease': predicted_class,
        'confidence': confidence,
        'description': disease_info['description'],
        'treatment': disease_info['treatment'],
    }

    return jsonify(result)

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
