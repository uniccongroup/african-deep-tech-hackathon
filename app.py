from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from PIL import Image
import base64
import io
from ultralytics import YOLO
import time
from datetime import datetime
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
# No SECRET_KEY needed if we don't use flash messages

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
MODEL_PATH = 'runs/classify/train4/weights/best.pt'
model = None

def load_model():
    """Load the YOLOv8 model with error handling"""
    global model
    try:
        if Path(MODEL_PATH).exists():
            model = YOLO(MODEL_PATH)
            logger.info(f"Model loaded successfully from {MODEL_PATH}")
            return True
        else:
            logger.error(f"Model file not found at {MODEL_PATH}")
            return False
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return False

# Initialize model on startup
model_loaded = load_model()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_malaria(image_path):
    """Make prediction on uploaded image"""
    if not model_loaded or model is None:
        return {
            'error': 'Model not loaded. Please check server configuration.',
            'prediction': None,
            'confidence': 0.0
        }
    
    try:
        # Make prediction
        results = model(image_path, verbose=False)
        
        # Extract prediction details
        predicted_class = results[0].names[results[0].probs.top1]
        confidence = results[0].probs.top1conf.item()
        
        # Get probabilities for both classes
        probs = results[0].probs.data.cpu().numpy()
        class_names = list(results[0].names.values())
        
        # Create detailed results
        detailed_results = {
            'prediction': predicted_class,
            'confidence': confidence,
            'all_probabilities': {
                class_names[i]: float(probs[i]) for i in range(len(class_names))
            },
            'recommendation': get_recommendation(predicted_class, confidence),
            'risk_level': get_risk_level(predicted_class, confidence)
        }
        
        return detailed_results
        
    except Exception as e:
        return {
            'error': str(e),
            'prediction': None,
            'confidence': 0.0
        }

def get_recommendation(prediction, confidence):
    """Get medical recommendation based on prediction"""
    if confidence < 0.7:
        return "Low confidence prediction. Please consult a medical professional for manual examination."
    elif prediction == 'Parasitized':
        if confidence > 0.95:
            return "High confidence malaria detection. Immediate medical attention recommended."
        else:
            return "Possible malaria infection detected. Please consult a healthcare provider for confirmation."
    else:
        if confidence > 0.95:
            return "No malaria parasites detected with high confidence. Regular monitoring recommended."
        else:
            return "Likely no malaria infection. Consider retesting if symptoms persist."

def get_risk_level(prediction, confidence):
    """Get risk level for UI styling"""
    if confidence < 0.7:
        return "warning"
    elif prediction == 'Parasitized':
        return "danger"
    else:
        return "success"

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction request - WITHOUT flash messages"""
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='No file selected')
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        start_time = time.time()
        results = predict_malaria(filepath)
        prediction_time = time.time() - start_time
        
        # Convert image to base64 for display
        with open(filepath, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Clean up uploaded file
        os.remove(filepath)
        
        return render_template('result.html', 
                             results=results,
                             image_data=img_data,
                             prediction_time=prediction_time)
    
    return render_template('index.html', error='Invalid file type. Please upload an image file.')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for prediction"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        results = predict_malaria(filepath)

        os.remove(filepath)
        
        return jsonify(results)
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/about')
def about():
    """About page"""
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
