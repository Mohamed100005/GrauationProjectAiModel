import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import torch
from torchvision import models, transforms
import torch.nn as nn
import io
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configuration
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'
PORT = int(os.getenv('PORT', 5000))
HOST = os.getenv('HOST', '0.0.0.0')

class_names = [
    'Dermatitis',
    'Fungal_infections',
    'Healthy',
    'Hypersensitivity',
    'demodicosis',
    'ringworm'
]

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model setup
try:
    model = models.resnet50()
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model = model.to(device)
    model.eval()
except Exception as e:
    print(f"Error loading model: {str(e)}")
    raise

# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.route('/')
def index():
    return jsonify({
        "status": "online",
        "message": "🚀 Dog Skin Disease Classifier API is live!",
        "version": "1.0.0",
        "endpoints": {
            "predict": "/api/v1/predict",
            "health": "/api/v1/health"
        }
    })

@app.route('/api/v1/health')
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })

@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({
            "error": "No image file provided",
            "status": "error",
            "required_format": "multipart/form-data with 'file' field"
        }), 400

    file = request.files['file']
    
    if not file.filename:
        return jsonify({
            "error": "Empty file provided",
            "status": "error"
        }), 400

    try:
        # Read and process the image
        image = Image.open(file).convert("RGB")
        image = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_idx = torch.argmax(probabilities).item()

        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item() * 100

        # Create detailed response
        prob_dict = {
            class_names[i]: round(probabilities[i].item() * 100, 2)
            for i in range(len(class_names))
        }

        return jsonify({
            "status": "success",
            "prediction": {
                "class": predicted_class,
                "confidence": round(confidence, 2),
                "all_probabilities": prob_dict
            },
            "model_info": {
                "device": str(device),
                "architecture": "ResNet50"
            }
        })

    except Exception as e:
        return jsonify({
            "error": str(e),
            "status": "error",
            "type": str(type(e).__name__)
        }), 500

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "error": "Resource not found",
        "status": "error"
    }), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({
        "error": "Internal server error",
        "status": "error"
    }), 500

if __name__ == '__main__':
    app.run(debug=DEBUG, host=HOST, port=PORT)
