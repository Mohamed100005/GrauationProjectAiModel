# Dog Skin Disease Classifier API

This API uses a deep learning model to classify dog skin diseases from images.

## Setup Instructions

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Start the server:
```bash
python run_server.py
```

3. Use the API:
```bash
python use_api.py
```

## API Endpoints

- `GET /` - API information
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Submit an image for disease classification

## Supported Diseases

- Dermatitis
- Fungal infections
- Healthy
- Hypersensitivity
- Demodicosis
- Ringworm

## File Structure

- `app.py` - Main Flask application with model loading and API endpoints
- `run_server.py` - Server startup script
- `use_api.py` - Client script for testing the API
- `requirements.txt` - Python dependencies
- `best_model.pth` - Trained PyTorch model
- `animal_skin_diseases.pkl` - Additional model data

## Usage Example

1. Start the server:
```bash
python run_server.py
```

2. In a new terminal, run the client:
```bash
python use_api.py
```

3. When prompted, enter the path to your dog skin image.

## Response Format

```json
{
    "status": "success",
    "prediction": {
        "class": "Disease Name",
        "confidence": 95.5,
        "all_probabilities": {
            "Dermatitis": 2.1,
            "Fungal_infections": 1.2,
            "Healthy": 95.5,
            "Hypersensitivity": 0.5,
            "demodicosis": 0.4,
            "ringworm": 0.3
        }
    }
}
```
