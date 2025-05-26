
Dog Skin Disease Classifier API

To run locally:
1. Install dependencies:
   pip install -r requirements.txt

2. Make sure best_model.pth is in the same directory.

3. Run the app:
   python app.py

To deploy on Azure App Service or any cloud:
- Use 'gunicorn' as the startup command:
   gunicorn app:app

POST request to /predict with 'file' parameter containing an image.
Response: JSON with predicted class and class probabilities.
