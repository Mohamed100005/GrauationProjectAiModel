import requests
import json
import time

# API Base URL - using default Flask port
BASE_URL = "http://127.0.0.1:5000"

def wait_for_api(max_retries=5):
    """Wait for the API to become available"""
    url = f"{BASE_URL}/"
    for i in range(max_retries):
        try:
            response = requests.get(url)
            if response.status_code == 200:
                print("API is ready!")
                return True
        except requests.exceptions.ConnectionError:
            print(f"Waiting for API to start... (attempt {i+1}/{max_retries})")
            time.sleep(2)
    return False

def test_api_health():
    """Test if the API is working"""
    url = f"{BASE_URL}/api/v1/health"
    try:
        response = requests.get(url)
        print("API Health Check:", response.json())
        return True
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
        return False

def predict_skin_disease(image_path):
    """Send an image to the API for prediction"""
    url = f"{BASE_URL}/api/v1/predict"
    
    try:
        # Open the image file
        with open(image_path, 'rb') as image_file:
            # Create the files parameter for the POST request
            files = {'file': image_file}
            
            # Send POST request to the API
            response = requests.post(url, files=files)
            
            # Print the response
            if response.status_code == 200:
                result = response.json()
                print("\nPrediction Results:")
                print("Detected Disease:", result['prediction']['class'])
                print("Confidence:", f"{result['prediction']['confidence']}%")
                print("\nAll Probabilities:")
                for disease, prob in result['prediction']['all_probabilities'].items():
                    print(f"{disease}: {prob}%")
            else:
                print("Error:", response.json())
    except FileNotFoundError:
        print(f"Error: Could not find the image file at: {image_path}")
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the API. Make sure the server is running.")
    except Exception as e:
        print(f"Error: An unexpected error occurred: {str(e)}")

if __name__ == "__main__":
    print("Checking API connection...")
    if wait_for_api():
        if test_api_health():
            # Then, test with an image
            image_path = input("\nEnter the path to your dog skin image: ")
            predict_skin_disease(image_path)
    else:
        print("\nCould not connect to the API. Please make sure:")
        print("1. The server (run_server.py) is running")
        print("2. No other application is using port 5000")
        print("3. Your firewall is not blocking the connection")
        print("\nTry accessing the API directly in your browser:")
        print(f"{BASE_URL}/api/v1/health") 