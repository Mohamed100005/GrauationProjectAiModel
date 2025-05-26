import requests
import os
from PIL import Image
import io

def test_api():
    # Base URL - change this if your server is running on a different address
    BASE_URL = "y"
    
    # Test 1: Check if API is online
    print("\n1. Testing API health...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print("Error:", str(e))

    # Test 2: Check health endpoint
    print("\n2. Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/v1/health")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print("Error:", str(e))

    # Test 3: Test prediction endpoint without image
    print("\n3. Testing prediction endpoint without image...")
    try:
        response = requests.post(f"{BASE_URL}/api/v1/predict")
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print("Error:", str(e))

    # Test 4: Test prediction endpoint with sample image
    print("\n4. Testing prediction endpoint with sample image...")
    try:
        # Create a sample image for testing
        img = Image.new('RGB', (224, 224), color='white')
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr = img_byte_arr.getvalue()

        files = {'file': ('test_image.png', img_byte_arr, 'image/png')}
        response = requests.post(f"{BASE_URL}/api/v1/predict", files=files)
        print(f"Status Code: {response.status_code}")
        print("Response:", response.json())
    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    print("Starting API tests...")
    test_api() 