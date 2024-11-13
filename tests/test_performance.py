import base64
import time
import pywt
import requests
import concurrent.futures
import cv2
import numpy as np
import logging

# Setting up logging for detailed output tracking
logging.basicConfig(level=logging.INFO)
print("Starting Performance Tests...")

# Constants
SERVER_URL = "http://127.0.0.1:5000/classify_image"  # URL for image upload endpoint
TEST_IMAGE_PATH = r'UI/test_images/kohli_1.jpg'  # Path to a test image for processing

def process_image_with_hair_cascade(image_path):
    """Simulate Haar cascade processing."""
    print("Starting Haar Cascade processing...")
    image = cv2.imread(image_path)
    cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = cascade.detectMultiScale(gray, 1.1, 4)
    print(f"Detected {len(faces)} face(s) using Haar Cascade.")
    return faces

def wavelet_transform(image_path):
    """Simulate wavelet transformation."""
    print("Starting Wavelet Transformation...")
    image = cv2.imread(image_path, 0)  # Read in grayscale
    coeffs = pywt.wavedec2(image, 'haar', level=2)
    print("Wavelet transformation completed.")
    return coeffs

def test01_image_processing_time():
    """Test the image processing time for Haar cascade and wavelet processing."""
    print("\nTesting image processing time for Haar Cascade and Wavelet Transformation.")
    
    start_time = time.time()
    faces = process_image_with_hair_cascade(TEST_IMAGE_PATH)
    haar_processing_time = time.time() - start_time
    print(f"Haar Cascade Processing Time: {haar_processing_time:.4f} seconds")

    start_time = time.time()
    wavelet_coeffs = wavelet_transform(TEST_IMAGE_PATH)
    wavelet_processing_time = time.time() - start_time
    print(f"Wavelet Transformation Time: {wavelet_processing_time:.4f} seconds")

def test02_classification_speed():
    """Test the classification speed."""
    print("\nTesting classification speed...")
    image_path = TEST_IMAGE_PATH
    
    # Convert image to base64
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    
    # Start timing the request
    start_time = time.time()
    
    # Attempt to make the POST request
    response = requests.post(SERVER_URL, {"image_data": encoded_string})
    classification_time = time.time() - start_time
    
    # Log and print response details
    if response.status_code == 200:
        print(f"Classification Response: {response.json()}")
    else:
        print(f"Expected status code 200, but got {response.status_code}.")
        print(f"Response Body: {response.text}")  # Log the response body for more insight
        logging.error(f"Error in response: Status code {response.status_code}, Body: {response.text}")

    print(f"Classification Speed: {classification_time:.4f} seconds")
    
    # Ensure the test checks for successful status
    assert response.status_code == 200, f"Test failed: Expected status code 200, but got {response.status_code}"

if __name__ == "__main__":
    print("Starting Performance Tests...")
    
    print("\n1. Image Processing Time Test:")
    test01_image_processing_time()
    
    print("\n2. Classification Speed Test:")
    test02_classification_speed()
