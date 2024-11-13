from http import client
import requests
import pytest
import base64

SERVER_URL = "http://127.0.0.1:5000/classify_image"  # URL for image upload endpoint
VALID_IMAGE_PATH = r'UI/test_images/kohli_1.jpg'  # Path to a test image for processing
INVALID_IMAGE_PATH = r'UI/test_images/invalid_file.txt'  # Replace with a non-image file path
NO_FACE_IMAGE_PATH = r'UI/test_images/no_face.jpg'

def test01_classify_image_with_valid_image():
    """Test uploading a valid image and receiving a classification."""
    print("\ntest01_classify_image_with_valid_image started...")
    with open(VALID_IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        response = requests.post(SERVER_URL, data={'image_data': encoded_string})
        
        print("Status Code:", response.status_code)
        print("Response Data:", response.json())
        
        assert response.status_code == 200
        response_data = response.json()
        
        # Confirm the response includes classification data
        assert len(response_data) > 0, "Expected classification data in response"
        assert "virat_kohli" in response_data[0]['class']  # Adjust based on your API response structure
    print("test01_classify_image_with_valid_image ended ✅")

def test02_invalid_image_upload_handling():
    """Test handling of invalid image uploads."""
    print("\ntest02_invalid_image_upload_handling started...")
    with open(INVALID_IMAGE_PATH, "rb") as image_file:
        response = requests.post(SERVER_URL, files={"file": image_file})
        
        print("Status Code:", response.status_code)
        
    assert response.status_code == 400
    print("test02_invalid_image_upload_handling ended ✅")

def test03_concurrent_requests_handling():
    """Test server handling of concurrent requests."""
    print("\ntest03_concurrent_requests_handling started...")
    import threading

    def send_request():
        with open(VALID_IMAGE_PATH, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            response = requests.post(SERVER_URL, data={'image_data': encoded_string})
            
            print("Thread Response Status Code:", response.status_code)
            assert response.status_code == 200  # Each thread should succeed

    threads = []
    for _ in range(10):  # Adjust the number of concurrent requests as needed
        thread = threading.Thread(target=send_request)
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()
    print("test03_concurrent_requests_handling ended ✅")

def test04_no_face_image_handling_in_data_flow():
    """Test handling of image with no detectable faces."""
    print("\ntest04_no_face_image_handling_in_data_flow started...")
    with open(NO_FACE_IMAGE_PATH, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        response = requests.post(SERVER_URL, data={'image_data': encoded_string})
        
        print("Status Code:", response.status_code)
        print("Response Data:", response.json())
        
        # Expect an empty list or an appropriate message when no faces are detected
        assert response.status_code == 200
        assert response.json() == []  # Modify if your API returns a specific message
    print("test04_no_face_image_handling_in_data_flow ended ✅")

# Example of running tests directly
if __name__ == "__main__":
    pytest.main()
