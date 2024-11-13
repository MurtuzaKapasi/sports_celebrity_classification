import sys
import os
import pytest
from flask import Flask
import base64

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from server import app  # Importing the Flask app from server.py

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_classify_image(client):
    # Replace 'test_base64_image_data' with actual base64 test image data for real testing

    with open(r"UI/test_images/kohli_1.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    # with open(r"UI/test_images/kohli_1.jpg", "rb") as image_file:
    response = client.post('/classify_image', data={'image_data': encoded_string})
    assert response.status_code == 200
    assert 'class' in response.json[0]
    assert 'class_probability' in response.json[0]
    assert 'class_dictionary' in response.json[0]
