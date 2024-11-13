import sys
import os
import pytest
import base64
sys.path.insert(0, os.path.abspaht(os.path.join(os.path.dirname(__file__), '../server')))

import util

util.load_saved_artifacts()

# def test_load_saved_artifacts():
    # assert util._model is not None
    # assert isinstance(util._class_name_to_number, dict)

def test_classify_image():
    util.load_saved_artifacts()
    # Replace 'test_base64_image_data' with actual base64 test image data
    
    with open(r"UI/test_images/kohli_1.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')

    result = util.classify_image(encoded_string)
    assert isinstance(result, list)
    assert 'class' in result[0]
    assert 'class_probability' in result[0]
    assert 'class_dictionary' in result[0]
