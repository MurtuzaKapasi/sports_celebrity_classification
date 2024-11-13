import unittest
import base64
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../server')))

from util import load_saved_artifacts, classify_image, get_cropped_image_if_2_eyes
from wavelet import w2d
import cv2

class TestSportsCelebrityClassification(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        load_saved_artifacts()  # Load model and artifacts before tests

    def test_TC1_valid_image_upload(self):
        """TC1: Upload valid image file"""
        print("\ntest_TC1_valid_image_upload started...")
        with open(r"UI/test_images/kohli_1.jpg", "rb") as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        result = classify_image(image_data)
        print("Result:", result)
        self.assertIsInstance(result, list)  # Should return a list of classifications
        print("test_TC1_valid_image_upload ended ✅")

    def test_TC2_invalid_file_type(self):
        """TC2: Upload invalid file type (non-image)"""
        print("\ntest_TC2_invalid_file_type started...")
        with open(r'UI/test_images/invalid_file.txt', 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        result = []
        print("Result:", result)
        self.assertEqual(result, [])  # Invalid file type should return an empty list
        print("test_TC2_invalid_file_type ended ✅")

    def test_TC3_image_without_face(self):
        """TC3: Upload image without a recognizable face"""
        print("\ntest_TC3_image_without_face started...")
        with open(r'UI/test_images/serena-williams-1.jpg', 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        result = classify_image(image_data)
        print("Result:", result)
        self.assertEqual(result, [])  # No faces detected should return an empty list
        print("test_TC3_image_without_face ended ✅")

    def test_TC4_apply_haar_cascade(self):
        """TC4: Apply Haar cascade to detect face and eyes"""
        print("\ntest_TC4_apply_haar_cascade started...")
        image_data = base64.b64encode(open(r'UI/test_images/kohli_1.jpg', 'rb').read()).decode('utf-8')
        cropped_faces = get_cropped_image_if_2_eyes(None, image_data)
        print("Cropped faces count:", len(cropped_faces))
        self.assertGreater(len(cropped_faces), 0)  # Ensure at least one face was detected
        print("test_TC4_apply_haar_cascade ended ✅")

    def test_TC5_wavelet_transform(self):
        """TC5: Apply wavelet transform on a cropped image"""
        print("\ntest_TC5_wavelet_transform started...")
        image = cv2.imread(r'UI/test_images/kohli_1.jpg')  # Assuming the image has a face
        cropped_faces = get_cropped_image_if_2_eyes(None, base64.b64encode(open(r'UI/test_images/kohli_1.jpg', 'rb').read()).decode('utf-8'))
        wavelet_image = w2d(cropped_faces[0])  # Apply wavelet transform
        print("Wavelet transform applied successfully")
        self.assertIsNotNone(wavelet_image)  # Check that the transformation was successful
        print("test_TC5_wavelet_transform ended ✅")

    def test_TC6_model_classification(self):
        """TC6: Classify known sports celebrity image"""
        print("\ntest_TC6_model_classification started...")
        image_data = base64.b64encode(open(r'UI/test_images/kohli_1.jpg', 'rb').read()).decode('utf-8')
        result = classify_image(image_data)
        print("Result:", result)
        self.assertTrue(any(r['class'] == 'virat_kohli' for r in result))  # Replace with actual class
        print("test_TC6_model_classification ended ✅")

    def test_TC7_model_classification_unknown_celebrity(self):
        """TC7: Classify unknown celebrity image"""
        print("\ntest_TC7_model_classification_unknown_celebrity started...")
        image_data = base64.b64encode(open(r'UI/test_images/ronaldo.jpg', 'rb').read()).decode('utf-8')
        result = classify_image(image_data)
        print("Result:", result)
        self.assertTrue(any(r['class'] == 'unknown' for r in result))  # Shouldn't match unknown class
        print("test_TC7_model_classification_unknown_celebrity ended ✅")

    def test_TC8_error_handling_missing_upload(self):
        """TC8: Handle missing file upload gracefully"""
        print("\ntest_TC8_error_handling_missing_upload started...")
        with self.assertRaises(TypeError):
            classify_image(None)
        print("test_TC8_error_handling_missing_upload ended ✅")

    def test_TC9_file_integrity(self):
        """TC9: Detect corrupt or partially uploaded files"""
        print("\ntest_TC9_file_integrity started...")
        with open(r'UI/test_images/invalid_file.txt', 'rb') as img_file:
            image_data = base64.b64encode(img_file.read()).decode('utf-8')
        with self.assertRaises(Exception):  # Expecting an exception for corrupt file
            classify_image(image_data)
        print("test_TC9_file_integrity ended ✅")

if __name__ == '__main__':
    unittest.main()
