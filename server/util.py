import joblib
import json
import numpy as np
import base64
import cv2
from wavelet import w2d

_class_name_to_number = {}
_class_number_to_name = {}

_model = None

# def classify_image(image_base64_data, file_path=None):

#     imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
#     if not imgs:
#         print("No face detected")
#         return []
    
#     result = []
#     for img in imgs:
#         scalled_raw_img = cv2.resize(img, (32, 32))
#         img_har = w2d(img, 'db1', 5)
#         scalled_img_har = cv2.resize(img_har, (32, 32))
#         combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

#         len_image_array = 32*32*3 + 32*32

#         final = combined_img.reshape(1,len_image_array).astype(float)
#         pred_class = _model.predict(final)[0]
#         pred_class_prob = np.around(_model.predict_proba(final)*100,2).tolist()[0][pred_class]
        
#         if pred_class_prob >= 75.00:
#             result.append({
#                 'class': class_number_to_name(pred_class),
#                 'class_probability': pred_class_prob,
#                 'class_dictionary': _class_name_to_number
#             })
#         else:
#             # If below 75%, classify as 'unknown'
#             result.append({
#                 'class': 'unknown',
#                 'class_probability': 0.00,
#                 'class_dictionary': {}
#             })
    
#     return result


def classify_image(image_base64_data, file_path=None):
    imgs = get_cropped_image_if_2_eyes(file_path, image_base64_data)
    
    # If no images (no faces detected), return an appropriate response
    if not imgs:
        print("No face detected")
        return []
    
    result = []
    
    for img in imgs:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        
    #     combined_img = np.vstack((
    #         scalled_raw_img.reshape(32 * 32 * 3, 1), 
    #         scalled_img_har.reshape(32 * 32, 1)
    #     ))
        
    #     len_image_array = 32 * 32 * 3 + 32 * 32
    #     final = combined_img.reshape(1, len_image_array).astype(float)
        
    #     # Get predictions
    #     pred_class = _model.predict(final)[0]
    #     pred_class_prob = np.around(_model.predict_proba(final) * 100, 2).tolist()[0][pred_class]
        
    #     # Only keep predictions above the threshold
    #     if pred_class_prob >= 20.00:
    #         result.append({
    #             'class': class_number_to_name(pred_class),
    #             'class_probability': pred_class_prob,
    #             'class_dictionary': _class_name_to_number
    #         })
    
    # # If no valid predictions above threshold, return unknown
    # if not result:
    #     return [{'class': 'unknown', 'class_probability': 0.00, 'class_dictionary': {}}]
    
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))

        len_image_array = 32*32*3 + 32*32

        final = combined_img.reshape(1,len_image_array).astype(float)
        result.append({
            'class': class_number_to_name(_model.predict(final)[0]),
            'class_probability': np.around(_model.predict_proba(final)*100,2).tolist()[0],
            'class_dictionary': _class_name_to_number
        })
        if not result:
            result.append({'class': 'unknown', 'class_probability': 0.00, 'class_dictionary': {}})
    return result


def class_number_to_name(class_num):
    return _class_number_to_name[class_num]

def load_saved_artifacts():
    print("loading saved artifacts...start")
    global _class_name_to_number
    global _class_number_to_name

    
    with open("server/artifacts/class_dictionary.json", "r") as f:
        _class_name_to_number = json.load(f)
        _class_number_to_name = {v:k for k,v in _class_name_to_number.items()}

    global _model
    if _model is None:
        with open('server/artifacts/saved_model.pkl', 'rb') as f:
            _model = joblib.load(f)
    print("loading saved artifacts...done")

def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    # encoded_data = b64str.split(',')[1]

    if ',' in b64str:
        encoded_data = b64str.split(',')[1]
    else:
        encoded_data = b64str
    
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def get_cropped_image_if_2_eyes(image_path, image_base64_data):
    face_cascade = cv2.CascadeClassifier('server/opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('server/opencv/haarcascades/haarcascade_eye.xml')

    if image_path:
        img = cv2.imread(image_path)
    else:
        img = get_cv2_image_from_base64_string(image_base64_data)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces
