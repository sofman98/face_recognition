import os
import cv2
import numpy as np
import argparse
import warnings
import time

from anti_spoofing.src.anti_spoof_predict import AntiSpoofPredict
from anti_spoofing.src.generate_patches import CropImage
from anti_spoofing.src.utility import parse_model_name


def verify_liveness(image):
    # setting up some value
    device_id = 0
    model_dir = "./anti_spoofing/ressources/anti_spoof_models"

    # #get the face coordinates
    # (top, right, bottom, left) = face_location

    # #add a little extra surroundings while making sure we don't overflow
    # (max_height, max_width, _) = image.shape

    # top = max(0, top-5)
    # bottom = min(max_height, bottom+5)
    # right = max(0, right-5)
    # left = min(max_width, left+5)

    # #crop the frame to encapsulate the face plus a little more        
    # image = image[top:bottom, left:right]

    # # the model
    model_test = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    image_bbox = model_test.get_bbox(image)
    prediction = np.zeros((1, 3))

    # sum the prediction from single model's result
    for model_name in os.listdir(model_dir):
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)
        prediction += model_test.predict(img, os.path.join(model_dir, model_name))

    # draw result of prediction
    label = np.argmax(prediction)
    value = prediction[0][label]/2

    #convert bbox to face locations
    left = image_bbox[0]
    top = image_bbox[1]
    right = image_bbox[0] + image_bbox[2]
    bottom = image_bbox[1] + image_bbox[3]

    face_location = (top, right, bottom, left)

    return face_location, label == 1