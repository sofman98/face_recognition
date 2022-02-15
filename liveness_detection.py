import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import os
import numpy as np
from tensorflow.keras.models import model_from_json

root_dir = os.getcwd()
# Load Anti-Spoofing Model graph
json_file = open('models/antispoofing_model.json','r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
# load antispoofing model weights 
model.load_weights('models/antispoofing_model.h5')
print("Anti-spoofing model loaded from disk")


#function to verify liveness of one face
def verify_liveness(frame, face_location, threshold=0.05):
    try:
        #get the face coordinates
        (top, right, bottom, left) = face_location

        #add a little extra surroundings while making sure we don't overflow
        (max_height, max_width, _) = frame.shape

        top = max(0, top-5)
        bottom = min(max_height, bottom+5)
        right = max(0, right-5)
        left = min(max_width, left+5)

        #crop the frame to encapsulate the face plus a little more        
        located_face = frame[top:bottom, left:right]

        # resize to fit the model's expected input shape
        resized_face = cv2.resize(located_face, (160, 160))
        resized_face = resized_face.astype("float") / 255.0
        resized_face = np.expand_dims(resized_face, axis=0)

        prediction = model.predict(resized_face)[0]

        return prediction <= threshold
    except Exception as e:
        print(e)
        return False