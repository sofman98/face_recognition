from face_recognition import face_locations

def get_face_locations(rgb_image):
    return face_locations(rgb_image)