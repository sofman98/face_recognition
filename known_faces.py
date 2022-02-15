from face_recognition import load_image_file, face_encodings

# Load sample pictures and learn how to recognize it.
zeghoud_image = load_image_file("faces/zeghoud_id.jpg")
zeghoud_face_encoding = face_encodings(zeghoud_image)[0]

bousri_image = load_image_file("faces/bousri_id.jpg")
bousri_face_encoding = face_encodings(bousri_image)[0]

zahar_image = load_image_file("faces/zahar_id.jpg")
zahar_face_encoding = face_encodings(zahar_image)[0]

khiar_image = load_image_file("faces/khiar_id.jpg")
khiar_face_encoding = face_encodings(khiar_image)[0]

hadjmiloud_image = load_image_file("faces/hadjmiloud_id.jpg")
hadjmiloud_face_encoding = face_encodings(hadjmiloud_image)[0]

labreche_image = load_image_file("faces/labreche_id.jpg")
labreche_face_encoding = face_encodings(labreche_image)[0]


# Create arrays of known face encodings and their names
known_face_encodings = [
    zeghoud_face_encoding,
    bousri_face_encoding,
    zahar_face_encoding,
    khiar_face_encoding,
    hadjmiloud_face_encoding,
    labreche_face_encoding,
]

known_face_names = [
    "Sofiane Zeghoud",
    "Houssam Bousri",
    "Youcef Zahar",
    "Hamza Khiar",
    "Abdelkader Hadj Miloud",
    "Labreche Massinissa"
]