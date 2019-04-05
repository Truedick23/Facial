import face_recognition
from PIL import Image, ImageDraw
import json

image = face_recognition.load_image_file("F:/FacialData/40-09.jpg")

face_landmarks_list = face_recognition.face_landmarks(image)

pil_image = Image.fromarray(image)
d = ImageDraw.Draw(pil_image)

for face_landmarks in face_landmarks_list:
    for facial_feature in face_landmarks.keys():
        print(facial_feature, face_landmarks[facial_feature])
        d.line(face_landmarks[facial_feature], width=2, fill='#000000')

pil_image.show()
