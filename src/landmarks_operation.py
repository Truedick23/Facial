import pymongo
import os
import face_recognition
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import dlib
import cv2
import imutils

def get_faces_collection():
    client = pymongo.MongoClient()
    return client.facial.landmark1

def get_facial_landmarks(path, pic_name):
    image = face_recognition.load_image_file(path)

    face_landmarks_list = face_recognition.face_landmarks(image)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    if len(face_landmarks_list) > 0:
        face_landmarks = face_landmarks_list[0]
        return face_landmarks
    else:
        return dict()

def init_pic_data():
    faces_info = get_faces_collection()
    path = 'F:/FacialData/cropped/'
    for person in range(1, 51):
        for direction in range(1, 26):
            person_str = str(person)
            if len(person_str) < 2:
                person_str = '0' + person_str
            direction_str = str(direction)
            if len(direction_str) < 2:
                direction_str = '0' + direction_str
            pic = person_str + '-' + direction_str + '.jpg'
            pic_path = path + pic
            print(pic_path)
            if os.path.isfile(pic_path):
                face_landmarks = get_facial_landmarks(pic_path, pic)
                if len(face_landmarks) == 0:
                    hasFace = False
                else:
                    hasFace = True
                faces_info.insert_one({
                    'PicName': pic,
                    'Person': person,
                    'Direction': direction,
                    'FacialLandmarks': face_landmarks,
                    'HasFace': hasFace,
                    'PicExists': True
                })
            else:
                faces_info.insert_one({
                    'PicName': pic,
                    'Person': person,
                    'Direction': direction,
                    'HasFace': False,
                    'FacialLandmarks': dict(),
                    'Exists': False
                })


def get_points_by_name(pic_name):
    faces_info = get_faces_collection()
    face = faces_info.find_one({'PicName': pic_name})
    points = face['FacialLandmarks']
    for name in points.keys():
        print(name)
        print(len(points[name]))

    path = 'F:/FacialData/cropped/' + pic_name
    image = face_recognition.load_image_file(path)

    pil_image = Image.fromarray(image)
    d = ImageDraw.Draw(pil_image)

    nose_tip = (points['nose_bridge'][3])
    chin = (points['chin'][8])
    left_eye_left_corner = (points['left_eye'][0])
    right_eye_right_corner = (points['right_eye'][3])
    left_mouth_corner = (((points['top_lip'][0][0] + points['bottom_lip'][6][0]) / 2), ((points['top_lip'][0][1] + points['bottom_lip'][6][1]) / 2))
    right_mouth_corner = (((points['top_lip'][6][0] + points['bottom_lip'][0][0]) / 2), ((points['top_lip'][6][1] + points['bottom_lip'][0][1]) / 2))

    landmarks = [nose_tip, chin, left_mouth_corner, right_eye_right_corner, left_eye_left_corner, left_mouth_corner, right_mouth_corner]

    for mark in landmarks:
        d.point(mark, fill='#ff0000')

    pil_image.show()

def geteye_rect(imgpath):
    face_detector = dlib.get_frontal_face_detector()
    landmark_predictor = dlib.shape_predictor("D:/QMDownload/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
    bgrImg = cv2.imread(imgpath)
    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)
    rects = face_detector(rgbImg, 1)

    shape = landmark_predictor(rgbImg, 0)

    for (x, y) in shape:
        cv2.circle(rgbImg, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", rgbImg)
    cv2.waitKey(0)
    plt.imshow(rgbImg)
    plt.show()

if __name__ == '__main__':
    # init_pic_data()
    geteye_rect('F:/FacialData/cropped/01-01.jpg')