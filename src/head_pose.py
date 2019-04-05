import cv2
import numpy as np
import pymongo
import face_recognition
import cv2
from cv2 import VideoCapture
from PIL import Image, ImageDraw

def get_faces_collection():
    client = pymongo.MongoClient()
    return client.facial.landmark1

def from_cam():
    video_capture = cv2.VideoCapture(1)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    w = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH) * 0.5)
    h = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.5)

    out = cv2.VideoWriter('D:/PycharmProjects/Facial/videos/output.avi', fourcc, 25, (w, h))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret == True:
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            face_landmarks_list = face_recognition.face_landmarks(small_frame)

            im = small_frame
            size = im.shape

            if len(face_landmarks_list) > 0:
                face_landmarks = face_landmarks_list[0]
                points = face_landmarks

                nose_tip = (points['nose_bridge'][3])
                chin = (points['chin'][8])
                left_eye_left_corner = (points['left_eye'][0])
                right_eye_right_corner = (points['right_eye'][3])
                left_mouth_corner = (((points['top_lip'][0][0] + points['bottom_lip'][6][0]) / 2),
                                     ((points['top_lip'][0][1] + points['bottom_lip'][6][1]) / 2))
                right_mouth_corner = (((points['top_lip'][6][0] + points['bottom_lip'][0][0]) / 2),
                                      ((points['top_lip'][6][1] + points['bottom_lip'][0][1]) / 2))

                image_points = np.array([
                    nose_tip, chin,
                    left_eye_left_corner, right_eye_right_corner,
                    left_mouth_corner, right_mouth_corner
                ], dtype='double')

                model_points = np.array([
                    (0.0, 0.0, 0.0),  # Nose tip
                    (0.0, -330.0, -65.0),  # Chin
                    (-225.0, 170.0, -135.0),  # Left eye left corner
                    (225.0, 170.0, -135.0),  # Right eye right corne
                    (-150.0, -150.0, -125.0),  # Left Mouth corner
                    (150.0, -150.0, -125.0)  # Right mouth corner

                ])

                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = np.array(
                    [[focal_length, 0, center[0]],
                     [0, focal_length, center[1]],
                     [0, 0, 1]], dtype="double"
                )

                dist_coeffs = np.zeros((4, 1))

                (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                              dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

                (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                                 translation_vector, camera_matrix, dist_coeffs)

                for p in image_points:
                    cv2.circle(im, (int(p[0]), int(p[1])), 1, (0, 0, 255), -1)

                p1 = (int(image_points[0][0]), int(image_points[0][1]))
                p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

                cv2.line(im, p1, p2, (255, 0, 0), 2)
                # Hit 'q' on the keyboard to quit!
            out.write(im)
            cv2.imshow('Video', im)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release handle to the webcam
    video_capture.release()
    out.release()
    cv2.destroyAllWindows()

def generate_model_points(face):
    points = face['FacialLandmarks']

    # nose_tip = (points['nose_tip'][])
    nose_tip = (points['nose_bridge'][3])
    left_eye_left_corner = (points['left_eye'][0])
    right_eye_right_corner = (points['right_eye'][3])
    left_mouth_corner = (((points['top_lip'][0][0] + points['bottom_lip'][6][0]) / 2),
                         ((points['top_lip'][0][1] + points['bottom_lip'][6][1]) / 2))
    right_mouth_corner = (((points['top_lip'][6][0] + points['bottom_lip'][0][0]) / 2),
                          ((points['top_lip'][6][1] + points['bottom_lip'][0][1]) / 2))

    nt_left_eye_left_corner = [left_eye_left_corner[0] - nose_tip[0], left_eye_left_corner[1] - nose_tip[1], -135.0]
    nt_right_eye_right_corner = [right_eye_right_corner[0] - nose_tip[0], right_eye_right_corner[1] - nose_tip[1], -135.0]

    nt_left_mouth_corner = [left_mouth_corner[0] - nose_tip[0], left_mouth_corner[1] - nose_tip[1], -125.0]
    nt_right_mouth_corner = [right_mouth_corner[0] - nose_tip[0], right_mouth_corner[1] - nose_tip[1], -125.0]


    nt_left_eye_left_corner[0] = nt_left_eye_left_corner[0] * (170.0 / nt_left_eye_left_corner[1])
    nt_left_eye_left_corner[1] = 170.0

    nt_right_eye_right_corner[0] = nt_right_eye_right_corner[0] * (170.0 / nt_right_eye_right_corner[1])
    nt_right_eye_right_corner[1] = 170.0

    nt_left_mouth_corner[0] = nt_left_mouth_corner[0] * (-150.0 / nt_left_mouth_corner[1])
    nt_left_mouth_corner[1] = -150.0

    nt_right_mouth_corner[0] = nt_right_mouth_corner[0] * (-150.0 / nt_right_mouth_corner[1])
    nt_right_mouth_corner[1] = -150.0

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (nt_left_eye_left_corner),  # Left eye left corner
        (nt_right_eye_right_corner),  # Right eye right corne
        (nt_left_mouth_corner),  # Left Mouth corner
        (nt_right_mouth_corner)  # Right mouth corner
    ])

    return model_points


def get_head_pose(person, direction):

    faces_info = get_faces_collection()

    face = faces_info.find_one({'$and': [{'Person': person}, {'Direction': direction}]})

    if face['HasFace'] == False:
        return False

    '''
    first_face = faces_info.find_one({'$and': [{'Person': person}, {'Direction': 1}]})
    if first_face == None:
        model_points = np.array([
            (0.0, 0.0, 0.0),  # Nose tip
            (0.0, -330.0, -65.0),  # Chin
            (-165.0, 170.0, -135.0),  # Left eye left corner
            (165.0, 170.0, -135.0),  # Right eye right corne
            (-150.0, -150.0, -125.0),  # Left Mouth corner
            (150.0, -150.0, -125.0)  # Right mouth corner
        ])
    else:
        try:
            model_points = generate_model_points(first_face)
        except:
            model_points = np.array([
                (0.0, 0.0, 0.0),  # Nose tip
                (0.0, -330.0, -65.0),  # Chin
                (-165.0, 170.0, -135.0),  # Left eye left corner
                (165.0, 170.0, -135.0),  # Right eye right corne
                (-150.0, -150.0, -125.0),  # Left Mouth corner
                (150.0, -150.0, -125.0)  # Right mouth corner
            ])
    '''



    pic_name = face['PicName']
    points = face['FacialLandmarks']

    path = 'F:/FacialData/cropped/' + pic_name

    # nose_tip = (points['nose_tip'][])
    nose_tip = (points['nose_bridge'][3])
    chin = (points['chin'][8])
    left_eye_left_corner = (points['left_eye'][0])
    right_eye_right_corner = (points['right_eye'][3])
    left_mouth_corner = (((points['top_lip'][0][0] + points['bottom_lip'][6][0]) / 2),
                         ((points['top_lip'][0][1] + points['bottom_lip'][6][1]) / 2))
    right_mouth_corner = (((points['top_lip'][6][0] + points['bottom_lip'][0][0]) / 2),
                          ((points['top_lip'][6][1] + points['bottom_lip'][0][1]) / 2))

    im = cv2.imread(path)
    size = im.shape

    image_points = np.array([
        nose_tip, chin,
        left_eye_left_corner, right_eye_right_corner,
        left_mouth_corner, right_mouth_corner
    ], dtype='double')


    #standard
    ''' 
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner                        
    ])
    '''

    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-165.0, 170.0, -135.0),  # Left eye left corner
        (165.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    #experimental
    '''
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (2, -350.0, -65.0),  # Chin
        (-232.58181818181816, 170.0, -135.0),  # Left eye left corner
        (232.58181818181816, 170.0, -135.0),  # Right eye right corne
        (-110.195789600584685, -150.0, -125.0),  # Left Mouth corner
        (110.195789600584685, -150.0, -125.0)  # Right mouth corner

    ])
    '''

    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.zeros((4, 1))

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)

    for p in image_points:
        cv2.circle(im, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    p1 = (int(image_points[0][0]), int(image_points[0][1]))
    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))

    '''
    predict = 0
    if p2[1] < p1[1]:
        predict = 1
    if p2[1] > p1[1]:
        predict = 2
    if abs(p2[1] - p1[1]) <= 25 and (abs(p2[0] - p1[0])) / (abs(p2[1] - p1[1]) + 1) >= 1.5:
        predict = 0


    actual = 0
    if direction in [13, 12, 11, 10, 25,
                     14, 3, 2, 9, 24]:
        actual = 1
    if direction in [15, 4, 1, 8, 23]:
        actual = 0
    if direction in [16, 5, 6, 7, 22,
                     17, 18, 19, 20, 21]:
        actual = 2

    if actual == 0:
        try:
            print((abs(p2[0] - p1[0])) / abs(p2[1] - p1[1]), pic_name)
        except:
            print(pic_name)
    '''

    cv2.line(im, p1, p2, (255, 0, 0), 2)

    '''
    faces_info.update_one(
        {'PicName': pic_name},
        {'$set': {'Predict': predict, 'Actual': actual}}
    )
    '''


    # Display image
    cv2.imwrite('F:/FacialData/Posed/8/' + pic_name, im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    print(pic_name)

def get_average_marks():
    faces_info = get_faces_collection()
    total_c = [0.0, .0]
    total_lelc = [.0, .0]
    total_lerc = [.0, .0]
    total_lmc = [.0, .0]
    total_rmc = [.0, .0]
    for face in faces_info.find({'$and': [{'Direction': 1}, {'HasFace': True}]}):
        points = face['FacialLandmarks']
        nose_tip = (points['nose_bridge'][3])
        chin = (points['chin'][8])
        left_eye_left_corner = (points['left_eye'][0])
        right_eye_right_corner = (points['right_eye'][3])
        left_mouth_corner = (((points['top_lip'][0][0] + points['bottom_lip'][6][0]) / 2),
                             ((points['top_lip'][0][1] + points['bottom_lip'][6][1]) / 2))
        right_mouth_corner = (((points['top_lip'][6][0] + points['bottom_lip'][0][0]) / 2),
                              ((points['top_lip'][6][1] + points['bottom_lip'][0][1]) / 2))

        total_c[0] = total_c[0] + (nose_tip[0] - chin[0])
        total_c[1] = total_c[1] + (nose_tip[1] - chin[1])

        total_lelc[0] = total_lelc[0] + (nose_tip[0] - left_eye_left_corner[0])
        total_lelc[1] = total_lelc[1] + (nose_tip[1] - left_eye_left_corner[1])

        total_lerc[0] = total_lerc[0] + (nose_tip[0] - right_eye_right_corner[0])
        total_lerc[1] = total_lerc[1] + (nose_tip[1] - right_eye_right_corner[1])

        total_lmc[0] = total_lmc[0] + (nose_tip[0] - left_mouth_corner[0])
        total_lmc[1] = total_lmc[1] + (nose_tip[1] - left_mouth_corner[1])

        total_rmc[0] = total_rmc[0] + (nose_tip[0] - right_mouth_corner[0])
        total_rmc[1] = total_rmc[1] + (nose_tip[1] - right_mouth_corner[1])

    total_amount = faces_info.count({'$and': [{'Direction': 1}, {'HasFace': True}]})

    ave_c = [total_c[0] / total_amount, total_c[1] / total_amount]
    ave_lelc = [total_lelc[0] / total_amount, total_lelc[1] / total_amount]
    ave_lerc = [total_lerc[0] / total_amount, total_lerc[1] / total_amount]
    ave_lmc = [total_lmc[0] / total_amount, total_lmc[1] / total_amount]
    ave_rmc = [total_rmc[0] / total_amount, total_rmc[1] / total_amount]

    '''
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner

    ])
    '''

    ave_c = [ave_c[0] * (-330.0 / ave_c[1]), -330.0, 0.0]
    ave_lelc = [-ave_lelc[0] * (170.0 / ave_lelc[1]), 170.0, -135.0]
    ave_lerc = [-ave_lerc[0] * (170.0 / ave_lerc[1]), 170.0, -135.0]
    ave_lmc = [-ave_lmc[0] * (-150.0 / ave_lmc[1]), -150.0, -125.0]
    ave_rmc = [-ave_rmc[0] * (-150.0 / ave_rmc[1]), -150.0, -125.0]

    print(ave_c, '\n', ave_lelc, '\n', ave_lerc, '\n', ave_lmc, '\n', ave_rmc)




def get_accuracy():
    faces_info = get_faces_collection()

    correct_up = faces_info.count({'$and': [{'Predict': 1}, {'Actual': 1}]})
    predicted_up = faces_info.count({'Predict': 1})
    actually_not_up = faces_info.count({'Actual': 1})

    all_correct_num = faces_info.count({'$or': [{'$and': [{'Predict': 1}, {'Actual': 1}]},
                                                {'$and': [{'Predict': 2}, {'Actual': 2}]},
                                                {'$and': [{'Predict': 0}, {'Actual': 0}]}]})
    all_num = faces_info.count({'HasFace': True})

    print('Accuracy: {:.2%}, {:.2%}, {:.2%}'.format((correct_up / predicted_up), (correct_up / actually_not_up), (all_correct_num / all_num)))


def process():
    for person in range(1, 51):
        for direction in range(1, 26):
            get_head_pose(person, direction)

if __name__ == '__main__':
    process()
