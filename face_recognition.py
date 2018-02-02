#!/usr/bin/python3

import cv2
import os
import sys
import shutil
import numpy as np

def draw_rectangle(img, rect, border_size, *rect_color):
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x+w, y+h), *rect_color, border_size)

def draw_text(img, text, x, y, _font, _scale, thickness, *color):
    cv2.putText(img, text, (x, y), _font, _scale, *color, thickness)

def detect_face(image, _classifier=None):

    if _classifier is None:
        print("Classifier not specified. Exiting...")
        exit()
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(_classifier)
    faces = face_cascade.detectMultiScale(gray_image,
                                          scaleFactor=1.2,
                                          minNeighbors=5);
    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray_image[y:y+w, x:x+h], faces[0]


def sample_collection(Web_cam=None, train_dir=None, User_Id=None,
                      samples=10, _pre='E', _format="jpg"):
    Web_cam_flag = False
    usage = "sample_collection <Web_cam handle> <train_dir> <user_id> <sample count> <prefix> <image format>"

    if not Web_cam:
        Web_cam_flag = True
        Web_cam = cv2.VideoCapture(0)

    if not train_dir:
        print("Path to <train_dir> not provided", usage, sep="\n")
        return

    if not User_Id:
        print("User_Id not provided", usage, sep="\n")
        return

    sample_dir = "".join((train_dir, os.sep, _pre, User_Id))

    if os.path.exists(sample_dir):
        shutil.rmtree(sample_dir)

    os.mkdir(sample_dir)

    count=0
    _print_msg = True

    while count < samples:
        return_value, frame = Web_cam.read()
        cv2.imshow('Test_sample_collection_frame', frame)
        if _print_msg:
            print('User: {},engine needs {} more samples\n "c"-> capture image \n "q"-> Stop sample collection.'.format(User_Id, samples-count))
        _print_msg = False

        if cv2.waitKey(1) & 0xFF == ord('c'):
            _sample_name = sample_dir + os.sep + 'test_image' + str(count) + '.' + _format
            cv2.imwrite(_sample_name, frame)
            count+=1
            _print_msg = True

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if Web_cam_flag:
        del(Web_cam)

def prepare_training_data(train_data_path, _classifier=None):
    if _classifier is None:
        print("Classifier not specified for training. Exiting...")
        exit()

    faces = []
    labels = []

    dirs = os.listdir(train_data_path)

    for dir_name in dirs:
        if not dir_name.startswith("E"):
            continue;

        label = int(dir_name.replace("E", ""))

        subject_dir_path = train_data_path + os.sep + dir_name
        subject_images = os.listdir(subject_dir_path)

        for image_name in subject_images:
            if image_name.startswith(".") and ".jpg" not in image_name:
                continue;
            image_path = subject_dir_path + os.sep + image_name
            image = cv2.imread(image_path)
            cv2.imshow("Training on Image...", image)
            cv2.waitKey(100)

            face, rect = detect_face(image, _classifier)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    return faces, labels


def predict(test_img, _classifier):
    if _classifier is None:
        print("Classifier not specified for prediction. Exiting...")
        exit()

    face, rect = detect_face(test_img, _classifier)

    if face is None or face.all() is None:
        return None, None

    label, confidence = face_recognizer.predict(face)
    return str(label), rect


if __name__ == "__main__":
    # Configuration for face_recognition
    CURRENT_DIR = os.getcwd()
    DEFAULT_SAMPLE_COUNT = 10
    TRAIN_PREFIX = "E"
    IMAGE_FORMAT = "jpg"
    TRAINING_DIR = CURRENT_DIR + os.sep + "training_dir"
    OPENCV_DIR = CURRENT_DIR + os.sep + "opencv-files"
    CLASSIFIER = OPENCV_DIR + os.sep + "haarcascade_frontalface_default.xml"
    FACE_RECT_COLOR = (0, 255, 0)
    FACE_RECT_BORDER_THICKNESS = 2
    FACE_ID_TEXT = cv2.FONT_HERSHEY_PLAIN
    FACE_ID_TEXT_SCALE = 1.5
    FACE_ID_TEXT_COLOR = (0, 255, 0)
    FACE_ID_TEXT_THICKNESS = 2
    TRAIN_FLAG = True
    subjects={'1009991':"Anil"}

    # get web camera handle
    web_cam_handler = cv2.VideoCapture(0)

    #create a face recognition engine, LBPH in our case
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    process_frame = True
    while True:
        if TRAIN_FLAG:
            # Prepare training data to face_recognition engine
            faces, labels = prepare_training_data(TRAINING_DIR,
                                                  CLASSIFIER)
            # Train the face_recognition engine
            face_recognizer.train(faces, np.array(labels))

            TRAIN_FLAG = False

        # Run face_recognition engine to identify faces
        ret, frame = web_cam_handler.read()
        if process_frame:
            predicted_label, face_rect = predict(frame, CLASSIFIER)
            if predicted_label in subjects.keys():
                predicted_label = subjects[predicted_label]
            else:
                predicted_label = "New Face"
            if isinstance(face_rect, np.ndarray):
                print("Predicted face={}. Press '{}' to {}".format(predicted_label,'q','Quit'))
                draw_rectangle(frame, face_rect,
                               FACE_RECT_BORDER_THICKNESS,
                               FACE_RECT_COLOR)
                draw_text(frame, predicted_label, face_rect[0],
                          face_rect[1]-5, FACE_ID_TEXT,
                          FACE_ID_TEXT_SCALE, FACE_ID_TEXT_THICKNESS,
                          FACE_ID_TEXT_COLOR)
            cv2.imshow("Webcam output", frame)
        process_frame = not process_frame

        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # Hit 't' on the keyboard to give sample training input!
        if cv2.waitKey(1) & 0xFF == ord('t'):
            _name = input("Enter name of subject: ")
            uid = str(input("Enter subject's User ID for training : "))
            subjects[uid] = _name
            sample_collection(Web_cam = web_cam_handler,
                              train_dir=TRAINING_DIR, User_Id=uid,
                              samples=DEFAULT_SAMPLE_COUNT,
                              _pre=TRAIN_PREFIX, _format=IMAGE_FORMAT)
            TRAIN_FLAG = True

    # Release handle to the webcam
    web_cam_handler.release()

    # Delete all cv2 Windows
    cv2.destroyAllWindows()
