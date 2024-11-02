import os
import cv2
import numpy as np
import face_recognition
import pickle
import mediapipe as mp
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, storage, db

# Firebase and model configuration
confidence_threshold = 0.5
cred = credentials.Certificate(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\accountkey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://face-verification-recognition-default-rtdb.firebaseio.com/",
    'storageBucket': "face-verification-recognition.appspot.com"
})

# Mediapipe models for face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

unrecognized_dir = r"Unrecognized_Faces"
os.makedirs(unrecognized_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
overlay_x, overlay_y = 0, 0

# List of the functions that are used in my code

def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at {path}")
        exit(1)
    return image

def display_text_lines(background, text_lines, start_x, start_y, font, font_scale, color, thickness, line_height):
    for i, line in enumerate(text_lines):
        cv2.putText(background, line, (start_x, start_y + i * line_height), font, font_scale, color, thickness, cv2.LINE_AA)

def save_unrecognized_face(image, counter):
    img_name = os.path.join(unrecognized_dir, f"unrecognized_{counter}.png")
    cv2.imwrite(img_name, image)
    print(f"Screenshot saved successfully in {img_name}")
    with open("counter.txt", "w") as file:
        file.write(str(counter + 1))

def load_counter():
    if os.path.exists("counter.txt"):
        with open("counter.txt", "r") as file:
            return int(file.read().strip())
    return 0

# Background images
default_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background2.png")
recognized_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background3.png")
unrecognized_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background5.png")

# Encoded images
with open("EncodeFile.p", "rb") as file:
    encodeListKnown, peopleId = pickle.load(file)
print(peopleId)

# Timer setup
check_duration = 20
display_duration = 10

# Face recognition and mesh detection loop
start_time = datetime.now()
face_recognized = False
recognized_person_id = None
img_counter = load_counter()

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while (datetime.now() - start_time).seconds < check_duration:
        ret, img = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        frame_resized = cv2.resize(img, (450, 450))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_resized, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )
                mp_drawing.draw_landmarks(
                    image=frame_resized, landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < confidence_threshold and not face_recognized:
                face_recognized = True
                recognized_person_id = peopleId[matchIndex]
                print("Known face detected:", recognized_person_id)

        roi = default_background[overlay_y:overlay_y + frame_resized.shape[0], overlay_x:overlay_x + frame_resized.shape[1]]
        combination = cv2.addWeighted(roi, 0.5, frame_resized, 0.5, 0)
        default_background[overlay_y:overlay_y + frame_resized.shape[0], overlay_x:overlay_x + frame_resized.shape[1]] = combination
        cv2.imshow('system', default_background)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    end_time = datetime.now() + timedelta(seconds=display_duration)
    final_background = recognized_background if face_recognized else unrecognized_background

    if face_recognized:
        studentInfo = db.reference(f'People/{recognized_person_id}').get()
        if studentInfo:
            text_lines = [
                f"Name: {studentInfo['name']}",
                f"Profession: {studentInfo['profession']}",
                f"Major: {studentInfo['major']}",
                f"Favorite Team: {studentInfo['Favourite_team']}"
            ]
            display_text_lines(final_background, text_lines, 70, 200, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, 20)
    else:
        save_unrecognized_face(img, img_counter)
        img_counter += 1

    while datetime.now() <= end_time:
        cv2.imshow('system', final_background)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

cap.release()
cv2.destroyAllWindows()
