import os
import cv2
import numpy as np
import face_recognition
import pickle
import mediapipe as mp
from datetime import datetime, timedelta

import firebase_admin
from firebase_admin import credentials, storage, db

confidence_threshold = 0.5
# Initialize Firebase
cred = credentials.Certificate(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\accountkey.json")
firebase_admin.initialize_app(cred,
                             {
                                  'databaseURL':"https://face-verification-recognition-default-rtdb.firebaseio.com/",
                                 'storageBucket':"face-verification-recognition.appspot.com"
                               })

# Load Mediapipe models for face mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

unrecognized_dir = r"Unrecognized_Faces"

# Create the directory if it doesn't exist
if not os.path.exists(unrecognized_dir):
    os.makedirs(unrecognized_dir)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Set overlay position
overlay_x, overlay_y = 0, 0

# Load and verify each background image
def load_image(path):
    image = cv2.imread(path)
    if image is None:
        print(f"Error: Unable to load image at {path}")
        exit(1)
    return image

default_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background2.png")
recognized_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background3.png")
unrecognized_background = load_image(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background5.png")

# Load encoded images
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, peopleId = encodeListKnownWithIds
print(peopleId)

# Set timers
check_duration = 20  # seconds for recognition check
display_duration = 10  # seconds for showing final result

# Start time for face recognition check
start_time = datetime.now()
face_recognized = False  # Flag to track if a face was recognized
recognized_person_id = None  # Store the ID of recognized person

# Use Mediapipe Face Mesh
with (mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh):

    # 60-second checking loop
    while (datetime.now() - start_time).seconds < check_duration:
        ret, img = cap.read()
        if not ret:
            print("Error: Unable to read from webcam.")
            break

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect face locations and encodings
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        frame_resized = cv2.resize(img, (450, 450))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process face landmarks with Mediapipe
        results = face_mesh.process(img_rgb)

        # Draw face mesh if faces are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())

        # Check for known face matches
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex] and faceDis[matchIndex] < confidence_threshold and not face_recognized:
                face_recognized = True  # Mark that a known face was detected
                recognized_person_id = peopleId[matchIndex]
                print("Known face detected:", recognized_person_id)


        # Overlay the webcam image on the default background
        roi = default_background[overlay_y:overlay_y + frame_resized.shape[0], overlay_x:overlay_x + frame_resized.shape[1]]
        combination = cv2.addWeighted(roi, 0.5, frame_resized, 0.5, 0)
        default_background[overlay_y:overlay_y + frame_resized.shape[0], overlay_x:overlay_x + frame_resized.shape[1]] = combination

        # Display the default background during checking
        cv2.imshow('system', default_background)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    counter_file = "counter.txt"

    # Load previous counter value if it exists
    if os.path.exists(counter_file):
        with open(counter_file, "r") as file:
            img_counter = int(file.read().strip())
    else:
        img_counter = 0


    # After 60 seconds, show recognized/unrecognized background for 10 seconds
    end_time = datetime.now() + timedelta(seconds=display_duration)

    if face_recognized:
        final_background = recognized_background
        # Retrieve and print recognized person's information from Firebase
        studentInfo = db.reference(f'People/{recognized_person_id}').get()
        if studentInfo:
            initial_x, initial_y = 70, 200  # Starting position
            line_height = 20  # Reduced space between lines

            text_lines = [
                f"Name: {studentInfo['name']}",
                f"Profession: {studentInfo['profession']}",
                f"Major: {studentInfo['major']}",
                f"Favorite Team: {studentInfo['Favourite_team']}"
            ]

            font = cv2.FONT_HERSHEY_SIMPLEX  # Font type
            font_scale = 0.5  # Reduced font scale for smaller text
            font_color = (255, 0, 0)  # White color in BGR
            thickness = 1  # Thickness of the text

            for i, line in enumerate(text_lines):
                cv2.putText(final_background, line, (initial_x, initial_y + i * line_height), font, font_scale,
                            font_color, thickness, cv2.LINE_AA)

    else:
        img_name = os.path.join(unrecognized_dir, f"unrecognized_{img_counter}.png")

        # Save the image
        cv2.imwrite(img_name, img)
        print(f"Screenshot saved successfully in {img_name}")

        # Increment the image counter for the next screenshot
        img_counter += 1

        with open(counter_file, "w") as file:
            file.write(str(img_counter))

        final_background =unrecognized_background
    while datetime.now() <= end_time:
        # Display the final background
        cv2.imshow('system', final_background)
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()


