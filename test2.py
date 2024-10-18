import os
import cv2
import numpy as np
import face_recognition
import pickle
import mediapipe as mp

# Initialize MediaPipe and Face Recognition
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize camera
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Load background image
background_img_path = r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background.png"
backgroundImg = cv2.imread(background_img_path)

if backgroundImg is None:
    print(f"Error: Could not load background image from {background_img_path}")
    exit()

# Load the encoded face data
encoded_file_path = "EncodeFile.p"
try:
    with open(encoded_file_path, "rb") as file:
        encodeListKnownWithIds = pickle.load(file)
        encodeListKnown, peopleId = encodeListKnownWithIds
        print("Loaded known faces:", peopleId)
except FileNotFoundError:
    print(f"Error: File '{encoded_file_path}' not found.")
    exit()

overlay_x = 0
overlay_y = 0

# MediaPipe FaceMesh
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Process only one face
        refine_landmarks=True,  # To enable iris landmarks
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret, img = cap.read()

        if not ret:
            print("Error: Failed to capture image.")
            break

        # Resize image for faster processing
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Detect faces in the current frame
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        # Resize original frame and convert to RGB for MediaPipe
        frame_resized = cv2.resize(img, (450, 450))
        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process the face landmarks with MediaPipe
        results = face_mesh.process(img_rgb)

        # Draw face landmarks if detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh and iris
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

        # Overlay the webcam frame onto the background
        if backgroundImg is not None:
            roi = backgroundImg[overlay_y:overlay_y + frame_resized.shape[0],
                  overlay_x:overlay_x + frame_resized.shape[1]]

            # Add weighted overlay
            combination = cv2.addWeighted(roi, 0.5, frame_resized, 0.5, 0)
            backgroundImg[overlay_y:overlay_y + frame_resized.shape[0],
            overlay_x:overlay_x + frame_resized.shape[1]] = combination

        # Face recognition for current frame
        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                print("Known face detected:", peopleId[matchIndex])

        # Display the background image with overlay
        cv2.imshow('Face Detection System', backgroundImg)

        # Break loop with '1' key
        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
