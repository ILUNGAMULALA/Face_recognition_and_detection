import os
import cv2
import numpy as np
import face_recognition
import pickle
import mediapipe as mp


from encoder import peopleId, encodeListKnownWithIds

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)

overlay_x = 0
overlay_y = 0
backgroundImg = cv2.imread(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\background.png")

#add the images to be changing in the list
#folderModePath = (r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\ressources")
#modePathList= os.listdir(folderModePath)
#imgModeList = []

#for path in modePathList:
#   imgModeList.append(cv2.imread(os.path.join(folderModePath)))"""

#load the encoded images
file = open("EncodeFile.p", "rb")
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, peopleId = encodeListKnownWithIds
print(peopleId)

#mediapipe code
with mp_face_mesh.FaceMesh(
    max_num_faces=1,  # Process only one face
    refine_landmarks=True,  # To enable iris landmarks
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while True:
        ret, img = cap.read()

        imgS = cv2.resize(img, (0,0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

        frame_resized = cv2.resize(img, (450, 450))

        img_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

        # Process the face landmarks
        results = face_mesh.process(img_rgb)

        # If faces are detected, draw the mesh on the image
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw the face mesh
                mp_drawing.draw_landmarks(
                    image=frame_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

                # Draw the iris
                mp_drawing.draw_landmarks(
                    image=frame_resized,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_IRISES,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                )

        # To put the image from the Webcam on the background one

        roi = backgroundImg[overlay_y:overlay_y + frame_resized.shape[0],
              overlay_x:overlay_x + frame_resized.shape[1]]
        combination = cv2.addWeighted(roi, 0.5, frame_resized, 0.5, 0)
        backgroundImg[overlay_y:overlay_y + frame_resized.shape[0],
        overlay_x:overlay_x + frame_resized.shape[1]] = combination

        for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            print("matches", matches)
            print("faceDist", faceDis)

            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                print("Known face detected")
                print(peopleId[matchIndex])

                frame_resized1 = cv2.imread(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\ressources\VERIFIED.png")
                overlay1_x = 450
                overlay1_y = 0
                roi = backgroundImg[overlay1_y:overlay1_y + frame_resized1.shape[0],
                      overlay1_x:overlay1_x + frame_resized1.shape[1]]
                combination1 = cv2.addWeighted(roi, 0.5, frame_resized1, 0.5, 0)
                backgroundImg[overlay1_y:overlay1_y + frame_resized1.shape[0],
                overlay1_x:overlay1_x + frame_resized1.shape[1]] = combination1
            else:
                print("No Known face detected")
                #print(peopleId[matchIndex])

                frame_resized1 = cv2.imread(
                    r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\ressources\NOT_ENROLLED.png")
                overlay1_x = 450
                overlay1_y = 0
                roi = backgroundImg[overlay1_y:overlay1_y + frame_resized1.shape[0],
                      overlay1_x:overlay1_x + frame_resized1.shape[1]]
                combination1 = cv2.addWeighted(roi, 0.5, frame_resized1, 0.5, 0)
                backgroundImg[overlay1_y:overlay1_y + frame_resized1.shape[0],
                overlay1_x:overlay1_x + frame_resized1.shape[1]] = combination1


        cv2.imshow('system', backgroundImg)

        if cv2.waitKey(1) & 0xFF == ord('1'):
            break

    cap.release()
    cv2.destroyAllWindows()
