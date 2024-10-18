﻿Here’s a comprehensive `README.md` file for my face detection and recognition project:

---

# Face Recognition and Detection System

This project is a real-time face detection and recognition system using **OpenCV**, **MediaPipe**, and **Face Recognition** libraries. It integrates **Firebase** to store user information and images, making it a robust system for recognizing faces, overlaying them on custom backgrounds, and retrieving user details from a database.

## Features
- Real-time face detection and recognition via webcam.
- Overlays detected faces on a custom background.
- Integrates with Firebase for image storage and user data management.
- Encodes known faces and matches them with faces detected in real-time.
- Displays user information for recognized faces.

## Benefits
- **Real-Time Performance**: The project processes and detects faces in real-time, making it suitable for real-world applications.
- **Secure Face Recognition**: Face encodings ensure that recognized faces are accurately matched with pre-encoded ones.
- **Firebase Integration**: Enables seamless cloud storage and database access for easy management and retrieval of face data.
- **Customizable Overlays**: Allows flexibility in how detected faces are presented on custom backgrounds.
- **Easy Scalability**: Additional faces and data can be easily added to the system.

## Libraries Used
1. **OpenCV**: Library for computer vision tasks, including face detection and image manipulation.
2. **MediaPipe**: Used for face landmark detection and iris tracking.
3. **Face Recognition**: For encoding and recognizing faces.
4. **Firebase**: Cloud storage and real-time database management.
5. **NumPy**: For numerical operations.
6. **Pickle**: For saving and loading encoded face data.
7. **Python Firestore**: For connecting and interacting with Firebase.

## Installation Guide

### Prerequisites
Ensure that you have **Python 3.x** installed. Install the required libraries by running the following:

```bash
pip install opencv-python opencv-contrib-python numpy face-recognition mediapipe firebase-admin
```

### Firebase Setup
1. Set up a Firebase project and enable the **Realtime Database** and **Storage** services.
2. Download the Firebase Admin SDK JSON key and place it in your project directory.
3. Add your Firebase credentials to the project in the `encoder.py` and `add_data.py` files.

### Project Files
The project consists of the following files:

- **main.py**: The main file where face detection, recognition, and background overlaying happens.
- **encoder.py**: Responsible for encoding face images and uploading them to Firebase storage.
- **add_data.py**: Populates the Firebase Realtime Database with user information.
- **background.png**: A customizable image used as the background where detected faces are displayed.

### File Structure

```
daniel_Project_face_detection/
│
├── list_of_students/
│   ├── background.png
│   └── pictures/
│       ├── John_Mwaura.jpg
│       ├── Elvis_Koros.jpg
│       └── ...
│
├── main.py
├── encoder.py
├── add_data.py
├── EncodeFile.p
└── accountkey.json
```

### Running the Project

#### 1. Encode Faces
Before running the face recognition system, you must encode the faces using `encoder.py`:

```bash
python encoder.py
```
This will process the images in the `list_of_students/pictures` folder and save the encodings in a `pickle` file (`EncodeFile.p`).

#### 2. Populate Firebase Database
To add user details into the Firebase Realtime Database, run the following:

```bash
python add_data.py
```
This will upload user information related to the faces.

#### 3. Run the Face Recognition System
Once the face encodings are ready, you can run the main program:

```bash
python main.py
```
This will open the webcam, detect faces, and overlay them on the background image. If a recognized face is detected, it will display user information on the console.

### Functionality Breakdown

#### `main.py`
- **Face Mesh Detection**: Uses **MediaPipe** to detect face landmarks and overlays the face mesh on the webcam feed.
- **Face Recognition**: The system uses **face_recognition** to match real-time webcam faces with pre-encoded faces.
- **Custom Background**: Detected faces are overlaid on a custom background using OpenCV's `addWeighted` function.
- **Matching Faces**: If a match is found, the system displays the corresponding user ID and details.

#### `encoder.py`
- **Face Encoding**: Encodes images stored in the `list_of_students/pictures` folder into facial feature vectors.
- **Firebase Upload**: The system uploads the images to Firebase storage for backup purposes.

#### `add_data.py`
- **Database Population**: Adds user information (such as name, profession, and favorite team) into Firebase's Realtime Database. The user details are linked to the facial encodings.

## Example Use Cases
1. **Attendance System**: Can be used in schools or workplaces to automatically detect and log people's attendance.
2. **Secure Access Control**: Can be adapted for access control systems where only authorized users are allowed access based on face recognition.
3. **Custom Video Conferencing Overlays**: The project can be modified for video calls where participants' faces are overlaid on different backgrounds in real-time.

## Future Improvements
- **Higher Accuracy with More Faces**: Expand the system to recognize more faces by adding more encoded images.
- **Cloud-Based Processing**: Shift face recognition computation to cloud services for better performance on low-end devices.
- **Web-Based Interface**: Integrate a web-based interface for easier user management and face data monitoring.

## Contact
For any questions or feedback, feel free to reach out via GitHub Issues or contact me directly.
