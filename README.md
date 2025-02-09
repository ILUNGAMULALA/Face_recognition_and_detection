
```markdown
# Face Recognition and Verification System

This project implements a real-time face recognition and verification system using **OpenCV**, **MediaPipe**, **Firebase**, and **face_recognition** library. The system detects and verifies faces against a pre-defined database, saves unrecognized faces, and displays details for recognized individuals. It is intended to be used in environments that require face authentication for entry, such as schools, offices, or restricted areas.

## Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Project Structure](#project-structure)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Functionality Details](#functionality-details)
8. [Acknowledgments](#acknowledgments)

## Features

- **Face Detection**: Uses OpenCV and face_recognition to detect faces.
- **Face Verification**: Checks detected faces against a pre-defined list of known individuals.
- **Database Integration**: Integrates with Firebase to fetch and store user data.
- **Image Overlay**: Uses MediaPipe for face mesh detection and displays overlays on recognized faces.
- **Unrecognized Face Storage**: Saves snapshots of unrecognized faces for future reference.

## Technologies Used

- **OpenCV**: For image processing and video capture.
- **MediaPipe**: For face mesh detection and overlay.
- **Firebase**: Used for real-time database and cloud storage.
- **face_recognition**: For encoding and comparing facial features.
- **NumPy**: For efficient numerical calculations.

## Project Structure

```plaintext
.
├── accountkey.json              # Firebase credentials file
├── main.py                      # Main program for face recognition and verification
├── encoder.py                   # Generates encodings for known faces and uploads them to Firebase
├── addData.py                   # Adds sample data to Firebase
├── EncodeFile.p                 # Pickle file for storing face encodings and IDs
├── list_of_students/
│   ├── pictures/                # Directory of images of known individuals
│   ├── background2.png          # Background images for different system states
│   ├── background3.png
│   ├── background5.png
├── Unrecognized_Faces/          # Directory to save images of unrecognized faces
└── counter.txt                  # Counter file to index unrecognized face images
```

## Installation

### Prerequisites

- Python 3.7+
- Firebase account and Firebase credentials JSON file
- Camera device for capturing real-time video

### Dependencies

Install dependencies via pip:

```bash
pip install opencv-python numpy face-recognition mediapipe firebase-admin
```

### Setting Up Firebase

1. Create a Firebase project and enable **Realtime Database** and **Cloud Storage**.
2. Download the Firebase credentials JSON file (`accountkey.json`) and place it in the project directory.

## Configuration

### Firebase Configuration

In `main.py`, `encoder.py`, and `addData.py`, configure Firebase by initializing the app with your database URL and storage bucket:

```python
cred = credentials.Certificate(r"path_to_your/accountkey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://your-project-id-default-rtdb.firebaseio.com/",
    'storageBucket': "your-project-id.appspot.com"
})
```

### Directory Setup

- Ensure the directory `list_of_students/pictures` contains images of known individuals. Filenames should represent the unique ID for each individual.
- `Unrecognized_Faces` directory will be automatically created to save snapshots of unrecognized faces.

## Usage

### Step 1: Add Data to Firebase

Run `addData.py` to populate Firebase with sample user data.

```bash
python addData.py
```

### Step 2: Generate Face Encodings

Run `encoder.py` to encode images of known individuals and save encodings in `EncodeFile.p`.

```bash
python encoder.py
```

### Step 3: Run the Face Recognition System

Run `main.py` to start the real-time face recognition and verification system.

```bash
python main.py
```

## Functionality Details

### Main Components

- **Face Recognition and Verification**: The program captures frames from a webcam, detects faces, and matches them against stored encodings. If a face is recognized, it fetches user information from Firebase and overlays it on the display.
- **Face Mesh Overlay**: Uses MediaPipe to create a visual overlay on detected faces.
- **Saving Unrecognized Faces**: Saves images of unrecognized individuals to `Unrecognized_Faces` directory with a unique counter.

### Firebase Structure

Sample Firebase data structure for users:

```json
{
  "People": {
    "1": {
      "name": "John Mwaura",
      "profession": "Student",
      "major": "Engineering",
      "Favourite_team": "Manchester United"
    },
    "2": {
      "name": "Elvis Koros",
      "profession": "Worker",
      "major": "KPLC",
      "Favourite_team": "Arsenal"
    }
  }
}
```

### Overlay Display

The background image is updated based on face recognition results:
- **Recognized Faces**: Shows a recognized background with user information.
- **Unrecognized Faces**: Shows an unrecognized background and stores the face for future verification.

## Acknowledgments

This project is inspired by the need for a secure and efficient face verification system suitable for academic and corporate environments. Special thanks to the contributors of OpenCV, MediaPipe, and Firebase libraries.
```

This `README.md` provides a complete guide, from setup to functionality, making it easy for others to understand and use your project.