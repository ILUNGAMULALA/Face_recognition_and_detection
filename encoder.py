import cv2
import face_recognition
import pickle
import os
import firebase_admin
from firebase_admin import credentials, storage

from firebase_admin import db
from scipy.constants import blob

cred = credentials.Certificate(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\accountkey.json")
firebase_admin.initialize_app(cred,
                             {
                                  'databaseURL':"https://face-verification-recognition-default-rtdb.firebaseio.com/",                                  'storageBucket':"face-verification-recognition.appspot.com"

                               })



folderPath= (r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\pictures")
#databasePath = (r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\list_of_students\database")
pathList = os.listdir(folderPath)
print(pathList)
imgList=[]
peopleId = []

for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    #print(os.path.splitext(path)[0])
    peopleId.append(os.path.splitext(path)[0])

    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


print(peopleId)

def findEncondings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

print("Enconding started...")
encodeListKnown = findEncondings(imgList)
encodeListKnownWithIds = [encodeListKnown, peopleId]
print("Enconding complete...")

file = open("EncodeFile.p", "wb")
pickle.dump(encodeListKnownWithIds, file)
file.close()

print("file saved successfully")
