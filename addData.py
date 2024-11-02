import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
cred = credentials.Certificate(r"C:\Users\danie\PycharmProjects\daniel_Project_face_detection\accountkey.json")
firebase_admin.initialize_app(cred,
                              {'databaseURL':"https://face-verification-recognition-default-rtdb.firebaseio.com/"
                               })

ref = db.reference('People')

data = {
    "1":
        {
            "name":"John Mwaura",
            "profession":"Student",
            "major":"Engineering",
            "Favourite_team":"Manchester United"
        },

    "2":
        {
            "name": "Elvis Koros",
            "profession": "Worker",
            "major": "KPLC",
            "Favourite_team": "Arsenal"
        },
    "3":
        {
            "name": "Daniel Kadurha",
            "profession": "Student",
            "major": "Biomedical Engineering",
            "Favourite_team": "Real Madrid"
        },
    "4":
        {
            "name": "Daniel Ishimwe",
            "profession": "Student",
            "major": "Information Technology",
            "Favourite_team": "Barcelona"
        },
    "5":
        {
            "name": "Gidian Metamo",
            "profession": "Student",
            "major": "Electrical Engineering",
            "Favourite_team": "Chelsea",
        },
    "6":
        {
            "name": "Morris Kinyua",
            "profession": "Student",
            "major": "Information Technology",
            "Favourite_team": "Manchester United",
        },
}

for key, value in data.items():
    ref.child(key).set(value)