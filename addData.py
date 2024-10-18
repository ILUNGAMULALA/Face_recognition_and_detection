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
            "Favourite_team":"Manchester United",
            "number_of_occurrence": 4,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },

    "2":
        {
            "name": "Elvis Koros",
            "profession": "Worker",
            "major": "KPLC",
            "Favourite_team": "Arsenal",
            "number_of_occurrence": 5,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },
    "3":
        {
            "name": "Daniel Kadurha",
            "profession": "Student",
            "major": "Biomedical Engineering",
            "Favourite_team": "Real Madrid",
            "number_of_occurrence": 5,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },
    "4":
        {
            "name": "Daniel Ishimwe",
            "profession": "Ongoing Student",
            "major": "Information Technology",
            "Favourite_team": "Barcelona",
            "number_of_occurrence": 6,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },
    "5":
        {
            "name": "Gidian Metamo",
            "profession": "Student",
            "major": "Electrical Engineering",
            "Favourite_team": "Chelsea",
            "number_of_occurrence": 4,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },
    "6":
        {
            "name": "Morris Kinyua",
            "profession": "Student",
            "major": "Information Technology",
            "Favourite_team": "Manchester United",
            "number_of_occurrence": 4,
            "Last_time_recorded": "2024-01-01 00:54:34"
        },
}

for key, value in data.items():
    ref.child(key).set(value)