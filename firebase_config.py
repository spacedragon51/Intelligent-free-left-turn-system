# firebase_config.py
import firebase_config

firebase_config = {
    "apiKey": "YOUR_API_KEY",
    "authDomain": "traffic-app.firebaseapp.com",
    "databaseURL": "https://traffic-app.firebaseio.com",
    "storageBucket": "traffic-app.appspot.com"
}

firebase = firebase.initialize_app(firebase_config)
db = firebase.database()

def push_violation_to_app(violation):
    db.child("violations").push(violation)
    
def get_live_feed_url():
    return "https://stream.yourserver.com/camera_feed"