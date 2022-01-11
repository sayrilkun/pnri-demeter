
import firebase_admin
import pyrebase
from firebase_admin import credentials
from firebase_admin import firestore
from firebase_admin import storage as admin_storage
import numpy as np
import cv2
from base64 import b64encode, b64decode

config = {
    "apiKey": "AIzaSyBH3WOpmUdPj0vGIpneswkW2CS8fFidlXw",
    "authDomain": "pnri-demeter.firebaseapp.com",
    "databaseURL": "https://pnri-demeter-default-rtdb.firebaseio.com",
    "projectId": "pnri-demeter",
    "storageBucket": "pnri-demeter.appspot.com",
    "messagingSenderId": "456214792415",
    "appId": "1:456214792415:web:773d7ea18f8ba214df816a",
    "measurementId": "G-00QH790MRG",
    # "serviceAccount": "serviceAccountKey.json"
}

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {"storageBucket": "pnri-demeter.appspot.com"})

db= firestore.client()

firebase = pyrebase.initialize_app(config)
storage= firebase.storage()

a=454

data = {
    'Pollinium' : '0.35mm x 0.17mm',
    'Retinacilum' : f'{a} x 0.16 x 0.08',
    'Translator' : f'{a-9} x 0.03 deep',
    'Caudicle Bulb Diameter' : '0.06' 
}

# auto ID
def add_data(data):
    db.collection('Hoya Species').add(data)

# set ID
def set_data(data):
    db.collection('Hoya Species').document('Hoya pldt').collection('url').document('file').set(data)

# add_data(data)
set_data(data)
# get all documents     
def get_data():
    docs = db.collection('Hoya Species').stream()

    for doc in docs:
        print(f'{doc.id} => {doc.to_dict()}')

# get single document
def get_SingleDocument():

    a='qjE4l8mLC70xTSRqzztj'
    
    doc_ref = db.collection('Hoya Species').document(a)

    doc = doc_ref.get()
    if doc.exists:
        # bal = u'{}'.format(get_bal.to_dict()['Balance'])
        print(f'Document data: {doc.to_dict()}')
    else:
        print(u'No such document!')

# get_SingleDocument()

def get_SingleData():
    a= 'Hoya chongke'
    doc_ref = db.collection('Hoya Species').document('Hoya chongke')
    pol = doc_ref.get(field_paths={'Pollinium'}).to_dict().get('Pollinium')
    # pol = get_pol.get('Pollinium')
    # pol = u'{}'.format(get_pol.to_dict()['Pollinium'])
    print(pol)

# get_SingleData()

def upload_image():
    storage.child("image2").put("image2.jpg")
    # print(storage.child("image2").get_url(None))

# upload_image()

# print(storage.child("image2").get_url(None))

def display_image():
    # with open("image2.jpg", 'rb') as f:
    #     data = f.read()
    # str = b64encode(data).decode('UTF-8')
    # storage.child("image").set({"data": str})


    # Retrieve image from Firebase
    retrieved = storage.child("image").get().val()
    retrData = retrieved["data"]
    JPEG = b64decode(retrData)

    image = cv2.imdecode(np.frombuffer(JPEG,dtype=np.uint8), cv2.IMREAD_COLOR)
    cv2.imwrite('result.jpg',image)

# display_image()

def tryers():
    morphology = ['Pollinium', 'Retinaculum', 'Caudicle Bulb Diameter', 'Translator']
    doc_ref = db.collection('Hoya Species').document('Hoya chongke')
    for i in range(len(morphology)):
        pol = doc_ref.get(field_paths={'Pollinium'}).to_dict().get('Pollinium')
        print(morphology[i])

# tryers()

def delete():
    db.collection('Hoya Species').document('Hoya chutu').delete()

# delete()

def delete_storage():
    # admin = firebase_admin.initialize_app(cred, {
    #   "storageBucket": "pnri-demeter.appspot.com"})
    bucket = admin_storage.bucket()
    blob = bucket.blob("Hoya kumu/")
    print(blob)
    blob.deleteFiles()
    # storage.delete(str("Hoya kumu/Hoya kumu_file.pdf"))

# delete_storage()

def delete_storager():
    # bucket = admin_storage.bucket()
    # blob = bucket.blob("Hoya kumu/Hoya kumu_file.pdf")
    # print(blob)
    # blob.delete()
    storage.delete("Hoya kumu")

# delete_storager()