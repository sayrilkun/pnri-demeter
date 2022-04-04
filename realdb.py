import pyrebase

config = {
    "apiKey": "AIzaSyBH3WOpmUdPj0vGIpneswkW2CS8fFidlXw",
    "authDomain": "pnri-demeter.firebaseapp.com",
    "databaseURL": "https://pnri-demeter-default-rtdb.firebaseio.com",
    "projectId": "pnri-demeter",
    "storageBucket": "pnri-demeter.appspot.com",
    "messagingSenderId": "456214792415",
    "appId": "1:456214792415:web:773d7ea18f8ba214df816a",
    "measurementId": "G-00QH790MRG",
}

firebase = pyrebase.initialize_app(config)
db= firebase.database()

data = { 
    'Name': '{name}',
    'Date Acquired':'{dateAcq}',
    'Accession Origin': '{accOrg}',
    'Project': '{project}',
    'Project Leader': '{prjLdr}',
    'Other Details': '{otherDtls}',
    'Pollinium': '{pollinium}',
    'Retinaculum': '{retinaculum}',
    'Translator': '{translator}',
    'Caudicle Bulb Diameter': '{caudicle}',
    'img_url' : '{img_url}',
    'file_url' : '{file_url}',
    'qr_url': '{qr_url}',
    'scan_id' : '{scan_id}'
    }

# db.child("Hoya MEME").child("Hoya curtis").set(data)
# user = db.child("Hoya").get()
# print(user.val()) # users
# all_user_ids = db.child("Hoya").shallow().get()
# print(all_user_ids.val())
# result = all_user_ids.val()
# for i in range(len(result)):
#     name = result
#     print(name)

# all_user_ids = db.child("Hoya").child("Hoya curtis").child("img_url").shallow().get()
# print(all_user_ids.val())

# all_users = db.child("Hoya").child("Hoya curtis").get()
# for user in all_users.each():
#     print(user.key()) # Morty
#     # print(user.val()) # {name": "Mortimer 'Morty' Smith"}

# all_users = db.child("Hoya").child("Hota ming mong").get()
# for user in all_users.each():
#     print(user.key()) # Morty
#     print(user.val()) # {name": "Mortimer 'Morty' Smith"}

single_doc = db.child("Hoya").child("Hota ming mong")
img_url = single_doc.child("img_url").get()
potek = "'"+img_url.val()+"'"
print(potek)