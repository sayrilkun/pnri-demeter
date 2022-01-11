Integrated Image Recognition in the Morphometric Analysis of the Hoya Pollinaria for Botanical Classification

Installation Requirements:

Python 3.9.6
Setup Python in VSCode:
https://code.visualstudio.com/docs/python/python-tutorial

Git:
https://git-scm.com/

Libraries to install: pip install <library>
  
firebase_admin (for backend database, cloud firestore)
  
pyrebase4 (for backend, cloud storage)
  
kivy (for frontend)
kivymd (for frontend)
opencv-python (for image processing)
pyzbar (for qr code scanner)
tkinter (file dialog)
qrcode (qr generator)

Documentations:

Python:
https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

Firestore Database:
https://firebase.google.com/docs/firestore

Kivy:
https://kivy.org/doc/stable/

KivyMD
https://kivymd.readthedocs.io/en/latest/

OpenCV
https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html

Material Icon Used:
https://icons8.com/

TO DO:
Data Collection -> Augmentation -> Segmentation -> Classification -> Morphometric Analysis -> Model
On-screen Camera/Scanner
Edit/Delete data
Search DB
Authentication Screen
Help
Settings

Finished:
Setup database and storage
setup UI
add camera
add qr scanner
add qr generator
upload image, data, file
save image, data, file
refresh list

Issues:
camera lag after reopen
search list
view image/ pdf
save qr code
no spinner on file upload
