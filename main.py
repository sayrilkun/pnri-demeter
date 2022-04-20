import cv2
import firebase_admin
import numpy as np
import random
import os
import webbrowser
import tkinter as tk
import pyrebase
import qrcode
import string
import threading

import utils.frame_capture as frame_capture
import utils.frame_draw as frame_draw
# import utils.camruler as camruler

from kivy.clock import mainthread
from kivymd.toast import toast
from datetime import datetime
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.button import MDIconButton, MDFloatingActionButton
from tkinter import filedialog
from kivy.properties import StringProperty

from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from pyzbar.pyzbar import decode
from kivymd.uix.list import IRightBodyTouch, OneLineListItem, OneLineIconListItem, OneLineAvatarIconListItem
from kivymd.uix.list import IconLeftWidget
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase
from kivymd.utils import asynckivy
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton
from kivymd.uix.dialog import MDDialog
from kivy.uix.image import AsyncImage
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.label import MDLabel
from kivy.uix.boxlayout import BoxLayout


Window.size=(400,700)

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
storage= firebase.storage()
db= firebase.database()

class ContentSpin(BoxLayout):
    pass
class Content(BoxLayout):
    pass

class KivyCamera(Image):
    pass

class ImageButton(ButtonBehavior):
    pass

class IconLeftSampleWidget(IRightBodyTouch, MDIconButton):
    pass

class OneLineIcon(OneLineAvatarIconListItem):
    pass

class OneLine(OneLineListItem):
    divider = None
class LoginScreen(Screen):
    pass
class MenuScreen(Screen):
    pass

class CameraScreen(Screen):
    pass

    # def on_enter(self):
    #     x=DemoApp()
    #     cam = self.ids.camie
    #     self.clock_event = Clock.schedule_interval(x.object_detection, 1.0 /60)
    #     cam.capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    # def on_leave(self, *args):
    #     cam = self.ids.camie
    #     cam.capture.release()
    #     cam.clear_widgets()
        # self.clock_event.cancel()

class ImageScreen(Screen):
    pass

class CollectionsScreen(Screen):
    pass

class ScannerScreen(Screen):
    def on_leave(self, *args):
        cam = self.ids.cam
        cam.capture.release()
        cam.clear_widgets()
        # self.clock_event.cancel()

class QRScreen(Screen):
    # self.help.transition.direction = 'left'
    def on_pre_enter(self):
    #     x=DemoApp()
    #     x.on_qr()
        myDate = self.ids.forem.text
        self.ids.link.add_widget(
            MDRaisedButton( text = "Open link",
            on_press = lambda x: webbrowser.open(myDate))
        )     

class GeneratorScreen(Screen):
    pass

class HelpScreen(Screen):
    pass

class SettingsScreen(Screen):
    pass

class SingleDocScreen(Screen):
    pass


class UploadDocScreen(Screen):
    def on_enter(self):
        for i in range(1,13):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""

    def eraser(self):
        for i in range(1,13):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""
        
class Tab(MDFloatLayout, MDTabsBase):
    pass

class DemoApp(MDApp):

###################################################################
# COLOR SCHEMES
###################################################################
    purple = 56/255,40/255,81/255,1
    dark_green = 43/255, 172/255, 127/255, 1
    light_green =  197/255, 230/255, 127/255, 1
    light_green2 = 27/255, 229/255, 127/255, 1
    light_green3 = 135/255, 230/255, 127/255, 1
    light_green4 = 193/255, 225/255, 193/255, 1
    dark_green2 = 37/255, 160/255, 127/255, 1
    dark1 = 143/255, 188/255, 143/255,1
    light1 = 60/255, 179/255, 113/255, 1
    dark2 = 46/255, 139/255, 87/255, 1

###################################################################
# DIALOGS
###################################################################
    dialog2= None
    dialog3= None
    dialog4= None
    dialog6= None


    def show_no_doc_dialog(self):
        if not self.dialog2:
            self.dialog2 = MDDialog(
                text= "File does not exist!",
                buttons=[
                    MDRaisedButton(text="OK", 
                    on_press = lambda x :self.dialog2.dismiss(force=True)
                    )
                ],
            )
        self.dialog2.open()

    def show_simple_dialog(self):
        if not self.dialog3:
            self.dialog3 = MDDialog(
                type="custom",
                content_cls=Content(),
            )
        self.dialog3.open()

    def delete_dialog(self):
        if not self.dialog4:
            self.dialog4 = MDDialog(
                text= "Are you sure you want to delete?",
                buttons=[
                    MDFlatButton(
                        text = 'Cancel',
                        on_press = lambda x: self.dialog4.dismiss(force=True)),
                    MDRaisedButton(
                        text="OK", 
                        on_press = lambda x : self.delete_doc(),
                        on_release = lambda x: self.dialog4.dismiss(force=True)

                    )
                ],
            )
        self.dialog4.open()

    @mainthread
    def spin_dialog(self):
        if not self.dialog6:
            self.dialog6 = MDDialog(
                type="custom",
                content_cls=ContentSpin(),
            )
        self.dialog6.open()

###################################################################
# SWITCH SCREEN
###################################################################
    def swtchScrn(self,*args):
        self.search_callback()
        self.help.current = 'collections'
        self.help.transition.direction = 'right'


    def swtchScreen(self,screen,*args):
        # self.refresh_callback()
        self.help.current = screen
        self.help.transition.direction = 'right'

###################################################################
# DELETE DOCUMENT
###################################################################
    def delete_doc(self):
        doc= self.help.get_screen('singledoc').ids.species.title
        db.child("Hoya").child(doc).remove()
        self.swtchScrn()


###################################################################
# GENERATE ENCRYPTED SCAN ID
###################################################################
    def generate(self):
        S = 8  # number of characters in the string.  
        # call random.choices() string module to find the string in Uppercase + numeric data.  
        ran = 'ac&@%!'+''.join(random.choices(string.ascii_uppercase + string.digits, k = S))    
        x= str(ran)
        return x


###################################################################
# GENERATE QR CODE
###################################################################
    def add_qr(self, name, scan_id):
        input_data = scan_id
        #Creating an instance of qrcode
        qr = qrcode.QRCode(
                version=1,
                box_size=10,
                border=5)
        qr.add_data(input_data)
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        filename = f'qr_codes/{name}_qr.png'
        img.save(filename)
        storage.child(f"{name}/{name}_qr").put(filename)
        qr_url = storage.child(f"{name}/{name}_qr").get_url(None)
        # print(qr_url)
        return qr_url

###################################################################
# LOG IN USER
###################################################################       
    def sign_in(self):
        username = self.help.get_screen('login').ids.username.text
        password = self.help.get_screen('login').ids.password.text

        if username == 'admin' and password == '12345':
            self.help.current = 'menu'
            self.help.transition.direction = 'right'
        
        else:
            self.help.get_screen('login').ids.status.text = 'Invalid credentials. Please try again.'

###################################################################
# GET DOCUMENT LISTS FROM DATABASE
###################################################################
    def search_list(self):
        async def search_list():

            search=self.help.get_screen('collections').ids.search.text
            all_docs = db.child("Hoya").get()
            for i in all_docs.each():
                name = i.key()
                if search in name:
                    await asynckivy.sleep(0)
                    self.help.get_screen('collections').ids.box.add_widget(
                        OneLineIcon(text= f'{name}',
                        on_release = lambda y: self.spin_dialog(),
                        on_press= lambda x, value_for_pass=name: self.passValue_thread(value_for_pass),

                        ))
        asynckivy.start(search_list())
    


    def search_callback(self, *args):
        '''A method that updates the state of your application
        while the spinner remains on the screen.'''

        def refresh_callback(interval):
            self.help.get_screen('collections').ids.box.clear_widgets()
            self.search_list()
            self.help.get_screen('collections').ids.refresh_layout.refresh_done()
            self.tick = 0

        Clock.schedule_once(refresh_callback, 1)

###################################################################
# GET DATA FOR EACH DOCUMENT
###################################################################

    def passValue_thread(self,*args):
        threading.Thread(target=self.passValue, args = args).start()

    def passValue(self, *args):


        args_str = ','.join(map(str,args))

        icon = 'https://firebasestorage.googleapis.com/v0/b/pnri-demeter.appspot.com/o/flower.png?alt=media&token=3553abca-251f-42a3-b939-5d8eefc10a9a'
        passportData = ['Name','Date of Acquisition', 'Accession Origin', 'Project', 'Project Leader', 'Other Detals']
        morphology = ['Pollinium', 'Retinaculum', 'Caudicle Bulb Diameter', 'Translator']

        img_url = db.child("Hoya").child(args_str).child("urls").child("img_url").get().val()
        qr_url= db.child("Hoya").child(args_str).child("urls").child("qr_url").get().val()
        file_url= db.child("Hoya").child(args_str).child("urls").child("file_url").get().val()

        screen2 = self.help.get_screen('singledoc')
        screen2.ids.datas.clear_widgets()
        screen2.ids.dataso.clear_widgets()


        def open(url):
            if url is None or url == '':
                self.show_no_doc_dialog()

            else:
                webbrowser.open(url)

        screen2.ids.imag.add_widget(
            MDRoundFlatButton(
                text = "View IMG",
                on_press = lambda x : open(img_url)
            )
        )
        screen2.ids.file.add_widget(
            MDRoundFlatButton(
                text = "View PDF",
                on_press = lambda x : open(file_url)
            )
        )
        screen2.ids.qr.add_widget(
            MDRaisedButton(
                text = "Save QR Code",
                on_press = lambda x : open(qr_url)
            )
        )

        if img_url is None or img_url == '':
            screen2.ids.img_url.source = icon

        else:
            screen2.ids.img_url.source = img_url

        if qr_url is None or qr_url == '':
            screen2.ids.qr_url.source = icon

        else:
            screen2.ids.qr_url.source = qr_url


        screen2.ids['species'].title = args_str
        # screen2.ids['datos'].text = datos
        # screen2.ids['header'].text = f"Morphometric Analysis of {args_str}"



        # for complete documents / to separeate passport data and morphology
        morphology = db.child("Hoya").child(args_str).child("Morphology").get()
        for datas in morphology.each():
            screen2.ids.dataso.add_widget(
                OneLine(
                    text=f"{datas.key()} : {datas.val()}"
                    # halign="center"
                )
            )
        
        passport_data = db.child("Hoya").child(args_str).child("Passport Data").get()
        for datas in passport_data.each():
            screen2.ids.datas.add_widget(
                OneLine(
                    text=f"{datas.key()} : {datas.val()}"
                    # halign="center"
                )
            )
        self.help.current = 'singledoc'    
        self.help.transition.direction = 'left'   
        self.dialog6.dismiss(force=True)

###################################################################
# UPLOAD DOCUMENT
###################################################################
    def upload_thread(self):
        threading.Thread(target=(self.upload)).start()

    def upload(self):
        
        # self.show_donot_dialog()

        input_fields = ['name','dateAcq','accOrg','project','prjLdr','otherDtls',
                        'pollinium','retinaculum','translator', 'caudicle', 'image', 'file']
            
        name = self.help.get_screen('uploaddoc').ids.input_1.text
        dateAcq = self.help.get_screen('uploaddoc').ids.input_2.text
        accOrg = self.help.get_screen('uploaddoc').ids.input_3.text
        project = self.help.get_screen('uploaddoc').ids.input_4.text
        prjLdr = self.help.get_screen('uploaddoc').ids.input_5.text
        otherDtls = self.help.get_screen('uploaddoc').ids.input_6.text
        pollinium = self.help.get_screen('uploaddoc').ids.input_7.text
        retinaculum = self.help.get_screen('uploaddoc').ids.input_8.text
        translator = self.help.get_screen('uploaddoc').ids.input_9.text
        caudicle = self.help.get_screen('uploaddoc').ids.input_10.text
        image = self.help.get_screen('uploaddoc').ids.input_11.text
        file = self.help.get_screen('uploaddoc').ids.input_12.text

        if image == "":
            img_url = ""
        else:
            storage.child(f"{name}/{name}_image").put(image)
            img_url = storage.child(f"{name}/{name}_image").get_url(None)
        
        if file == "":
            file_url = ""
        else:
            storage.child(f"{name}/{name}_file").put(file)
            file_url = storage.child(f"{name}/{name}_file").get_url(None)

        scan_id = self.generate()

        qr_url = self.add_qr(name, scan_id)

        data = { 
            "Passport Data": {
                'Name': f'{name}',
                'Date Acquired':f'{dateAcq}',
                'Accession Origin': f'{accOrg}',
                'Project': f'{project}',
                'Project Leader': f'{prjLdr}',
                'Other Details': f'{otherDtls}',
            },
            "Morphology":{
                'Pollinium': f'{pollinium}',
                'Retinaculum': f'{retinaculum}',
                'Translator': f'{translator}',
                'Caudicle Bulb Diameter': f'{caudicle}',
            },
            "urls":{
                'img_url' : f'{img_url}',
                'file_url' : f'{file_url}',
                'qr_url': f'{qr_url}',
                'scan_id' : f'{scan_id}'
            }
            }

        db.child('Hoya').child(f'{name}').set(data)

        self.dialog6.dismiss(force=True)
        toast("Document Saved Successfully")
        self.swtchScrn()



    def add_img(self):
        root = tk.Tk()
        root.withdraw()
        image = filedialog.askopenfilename()
        print(image)
        # print(file)
        self.help.get_screen('uploaddoc').ids.input_11.text = image

    def add_file(self):
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename()
        print(file)
        # print(file)
        self.help.get_screen('uploaddoc').ids.input_12.text = file

###################################################################
# QR CODE SCANNER
###################################################################
    def show_camscanner(self):
        cam = self.help.get_screen('scanner').ids.cam
        self.clock_event = Clock.schedule_interval(self.update, 1.0 /30)
        cam.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)

    def update(self, dt):
        cam = self.help.get_screen('scanner').ids.cam
        ret, frame = cam.capture.read()
        # duts = []
        if ret:
            # cv2.putText(frame, 'dfgd', (50, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            for barcode in decode(frame):
                
                myData = barcode.data.decode('utf-8')
                sub = 'ac&@%!'
                if sub in myData:
                    hey = db.child("Hoya").order_by_child("scan_id").equal_to(myData).get()
                    for user in hey.each():
                        self.help.get_screen('qr').ids.forem.text = user.key()  
                else:
                    self.help.get_screen('qr').ids.forem.text = "NO DOCUMENT AVAILABLE"
                # qr_ref = db.collection('Hoya')
                # docs = qr_ref.where('scan_id', '==', f'{myData}').get()
                # for doc in docs:
                #     self.help.get_screen('qr').ids.forem.text = doc.id
                self.help.current = 'qr'
                # webbrowser.open(myData)
                pts = np.array([barcode.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 0, 255), 5)
                pts2 = barcode.rect
                cv2.putText(frame, myData,(pts2[0],pts2[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                            
            buf1 = cv2.flip(frame, -1)
            buf = buf1.tobytes()
            image_texture =Texture.create(
                size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            cam.texture = image_texture


###################################################################
# OPEN SCANNED DOCUMENT TO APP
###################################################################
    def oks_qr(self):
        myDate = self.help.get_screen('qr').ids.forem.text
        try:
            self.passValue(myDate)
        except Exception as e:
            toast("No Such Document!")

    def oks_thread(self):
        threading.Thread(target=(self.oks_qr)).start()

###################################################################
# OPEN SCANNED LINK TO WEBSITE
###################################################################
    def my_qr(self):
        myDate = self.ids.forem.text
        self.ids.link.add_widget(
            MDRaisedButton( text = "Open link",
            on_press = lambda x: webbrowser.open(myDate))
        )     


###################################################################
# CAMERA
###################################################################
    def capture(self):
        self.help.get_screen('image').ids.cap_img.source = 'captured_img/image.png'
        self.swtchScreen('image')

    def save_img(self):
        self.help.get_screen('uploaddoc').ids.input_11.text = 'captured_img/image.png'
        self.swtchScreen('uploaddoc') 


    def show_camie(self):
        # disabled cam, naglalag kasi
        cam = self.help.get_screen('camera').ids.camie
        self.clock_event = Clock.schedule_interval(self.object_detection, 1.0 /60)
        cam.capture = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
        self.help.current = 'camera'
        self.help.transition.direction = "left"

        
    def object_detection(self, dt):

        cam = self.help.get_screen('camera').ids.camie
        ret, frame = cam.capture.read()  
        if ret:


            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture =Texture.create(
                size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            cam.texture = image_texture
            self.help.get_screen('camera').ids.capture.add_widget(
                MDFloatingActionButton( 
                    text = "Capture", 
                    icon = 'camera-iris', 
                    md_bg_color="#f8d7e3", 
                    text_color = "#211c29",
                    on_press = lambda x: cv2.imwrite(f'captured_img/image.png', frame),
                    on_release = lambda x : self.capture())
            )
    def cam_start(self):
        camruler.start()

    def camruler_thread(self):
        threading.Thread(target=(self.cam_start)).start()
        # self.dialog6.dismiss(force=True)

    def on_start(self):
        self.search_list()
        # self.on_qr()

    def build(self):
        # screen =Screen()
        
        self.title='Demeter'
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "LightGreen"   

        self.help = Builder.load_file('main.kv')
        # screen.add_widget(self.help)
        return self.help

DemoApp().run()