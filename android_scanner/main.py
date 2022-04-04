import cv2
import numpy as np
import random
import webbrowser

from kivymd.uix.snackbar import Snackbar
from kivymd.uix.button import MDIconButton, MDFloatingActionButton
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

class KivyCamera(Image):
    pass

class MenuScreen(Screen):
    pass

class SettingsScreen(Screen):
    pass

class GeneratorScreen(Screen):
    pass

class HelpScreen(Screen):
    pass
class ScannerScreen(Screen):
    def on_leave(self, *args):
        cam = self.ids.cam
        cam.capture.release()

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

class DemoApp(MDApp):
    purple = 56/255,40/255,81/255,1

    def update(self, dt):
        cam = self.help.get_screen('scanner').ids.cam
        ret, frame = cam.capture.read()
        # duts = []
        if ret:
            # cv2.putText(frame, 'dfgd', (50, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            for barcode in decode(frame):
                
                myData = barcode.data.decode('utf-8')
                self.help.get_screen('qr').ids.forem.text = myData
                # qr_ref = db.collection('Hoya')
                # docs = qr_ref.where('scan_id', '==', f'{myData}').get()
                # for doc in docs:
                #     self.help.get_screen('qr').ids.forem.text = doc.id
                # self.help.current = 'qr'
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

    def show_cam(self):
        cam = self.help.get_screen('scanner').ids.cam
        self.clock_event = Clock.schedule_interval(self.update, 1.0 /30)
        cam.capture = cv2.VideoCapture(0,cv2.CAP_DSHOW)



    def build(self):
        # screen =Screen()
        
        self.title='Demeter'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"   

        self.help = Builder.load_file('main.kv')
        # screen.add_widget(self.help)
        return self.help

DemoApp().run()