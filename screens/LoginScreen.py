from kivy.uix.screenmanager import Screen
from yutils.pyre import db
from kivymd.toast import toast

class LoginScreen(Screen):
     

    def on_leave(self):
        self.ids.username.text = ''
        self.ids.password.text = ''