from kivy.uix.screenmanager import Screen
from yutils.pyre import db
from kivymd.toast import toast

class LoginScreen(Screen):
###################################################################
# LOG IN USER
###################################################################       
    def sign_in(self):
        # auth = {}
        auth = db.child("Auth").get().val()
        username = self.ids.username.text
        password = self.ids.password.text
        
        if username == auth['username'] and password == auth['password']:
            self.manager.current = 'menu'
            self.manager.transition.direction = 'right'
        
        else:
            toast('Invalid credentials. Please try again.')
    def on_leave(self):
        self.ids.username.text = ''
        self.ids.password.text = ''