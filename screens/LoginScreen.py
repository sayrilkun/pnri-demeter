from kivy.uix.screenmanager import Screen

class LoginScreen(Screen):
###################################################################
# LOG IN USER
###################################################################       
    def sign_in(self):
        username = self.ids.username.text
        password = self.ids.password.text

        if username == 'admin' and password == '12345':
            self.manager.current = 'menu'
            self.manager.transition.direction = 'right'
        
        else:
            self.ids.status.text = 'Invalid credentials. Please try again.'
