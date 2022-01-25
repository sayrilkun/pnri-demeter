from kivy.lang import Builder
from kivymd.app import MDApp
from kivymd.uix.dialog import MDDialog
from kivymd.uix.button import MDFlatButton
from kivy.core.window import Window
# set window size
Window.size=(300,500)
class LoginApp(MDApp):
    dialog = None
    def build(self):
        # define theme colors
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "DeepPurple"
        self.theme_cls.accent_palette = "Teal"
        # load and return kv string
        return Builder.load_file('login.kv')
    
    def login(self):
        # check entered username and password
        if self.root.ids.user.text=='admin' and self.root.ids.password.text=='admin123':
            if not self.dialog:
                # create dialog
                self.dialog = MDDialog(
                    title="Log In",
                    text=f"Welcome {self.root.ids.user.text}!",
                    buttons=[
                        MDFlatButton(
                            text="Ok", text_color=self.theme_cls.primary_color,
                            on_release=self.close
                        ),
                    ],
                )
            # open and display dialog
            self.dialog.open()
        else:
            self.root.ids.status.text = 'Incorrect Credentials. Please Try Again.'
    def close(self, instance):
        # close dialog
        self.dialog.dismiss()
# run app    
LoginApp().run()