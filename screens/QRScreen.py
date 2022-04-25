from kivy.uix.screenmanager import Screen
import webbrowser
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton


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