from kivy.uix.screenmanager import Screen

class IdentifierScreen(Screen):
    def panget(self):
        self.help.get_screen('iden').ids.bonk.text = "BOBOX"

    def pang(self):
        print("agdads")