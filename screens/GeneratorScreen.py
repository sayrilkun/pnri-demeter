from kivy.uix.screenmanager import Screen

class GeneratorScreen(Screen):
    def on_enter(self):
        self.ids.bonk.text = "BOBOX"
