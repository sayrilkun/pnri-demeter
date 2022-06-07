from kivy.uix.screenmanager import Screen

class UploadDocScreen(Screen):
    def on_enter(self):

        for i in range(1,8):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""

    def eraser(self):
        for i in range(1,8):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""
