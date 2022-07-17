from kivy.uix.screenmanager import Screen

class UploadDocScreen(Screen):
    def on_leave(self):

        for i in range(1,8):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""
        for i in range(11,14):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""

    def eraser(self):
        for i in range(1,8):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""
        for i in range(11,14):
            self.manager.get_screen('uploaddoc').ids[f'input_{i}'].text = ""