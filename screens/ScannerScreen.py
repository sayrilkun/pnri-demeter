from kivy.uix.screenmanager import Screen

class ScannerScreen(Screen):
    def on_leave(self, *args):
        cam = self.ids.cam
        cam.capture.release()
        cam.clear_widgets()