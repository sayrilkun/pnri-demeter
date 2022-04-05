from kivy.clock import mainthread
from kivy.lang import Builder
import threading
from kivymd.uix.dialog import MDDialog
from kivy.uix.boxlayout import BoxLayout
from kivymd.toast import toast


from kivymd.app import MDApp

KV = '''
#: import threading threading
Screen:
    BoxLayout:
                
        Button:
            text: 'Run Long Process'
            size_hint: None, None
            size: dp(150), dp(150)
            on_release: 
                app.spin_dialog()
                # app.spinner_toggle()
                app.long_process_thread()
                app.spin_dialog()
                # app.spinner_toggle()
        Button:
            text: 'pota'
            size_hint: None, None
            size: dp(150), dp(150)
            on_release: 
                
                app.spin_dialog()
                app.long_process_thread()
                

<ContentSpin>:
    orientation: "vertical"
    spacing: "12dp"
    size_hint_y: None
    height: "60dp"

    MDSpinner:
        id: spinner
        size_hint: None, None
        size: dp(46), dp(46)
        pos_hint: {'center_x': .5, 'center_y': .5}
        # active: check.active
'''
class ContentSpin(BoxLayout):
    pass

class Test(MDApp):
    dialog3= None

    def build(self):
        return Builder.load_string(KV)

    # @mainthread
    # def spinner_toggle(self):
    #     print('Spinner Toggle')
    #     app = self.get_running_app()
    #     if app.root.ids.spinner.active == False:
    #         app.root.ids.spinner.active = True
    #     else:
    #         app.root.ids.spinner.active = False
    @mainthread
    def spin_dialog(self):
        # app = self.get_running_app()
        # if app.root.ids.spinner.active == False:
        #     app.root.ids.spinner.active = True
        # else:
        #     app.root.ids.spinner.active = False
            
        if not self.dialog3:

            self.dialog3 = MDDialog(
                type="custom",
                content_cls=ContentSpin(),
            )
        self.dialog3.open()

    def long_process(self):
        # self.spin_dialog()

        for x in range(10000):
            print(x)
        self.dialog3.dismiss(force=True)
        toast("Document Saved Successfully")
        # self.spin_dialog()
        # self.spinner_toggle()


    def long_process_thread(self):
        # self.spin_dialog()

        # self.spinner_toggle()
        threading.Thread(target=(self.long_process)).start()



Test().run()