from kivymd.uix.dialog import MDDialog
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton


class ContentEdit(BoxLayout):
    pass

class ContentSpin(BoxLayout):
    pass

class Content(BoxLayout):
    pass

def show_no_doc_dialog(self):
    if not self.dialog2:
        self.dialog2 = MDDialog(
            text= "File does not exist!",
            buttons=[
                MDRaisedButton(text="OK", 
                on_press = lambda x :self.dialog2.dismiss(force=True)
                )
            ],
        )
    self.dialog2.open()
    
def show_simple_dialog(self):
    if not self.dialog3:
        self.dialog3 = MDDialog(
            type="custom",
            content_cls=Content(),
        )
    self.dialog3.open()

def delete_dialog(self):
    if not self.dialog4:
        self.dialog4 = MDDialog(
            text= "Are you sure you want to delete?",
            buttons=[
                MDFlatButton(
                    text = 'Cancel',
                    on_press = lambda x: self.dialog4.dismiss(force=True)),
                MDRaisedButton(
                    text="OK", 
                    on_press = lambda x : self.delete_doc(),
                    on_release = lambda x: self.dialog4.dismiss(force=True)

                )
            ],
        )
    self.dialog4.open()

# @mainthread
def spin_dialog(self):
    if not self.dialog6:
        self.dialog6 = MDDialog(
            type="custom",
            content_cls=ContentSpin(),
        )
    self.dialog6.open()

def form_dialog(self):
    if not self.dialog:
        self.dialog = MDDialog(
            title="Add Data",
            type="custom",
            content_cls=ContentEdit(),
            buttons=[
                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press = lambda x : self.dialog.dismiss(force=True)

                ),
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    text_color=self.theme_cls.primary_color,
                    on_press = lambda x : self.add_textfields()
                ),
            ],
        )
    self.dialog.open()

