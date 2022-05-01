from kivy.uix.screenmanager import Screen
from utils.pyre import db
from itertools import islice
from kivymd.utils import asynckivy
from kivymd.uix.list import OneLineAvatarIconListItem, OneLineListItem
from kivy.clock import Clock
from screens.SingleDocScreen import SingleDocScreen
import utils.dialog as dialog
import webbrowser
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton
import threading

class CollectionsScreen(Screen):
    pass

# class OneLine(OneLineListItem):
#     divider = None    
# class OneLineIcon(OneLineAvatarIconListItem):
#     pass
# class CollectionsScreen(Screen):
#     arr = [] 
#     paginated = ()
#     page_length = 0
#     page_number = 0
#     dialog6 = None

#     def on_pre_enter(self):
#             self.pop_array()
#             self.list_array()
        
#     def next_button(self):
#         self.page_number+=1
#         print(self.page_number)
#         if self.page_number < self.page_length:
#             self.list_callback()
#             self.ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
#             self.ids.page_numberad.text = f"{(self.page_number+1)*10} out of {len(self.arr)}"

#         else:
#             self.page_number-=1

#     def prev_button(self):
#         self.page_number-=1
#         print(self.page_number)
#         if self.page_number >= 0:
#             self.list_callback()
#             self.ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
#             self.ids.page_numberad.text = f"{(self.page_number+1)*10} of {len(self.arr)}"

#         else:
#             self.page_number+=1  

#     def pop_array(self):
#         all_docs = db.child("Hoya").get()
#         for i in all_docs.each():
#             self.arr.append(i.key())
#             print(i.key())
#         def chunk(it, size):
#             it = iter(it)
#             return iter(lambda: tuple(islice(it, size)), ())
#         self.paginated = list(chunk(self.arr,10))
#         self.page_length = len(self.paginated)
#         self.ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
#         self.ids.page_numberad.text = f"{(self.page_number+1)*10} of {len(self.arr)}"

#     def list_array(self):
#         async def search_list():
#             search=self.ids.search.text
#             for i in self.paginated[self.page_number]:
#                 if search in i:
#                     await asynckivy.sleep(0)
#                     self.ids.box.add_widget(
#                         OneLineIcon(text= f'{i}',
#                         # on_touch_down = lambda x: print("adfads")
#                         on_release = lambda y: dialog.spin_dialog(self),
#                         on_press= lambda x, value_for_pass=i: self.passValue_thread(value_for_pass),

#                         ))  
#         asynckivy.start(search_list())  

#     def search_array(self):
#         async def search_list():
#             search=self.ids.search.text
#             if search == '':
#                 self.list_callback()
#             else:    
#                 for i in self.arr:
#                     if search in i:
#                         await asynckivy.sleep(0)
#                         self.ids.box.add_widget(
#                             OneLineIcon(text= f'{i}',
#                             # on_touch_down = lambda x: print("adfads")
#                             on_release = lambda y: self.spin_dialog(self),
#                             on_press= lambda x, value_for_pass=i: self.passValue_thread(value_for_pass),

#                             ))
                        
#         asynckivy.start(search_list())    

#     def list_callback(self, *args):
#         '''A method that updates the state of your application
#         while the spinner remains on the screen.'''

#         def refresh_callback(interval):
#             self.ids.box.clear_widgets()
#             self.list_array()
#             self.ids.refresh_layout.refresh_done()
#             self.tick = 0

#         Clock.schedule_once(refresh_callback, 1)

#     def search_callback(self, *args):
#         '''A method that updates the state of your application
#         while the spinner remains on the screen.'''

#         def refresh_callback(interval):
#             self.ids.box.clear_widgets()
#             self.search_array()
#             self.ids.refresh_layout.refresh_done()
#             self.tick = 0

#         Clock.schedule_once(refresh_callback, 1)

#     def refresh_callback(self, *args):
#         '''A method that updates the state of your application
#         while the spinner remains on the screen.'''

#         def refresh_callback(interval):
#             self.ids.box.clear_widgets()
#             self.arr = []
#             self.pop_array()
#             self.list_array()
#             self.ids.refresh_layout.refresh_done()
#             self.tick = 0

#         Clock.schedule_once(refresh_callback, 1)

#     def passValue_thread(self,*args):
#         threading.Thread(target=self.passValue, args = args).start()

#     def passValue(self, *args):


#         args_str = ','.join(map(str,args))

#         icon = 'https://firebasestorage.googleapis.com/v0/b/pnri-demeter.appspot.com/o/flower.png?alt=media&token=3553abca-251f-42a3-b939-5d8eefc10a9a'
#         passportData = ['Name','Date of Acquisition', 'Accession Origin', 'Project', 'Project Leader', 'Other Detals']
#         morphology = ['Pollinium', 'Retinaculum', 'Caudicle Bulb Diameter', 'Translator']

#         img_url = db.child("Hoya").child(args_str).child("urls").child("img_url").get().val()
#         qr_url= db.child("Hoya").child(args_str).child("urls").child("qr_url").get().val()
#         file_url= db.child("Hoya").child(args_str).child("urls").child("file_url").get().val()

#         screen2 = self.manager.get_screen('singledoc')
#         screen2.ids.datas.clear_widgets()
#         screen2.ids.dataso.clear_widgets()


#         def open(url):
#             if url is None or url == '':
#                 self.show_no_doc_dialog()

#             else:
#                 webbrowser.open(url)

#         screen2.ids.imag.add_widget(
#             MDRoundFlatButton(
#                 text = "View IMG",
#                 on_press = lambda x : open(img_url)
#             )
#         )
#         screen2.ids.file.add_widget(
#             MDRoundFlatButton(
#                 text = "View PDF",
#                 on_press = lambda x : open(file_url)
#             )
#         )
#         screen2.ids.qr.add_widget(
#             MDRaisedButton(
#                 text = "Save QR Code",
#                 on_press = lambda x : open(qr_url)
#             )
#         )

#         if img_url is None or img_url == '':
#             screen2.ids.img_url.source = icon

#         else:
#             screen2.ids.img_url.source = img_url

#         if qr_url is None or qr_url == '':
#             screen2.ids.qr_url.source = icon

#         else:
#             screen2.ids.qr_url.source = qr_url


#         screen2.ids['species'].title = args_str

#         morphology = db.child("Hoya").child(args_str).child("Morphology").get()
#         for datas in morphology.each():
#             screen2.ids.dataso.add_widget(
#                 OneLine(
#                     text=f"{datas.key()} : {datas.val()}"
#                     # halign="center"
#                 )
#             )
        
#         passport_data = db.child("Hoya").child(args_str).child("Passport Data").get()
#         for datas in passport_data.each():
#             screen2.ids.datas.add_widget(
#                 OneLine(
#                     text=f"{datas.key()} : {datas.val()}"
#                     # halign="center"
#                 )
#             )
#         self.manager.current = 'singledoc'    
#         self.manager.transition.direction = 'left'   
#         self.dialog6.dismiss(force=True)