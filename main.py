from utils.imports import *

import os
arr = os.listdir("kv/")
for i in arr:
    Builder.load_file(f'kv/{i}')

Window.size=(400,700)
global config
config = {
    "apiKey": "AIzaSyBH3WOpmUdPj0vGIpneswkW2CS8fFidlXw",
    "authDomain": "pnri-demeter.firebaseapp.com",
    "databaseURL": "https://pnri-demeter-default-rtdb.firebaseio.com",
    "projectId": "pnri-demeter",
    "storageBucket": "pnri-demeter.appspot.com",
    "messagingSenderId": "456214792415",
    "appId": "1:456214792415:web:773d7ea18f8ba214df816a",
    "measurementId": "G-00QH790MRG",
}


firebase = pyrebase.initialize_app(config)
storage= firebase.storage()
db= firebase.database()

# global morph
# morph  = {
#         "Morphology":{
#         }
#     }

class KivyCamera(Image):
    pass

class ImageButton(ButtonBehavior):
    pass

class IconLeftSampleWidget(IRightBodyTouch, MDIconButton):
    pass

class OneLineIcon(OneLineAvatarIconListItem):
    pass

class ThreeLineIcon(ThreeLineAvatarIconListItem):
    pass

class OneLine(OneLineListItem):
    divider = None    

class Tab(MDFloatLayout, MDTabsBase):
    pass


class DemoApp(MDApp):



###################################################################
# COLOR SCHEMES
###################################################################
    purple = 56/255,40/255,81/255,1
    dark_green = 43/255, 172/255, 127/255, 1
    light_green =  197/255, 230/255, 127/255, 1
    light_green2 = 27/255, 229/255, 127/255, 1
    light_green3 = 135/255, 230/255, 127/255, 1
    light_green4 = 193/255, 225/255, 193/255, 1
    dark_green2 = 37/255, 160/255, 127/255, 1
    dark1 = 143/255, 188/255, 143/255,1
    light1 = 60/255, 179/255, 113/255, 1
    dark2 = 46/255, 139/255, 87/255, 1

    arr = [] 
    paginated = ()
    page_length = 0
    page_number = 0
    
    morph  = {
            "Morphology":{
            }
        }

###################################################################
# DIALOGS
###################################################################

    dialog = None
    def form_dialog(self):
        dialog.form_dialog(self)

    dialog2= None
    def show_no_doc_dialog(self):
        dialog.show_no_doc_dialog(self)

    dialog3= None
    def show_simple_dialog(self):
        dialog.show_simple_dialog(self)

    dialog4= None
    def delete_dialog(self):
        dialog.delete_dialog(self)

    dialog6= None    
    @mainthread
    def spin_dialog(self):
        dialog.spin_dialog(self)

    dialog7= None
    def show_morph(self,morph, *args):
        dialog.show_morph(self, morph, *args)

    dialog8= None
    def show_dialog(self):
        dialog.show_dialog(self)

###################################################################
# SWITCH SCREEN
###################################################################
    def swtchScrn(self,*args):
        self.page_number = 0
        self.refresh_callback()
        self.help.current = 'collections'
        self.help.transition.direction = 'right'


    def swtchScreen(self,screen,*args):
        # self.refresh_callback()
        self.clear_morph()
        self.help.current = screen
        self.help.transition.direction = 'right'

###################################################################
# DELETE DOCUMENT
###################################################################
    def delete_doc(self):
        doc= self.help.get_screen('singledoc').ids.species.title
        db.child("Hoya").child(doc).remove()
        self.swtchScrn()

###################################################################
# GET DOCUMENT LISTS FROM DATABASE
###################################################################


    def next_button(self):
        self.page_number+=1
        print(self.page_number)
        if self.page_number < self.page_length:
            self.list_callback()
            self.help.get_screen('collections').ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
            self.help.get_screen('collections').ids.page_numberad.text = f"{(self.page_number+1)*10} out of {len(self.arr)}"

        else:
            self.page_number-=1

    def prev_button(self):
        self.page_number-=1
        print(self.page_number)
        if self.page_number >= 0:
            self.list_callback()
            self.help.get_screen('collections').ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
            self.help.get_screen('collections').ids.page_numberad.text = f"{(self.page_number+1)*10} of {len(self.arr)}"

        else:
            self.page_number+=1   

    def pop_array(self):
        all_docs = db.child("Hoya").get()
        for i in all_docs.each():
            self.arr.append(i.key())
        def chunk(it, size):
            it = iter(it)
            return iter(lambda: tuple(islice(it, size)), ())
        self.paginated = list(chunk(self.arr,10))
        self.page_length = len(self.paginated)
        self.help.get_screen('collections').ids.page_number.text = f"Page {self.page_number+1} of {self.page_length}"
        self.help.get_screen('collections').ids.page_numberad.text = f"{(self.page_number+1)*10} of {len(self.arr)}"
        
    def list_array(self):
        async def search_list():
            search=self.help.get_screen('collections').ids.search.text
            for i in self.paginated[self.page_number]:
                if search in i:
                    await asynckivy.sleep(0)
                    # sample_num = db.child("Hoya").child(i).child("sample_num").get()
                    # status = db.child("Hoya").child(i).child("status").get()
                    self.help.get_screen('collections').ids.box.add_widget(
                        OneLineIcon(text= f'{i}',
                            # secondary_text = f'Status: {status.val()}',
                            # tertiary_text = f'{sample_num.val()} samples',
                        # on_touch_down = lambda x: print("adfads")
                        on_release = lambda y: self.spin_dialog(),
                        on_press= lambda x, value_for_pass=i: self.passValue_thread(value_for_pass),

                        ))  
        asynckivy.start(search_list())  

    def search_array(self):
        async def search_list():
            search=self.help.get_screen('collections').ids.search.text
            if search == '':
                self.list_callback()
            else:    
                for i in self.arr:
                    if search in i:
                        await asynckivy.sleep(0)
                        # sample_num = db.child("Hoya").child(i).child("sample_num").get()
                        # status = db.child("Hoya").child(i).child("status").get()
                        self.help.get_screen('collections').ids.box.add_widget(
                            OneLineIcon(text= f'{i}',
                            # secondary_text = f'Status: {status.val()}',
                            # tertiary_text = f'{sample_num.val()} samples',
                            # on_touch_down = lambda x: print("adfads")
                            on_release = lambda y: self.spin_dialog(),
                            on_press= lambda x, value_for_pass=i: self.passValue_thread(value_for_pass),

                            ))
                        
        asynckivy.start(search_list())    

    def list_callback(self, *args):
        '''A method that updates the state of your application
        while the spinner remains on the screen.'''

        def refresh_callback(interval):
            self.help.get_screen('collections').ids.box.clear_widgets()
            self.list_array()
            self.help.get_screen('collections').ids.refresh_layout.refresh_done()
            self.tick = 0

        Clock.schedule_once(refresh_callback, 1)

    def search_callback(self, *args):
        '''A method that updates the state of your application
        while the spinner remains on the screen.'''

        def refresh_callback(interval):
            self.help.get_screen('collections').ids.box.clear_widgets()
            self.search_array()
            self.help.get_screen('collections').ids.refresh_layout.refresh_done()
            self.tick = 0

        Clock.schedule_once(refresh_callback, 1)

    def refresh_callback(self, *args):
        '''A method that updates the state of your application
        while the spinner remains on the screen.'''

        def refresh_callback(interval):
            self.help.get_screen('collections').ids.box.clear_widgets()
            self.arr = []
            self.pop_array()
            self.list_array()
            self.help.get_screen('collections').ids.refresh_layout.refresh_done()
            self.tick = 0

        Clock.schedule_once(refresh_callback, 1)
###################################################################
# GET DATA FOR EACH DOCUMENT
###################################################################

    def passValue_thread(self,*args):
        threading.Thread(target=self.passValue, args = args).start()

    def passValue(self, *args):


        args_str = ','.join(map(str,args))

        icon = 'https://firebasestorage.googleapis.com/v0/b/pnri-demeter.appspot.com/o/flower.png?alt=media&token=3553abca-251f-42a3-b939-5d8eefc10a9a'
        passportData = ['Name','Date of Acquisition', 'Accession Origin', 'Project', 'Project Leader', 'Other Detals']
        morphology = ['Pollinium', 'Retinaculum', 'Caudicle Bulb Diameter', 'Translator']

        img_url = db.child("Hoya").child(args_str).child("urls").child("img_url").get().val()
        qr_url= db.child("Hoya").child(args_str).child("urls").child("qr_url").get().val()
        file_url= db.child("Hoya").child(args_str).child("urls").child("file_url").get().val()

        screen2 = self.help.get_screen('singledoc')
        screen2.ids.datas.clear_widgets()
        screen2.ids.dataso.clear_widgets()


        def open(url):
            if url is None or url == '':
                self.show_no_doc_dialog()

            else:
                webbrowser.open(url)

        screen2.ids.imag.add_widget(
            MDRoundFlatButton(
                text = "View IMG",
                on_press = lambda x : open(img_url)
            )
        )
        screen2.ids.file.add_widget(
            MDRoundFlatButton(
                text = "View PDF",
                on_press = lambda x : open(file_url)
            )
        )
        screen2.ids.qr.add_widget(
            MDRaisedButton(
                text = "Save QR Code",
                on_press = lambda x : open(qr_url)
            )
        )

        if img_url is None or img_url == '':
            screen2.ids.img_url.source = icon

        else:
            screen2.ids.img_url.source = img_url

        if qr_url is None or qr_url == '':
            screen2.ids.qr_url.source = icon

        else:
            screen2.ids.qr_url.source = qr_url


        screen2.ids['species'].title = args_str
        # screen2.ids['datos'].text = datos
        # screen2.ids['header'].text = f"Morphometric Analysis of {args_str}"



        # for complete documents / to separeate passport data and morphology
        morphology = db.child("Hoya").child(args_str).child("Morphology").get()
        morphi  = {
                "Morphology":{
                }
            }
        morphi['Morphology'].update(morphology.val())
        for datas in morphology.each():
            screen2.ids.dataso.add_widget(
                OneLine(
                    text=f"{datas.key()}",
                    on_press= lambda x, value_for_pass= datas.key(): self.show_morph(morphi,value_for_pass)
                    # halign="center"
                )
            )
        
        passport_data = db.child("Hoya").child(args_str).child("Passport Data").get()
        for datas in passport_data.each():
            screen2.ids.datas.add_widget(
                OneLine(
                    text=f"{datas.key()} : {datas.val()}"
                    # halign="center"
                )
            )
        self.help.current = 'singledoc'    
        self.help.transition.direction = 'left'   
        self.dialog6.dismiss(force=True)

###################################################################
# UPLOAD/EDIT STATE
###################################################################
    def change_state(self, num):
        global states
        states = num
        # return state
    
    def check_state(self):

        if states == 0:
            self.switchScreen('uploaddoc')
        elif states == 1:
            # self.switchScreen('edit')
            pass
        else:
            toast('Invalid')

    def upload_state(self):
        self.change_state(0)
        self.check_state()

    def edit_state(self):
        self.change_state(1)
        self.check_state()
###################################################################
# EDIT DOCUMENT
###################################################################
        
    def edit_doc(self):
        passportData = ['Name','Date of Acquisition', 'Accession Origin', 'Project', 'Project Leader', 'Other Detals']
        # morphology = ['Pollinium', 'Retinaculum', 'Caudicle Bulb Diameter', 'Translator']

        self.swtchScreen('edit')
        doc= self.help.get_screen('singledoc').ids.species.title

        passport_data = db.child("Hoya").child(doc).child("Passport Data").get()
        for datas in passport_data.each():
            self.help.get_screen('edit').ids.passpo.add_widget(
                MDTextField(
                    # id= "Adf",
                    hint_text= datas.key(),
                    text = datas.val(),
                    size_hint = (.75,0.08),
                    mode = "rectangle",
                    color_mode = 'accent',
                    pos_hint = {"center_x": 0.5}
                )
            )
            

        morphology = db.child("Hoya").child(doc).child("Morphology").get()
        for datas in morphology.each():
            self.help.get_screen('edit').ids.morph.add_widget(
                MDTextField(
                    # id = datas.key(),
                    hint_text= datas.key(),
                    text = datas.val(),
                    size_hint = (.75,0.08),
                    mode = "rectangle",
                    color_mode = 'accent',
                    pos_hint = {"center_x": 0.5}
                )
            )

    def update_doc(self):
        pass
        # img_url = db.child("Hoya").child(args_str).child("urls").child("img_url").get().val()
        # qr_url= db.child("Hoya").child(args_str).child("urls").child("qr_url").get().val()
        # file_url= db.child("Hoya").child(args_str).child("urls").child("file_url").get().val()
        # self.swtchScreen('edit')

        # name = self.help.get_screen('uploaddoc').ids.input_1.text
        # dateAcq = self.help.get_screen('uploaddoc').ids.input_2.text
        # accOrg = self.help.get_screen('uploaddoc').ids.input_3.text
        # project = self.help.get_screen('uploaddoc').ids.input_4.text
        # prjLdr = self.help.get_screen('uploaddoc').ids.input_5.text
        # otherDtls = self.help.get_screen('uploaddoc').ids.input_6.text
        # pollinium = self.help.get_screen('uploaddoc').ids.input_7.text
        # retinaculum = self.help.get_screen('uploaddoc').ids.input_8.text
        # translator = self.help.get_screen('uploaddoc').ids.input_9.text
        # caudicle = self.help.get_screen('uploaddoc').ids.input_10.text
        # image = self.help.get_screen('uploaddoc').ids.input_11.text
        # file = self.help.get_screen('uploaddoc').ids.input_12.text

    dict = {}
    def add_textfields(self):
        self.dialog.dismiss(force=True)
        # self.dialog.content_cls.ids.title.text, self.dialog.content_cls.ids.desc.text = "",""

        title = self.dialog.content_cls.ids.title.text
        desc = self.dialog.content_cls.ids.desc.text
        self.dict[f'{title}'] = f'{desc}'
        print(self.dict)
        self.help.get_screen('edit').ids.morph.add_widget(
        MDTextField(hint_text= title,
                    text = desc,
                    size_hint = (.75,0.08),
                    mode = "rectangle",
                    color_mode = 'accent',
                    pos_hint = {"center_x": 0.5}
        ))

###################################################################
# UPLOAD DOCUMENT
###################################################################
    def puppy(self):


        sn = self.dialog8.content_cls.ids.input_19.text
        cb = self.dialog8.content_cls.ids.input_7.text
        el = self.dialog8.content_cls.ids.input_8.text
        hl = self.dialog8.content_cls.ids.input_9.text
        pl = self.dialog8.content_cls.ids.input_10.text
        pw = self.dialog8.content_cls.ids.input_13.text
        rl = self.dialog8.content_cls.ids.input_14.text
        shl = self.dialog8.content_cls.ids.input_15.text
        tal = self.dialog8.content_cls.ids.input_16.text
        td = self.dialog8.content_cls.ids.input_17.text
        wl = self.dialog8.content_cls.ids.input_18.text

        sopu = {
            f'{sn}':{
                'Caudicle Bulb': f'{cb}',
                "Extension": f'{el}',
                "Hips": f'{hl}',
                'Pollinium Length': f'{pl}',
                "Pollinium Widest": f'{pw}',
                "Retinaculum Length": f'{rl}',
                "Shoulder": f'{shl}',
                "Translator Arm Length": f'{tal}',
                "Translator Depth": f'{td}',
                "Waist": f'{wl}',
                }
            }
        if sn == '':
            toast("Name Cannot Be Blank!")
        else:
            self.morph['Morphology'].update(sopu)
            print(self.morph)

            self.help.get_screen('uploaddoc').ids.box.add_widget(
                OneLineIcon(text= sn,
                # on_touch_down = lambda x: print("adfads")
                # on_release = lambda y: self.show_morph(),
                on_press= lambda x, value_for_pass= sn: self.show_morph(self.morph,value_for_pass),

                ))

            self.help.get_screen('iden').ids.box.add_widget(
                OneLineIcon(text= sn,
                # on_touch_down = lambda x: print("adfads")
                # on_release = lambda y: self.show_morph(),
                on_press= lambda x, value_for_pass= sn: self.show_morph(self.morph,value_for_pass),

                ))

            for i in range(7,11):
                self.dialog8.content_cls.ids[f'input_{i}'].text = ""
            
            for i in range(13,20):
                self.dialog8.content_cls.ids[f'input_{i}'].text = ""

    def clear_morph(self):
        self.morph =  {
            "Morphology":{
                }
            }
        self.help.get_screen('uploaddoc').ids.box.clear_widgets()
        self.help.get_screen('iden').ids.box.clear_widgets()



    def upload_thread(self):
        self.spin_dialog()
        threading.Thread(target=(self.upload)).start()

    def upload(self):
        input_fields = ['name','dateAcq','accOrg','project','prjLdr','otherDtls',
                        'pollinium','retinaculum','translator', 'caudicle', 'image', 'file']
            
        name = self.help.get_screen('uploaddoc').ids.input_1.text
        dateAcq = self.help.get_screen('uploaddoc').ids.input_2.text
        accOrg = self.help.get_screen('uploaddoc').ids.input_3.text
        project = self.help.get_screen('uploaddoc').ids.input_4.text
        prjLdr = self.help.get_screen('uploaddoc').ids.input_5.text
        otherDtls = self.help.get_screen('uploaddoc').ids.input_6.text
        status = self.help.get_screen('uploaddoc').ids.input_7.text

        sample_num = len(self.morph['Morphology'].keys())

        image = self.help.get_screen('uploaddoc').ids.input_11.text
        file = self.help.get_screen('uploaddoc').ids.input_12.text

        if image == "":
            img_url = ""
        else:
            storage.child(f"{name}/{name}_image").put(image)
            img_url = storage.child(f"{name}/{name}_image").get_url(None)
        
        if file == "":
            file_url = ""
        else:
            storage.child(f"{name}/{name}_file").put(file)
            file_url = storage.child(f"{name}/{name}_file").get_url(None)

        scan_id = self.generate()

        qr_url = self.add_qr(name, scan_id)

        data = { 
            "Passport Data": {
                'Name': f'{name}',
                'Date Acquired':f'{dateAcq}',
                'Accession Origin': f'{accOrg}',
                'Project': f'{project}',
                'Project Leader': f'{prjLdr}',
                'Other Details': f'{otherDtls}',
                'Status': f'{status}',
            },

            "urls":{
                'img_url' : f'{img_url}',
                'file_url' : f'{file_url}',
                'qr_url': f'{qr_url}',
            },

            'scan_id' : f'{scan_id}',
            'status': f'{status}',
            'sample_num':sample_num,
        }
        data.update(self.morph)

        if name == '':
            toast("Name Cannot Be Blank!")
            self.dialog6.dismiss(force=True)
        elif name in self.arr:
            toast("Unable to upload! Name exists in database!")
            self.dialog6.dismiss(force=True)
        else:
            db.child('Hoya').child(f'{name}').set(data)
            self.clear_morph()
            self.dialog6.dismiss(force=True)
            toast("Document Saved Successfully")
            self.swtchScrn()

    def add_img(self):
        root = tk.Tk()
        root.withdraw()
        image = filedialog.askopenfilename()
        print(image)
        # print(file)
        self.help.get_screen('uploaddoc').ids.input_11.text = image

    def add_file(self):
        root = tk.Tk()
        root.withdraw()
        file = filedialog.askopenfilename()
        print(file)
        # print(file)
        self.help.get_screen('uploaddoc').ids.input_12.text = file

###################################################################
# GENERATE ENCRYPTED SCAN ID
###################################################################
    def generate(self):
        S = 8  # number of characters in the string.  
        # call random.choices() string module to find the string in Uppercase + numeric data.  
        ran = 'ac&@%!'+''.join(random.choices(string.ascii_uppercase + string.digits, k = S))    
        x= str(ran)
        return x

###################################################################
# GENERATE QR CODE
###################################################################
    def add_qr(self, name, scan_id):
        input_data = scan_id
        #Creating an instance of qrcode
        qr = qrcode.QRCode(
                version=1,
                box_size=10,
                border=5)
        qr.add_data(input_data)
        qr.make(fit=True)
        img = qr.make_image(fill='black', back_color='white')
        filename = f'assets/qr_codes/{name}_qr.png'
        img.save(filename)
        storage.child(f"{name}/{name}_qr").put(filename)
        qr_url = storage.child(f"{name}/{name}_qr").get_url(None)
        # print(qr_url)
        return qr_url

###################################################################
# QR CODE SCANNER
###################################################################
    def show_camscanner(self):
        cam = self.help.get_screen('scanner').ids.cam
        self.clock_event = Clock.schedule_interval(self.update, 1.0 /30)
        cam.capture = cv2.VideoCapture('http://192.168.1.2:4747/video',cv2.CAP_DSHOW)

    def update(self, dt):
        cam = self.help.get_screen('scanner').ids.cam
        ret, frame = cam.capture.read()
        # duts = []
        if ret:
            # cv2.putText(frame, 'dfgd', (50, 50),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
            
            for barcode in decode(frame):
                
                myData = barcode.data.decode('utf-8')
                sub = 'ac&@%!'
                if sub in myData:
                    hey = db.child("Hoya").order_by_child("scan_id").equal_to(myData).get()
                    for user in hey.each():
                        self.help.get_screen('qr').ids.forem.text = user.key()  
                else:
                    self.help.get_screen('qr').ids.forem.text = "NO DOCUMENT AVAILABLE"
                # qr_ref = db.collection('Hoya')
                # docs = qr_ref.where('scan_id', '==', f'{myData}').get()
                # for doc in docs:
                #     self.help.get_screen('qr').ids.forem.text = doc.id
                self.help.current = 'qr'
                # webbrowser.open(myData)
                pts = np.array([barcode.polygon], np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], True, (255, 0, 255), 5)
                pts2 = barcode.rect
                cv2.putText(frame, myData,(pts2[0],pts2[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
                            
            buf1 = cv2.flip(frame, -1)
            buf = buf1.tobytes()
            image_texture =Texture.create(
                size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            cam.texture = image_texture


###################################################################
# OPEN SCANNED DOCUMENT TO APP
###################################################################
    def oks_qr(self):
        myDate = self.help.get_screen('qr').ids.forem.text
        try:
            self.passValue(myDate)
        except Exception as e:
            toast("No Such Document!")

    def oks_thread(self):
        threading.Thread(target=(self.oks_qr)).start()

###################################################################
# OPEN SCANNED LINK TO WEBSITE
###################################################################
    def my_qr(self):
        myDate = self.ids.forem.text
        self.ids.link.add_widget(
            MDRaisedButton( text = "Open link",
            on_press = lambda x: webbrowser.open(myDate))
        )     

###################################################################
# CREATE TABLES
###################################################################
# def iden_thread(self, *args):
#     self.spin_dialog()
#     threading.Thread(target=(self.identify), args=self.morph).start()

    def createTable(self):
        morph = {'Morphology': {'sample 1': {'Caudicle Bulb': '0.08', 'Extension': '0.13', 'Hips': '0.34', 'Pollinium Length': '0.73', 'Pollinium Widest': '0.38', 'Retinaculum Length': '0.20', 'Shoulder': '0.08', 'Translator Arm Length': '0.16', 'Translator Depth': '0.06', 'Waist': '0.13'}, 'sample 2': {'Caudicle Bulb': '0.06', 'Extension': '0.11', 'Hips': '0.06', 'Pollinium Length': '0.81', 'Pollinium Widest': '0.19', 'Retinaculum Length': '0.61', 'Shoulder': '0.17', 'Translator Arm Length': '0.23', 'Translator Depth': '0.05', 'Waist': '0.12'}, 'sample 3': {'Caudicle Bulb': '0.12', 'Extension': '0.17', 'Hips': '0.02', 'Pollinium Length': '0.62', 'Pollinium Widest': '0.24', 'Retinaculum Length': '0.35', 'Shoulder': '0.26', 'Translator Arm Length': '0.03', 'Translator Depth': '0.02', 'Waist': '0.08'}}}

        a,b,c = iden.identify(morph)
        print(a,b,c)
        
        



###################################################################
# CAMERA
###################################################################
    def capture(self):
        self.help.get_screen('image').ids.cap_img.source = 'assets/captured_img/image.png'
        self.swtchScreen('image')

    def save_img(self):
        self.help.get_screen('uploaddoc').ids.input_11.text = 'assets/captured_img/image.png'
        self.swtchScreen('uploaddoc') 


    def show_camie(self):
        # disabled cam, naglalag kasi
        cam = self.help.get_screen('camera').ids.camie
        self.clock_event = Clock.schedule_interval(self.object_detection, 1.0 /60)
        cam.capture = cv2.VideoCapture(1,cv2.CAP_DSHOW) 
        self.help.current = 'camera'
        self.help.transition.direction = "left"

        
    def object_detection(self, dt):

        cam = self.help.get_screen('camera').ids.camie
        ret, frame = cam.capture.read()  
        if ret:


            buf1 = cv2.flip(frame, 0)
            buf = buf1.tobytes()
            image_texture =Texture.create(
                size = (frame.shape[1], frame.shape[0]), colorfmt='bgr'
            )
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            cam.texture = image_texture
            self.help.get_screen('camera').ids.capture.add_widget(
                MDFloatingActionButton( 
                    text = "Capture", 
                    icon = 'camera-iris', 
                    md_bg_color="#f8d7e3", 
                    text_color = "#211c29",
                    on_press = lambda x: cv2.imwrite(f'assets/captured_img/image.png', frame),
                    on_release = lambda x : self.capture())
            )
    def cam_start(self):
        camruler.start()

    def camruler_thread(self):
        threading.Thread(target=(self.cam_start)).start()
        # self.dialog6.dismiss(force=True)

    def on_start(self):
        self.pop_array()
        self.list_array()
        # GeneratorScreen.panget(self)
        # CollectionsScreen.pop_array(self)
        # CollectionsScreen.list_array(self)


    def build(self):
        self.title='Demeter'
        self.theme_cls.theme_style = "Light"
        self.theme_cls.primary_palette = "LightGreen"   

        self.help = Builder.load_file('main.kv')
        # screen.add_widget(self.help)
        return self.help

DemoApp().run()