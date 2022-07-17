from kivymd.uix.dialog import MDDialog
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton
from kivy.properties import ObjectProperty
from kivymd.uix.list import TwoLineListItem
from kivy.metrics import dp
from kivymd.uix.datatables import MDDataTable
from yutils.imports import *

class TwoLine(TwoLineListItem):
    pass

class Morph(BoxLayout):
    confirm_check_in_list=ObjectProperty()
    def check_conflicts(self,morph, args_str):
        self.ids.confirm_check_in_list.clear_widgets()

        self.ids.confirm_check_in_list.add_widget(
            TwoLine(text=args_str,
                    secondary_text='Sample Name',
            )
        )        
        for key,value in morph['Morphology'][args_str].items():
            self.ids.confirm_check_in_list.add_widget(
                TwoLine(text=value,
                        secondary_text=key,
                )
            )

class Ttest(BoxLayout):
    confirm_check_in_list=ObjectProperty()
    def check_conflicts(self,uk,passed_species,d,args_str):
        morph = {'Morphology': {'sample 1': {'Caudicle Bulb': '0.08', 'Extension': '0.13', 'Hips': '0.34', 'Pollinium Length': '0.73', 'Pollinium Widest': '0.38', 'Retinaculum Length': '0.20', 'Shoulder': '0.08', 'Translator Arm Length': '0.16', 'Translator Depth': '0.06', 'Waist': '0.13'}, 'sample 2': {'Caudicle Bulb': '0.06', 'Extension': '0.11', 'Hips': '0.06', 'Pollinium Length': '0.81', 'Pollinium Widest': '0.19', 'Retinaculum Length': '0.61', 'Shoulder': '0.17', 'Translator Arm Length': '0.23', 'Translator Depth': '0.05', 'Waist': '0.12'}, 'sample 3': {'Caudicle Bulb': '0.12', 'Extension': '0.17', 'Hips': '0.02', 'Pollinium Length': '0.62', 'Pollinium Widest': '0.24', 'Retinaculum Length': '0.35', 'Shoulder': '0.26', 'Translator Arm Length': '0.03', 'Translator Depth': '0.02', 'Waist': '0.08'}}}

        self.ids.confirm_check_in_list.clear_widgets()
        # data_tables = MDDataTable(
        #     size_hint=(1, 1),
        #     use_pagination=True,
        #     check=True,
        #     # name column, width column, sorting function column(optional)
        #     column_data=[
        #         ("No.", dp(30)),
        #         ("Status", dp(30)),
        #         ("Signal Name", dp(60)),
        #         ("Severity", dp(30)),
        #         ("Stage", dp(30)),
        #         ("Schedule", dp(30), lambda *args: print("Sorted using Schedule")),
        #         ("Team Lead", dp(30)),
        #     ],
        # )
        # for passer in passed_species:


        t_test_val = []
        array_t_test = []
        landmarks = ['Caudicle Bulb', 
                    "Extension",
                    "Hips",
                    'Pollinium Length',
                    "Pollinium Widest",
                    "Retinaculum Length",
                    "Shoulder",
                    "Translator Arm Length",
                    "Translator Depth",
                    "Waist",]
        
        for i in range(len(uk.T)):
            val = ttest_ind(d[args_str][i], uk[i])
            array_t_test.append(round(val[1],2))
            # print(val[1])
        t_test_val.append(array_t_test)
        # t_test_val

        print(t_test_val)

        t_test_df = pd.DataFrame(t_test_val).T
        t_test_df.insert(0,"landmarks",landmarks) 
        
        conditions = [
            (t_test_df[0] <= 0.05),
            (t_test_df[0] > 0.05)
            ]
        values = ['significant difference', 'no significant difference']

        t_test_df['interpretaion'] = np.select(conditions, values)


        t_test_row = list(t_test_df.itertuples(index=False, name=None))
        print(t_test_row)
        # layout = AnchorLayout()
        data_tables = MDDataTable(
            size_hint=(1, 1),
            use_pagination=True,
            rows_num=10,
            column_data=[
                (args_str, dp(35)),
                ("pvalue", dp(15)),
                ("interpretation", dp(50)),
            ],
            row_data= t_test_row
        )
        # self.help.get_screen('result').ids.ttest.add_widget(data_tables) 

        self.ids.confirm_check_in_list.add_widget(data_tables)
class Sample(BoxLayout):
    pass

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

def show_morph(self, morph, *args):
    args_str = ','.join(map(str,args))
    if not self.dialog7:
        self.dialog7 = MDDialog(
            title="Sample",
            content_cls=Morph(),
            type="custom",

        )
    self.dialog7.content_cls.check_conflicts(morph,args_str)
    self.dialog7.open()

    
def show_dialog(self):


    if not self.dialog8:
        self.dialog8 = MDDialog(
            title="Add Sample",
            type="custom",
            content_cls=Sample(),
            buttons=[

                MDFlatButton(
                    text="CANCEL",
                    theme_text_color="Custom",
                    on_press= lambda x: self.dialog8.dismiss(force=True)
                ),
                MDFlatButton(
                    text="OK",
                    theme_text_color="Custom",
                    on_press= lambda x: self.puppy(),
                    on_release= lambda y: self.dialog8.dismiss(force=True)                      
                ),
            ],
        )
    self.dialog8.open()

def ttest_dialog(self,uk,passed_species,d, *args):
    args_str = ','.join(map(str,args))
    if not self.dialog9:
        self.dialog9 = MDDialog(
            # title=args_str,
            content_cls=Ttest(),
            type="custom",

        )
    self.dialog9.content_cls.check_conflicts(uk,passed_species,d,args_str)
    self.dialog9.open()