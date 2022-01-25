from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen
from kivy.core.window import Window

Window.size = (300,500)
class MenuScreen(Screen):
    pass

class Tree(MDApp):

    def build(self):
        # screen =Screen()
        
        self.title='Sample'
        self.theme_cls.theme_style = "Dark"
        self.theme_cls.primary_palette = "Blue"   

        self.help = Builder.load_file('try.kv')
        # screen.add_widget(self.help)
        return self.help

Tree().run()