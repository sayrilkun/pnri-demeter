from kivy.uix.screenmanager import Screen
# import utils.pyre as pyre
from utils.pyre import Fireboo

class CollectionsScreen(Screen):
    pass
    # def on_enter(self):
    #     db = Fireboo()
    #     all_docs = db.child("Hoya").get()
    #     for i in all_docs.each():
    #         print(i)

# class ClassA(object):
#     def __init__(self):
#         self.var1 = 1
#         self.var2 = 2

#     def methodA(self):
#         self.var1 = self.var1 + self.var2
#         return self.var1



# class ClassB(ClassA):
#     def __init__(self, class_a):
#         self.var1 = class_a.var1
#         self.var2 = class_a.var2

# object1 = ClassA()
# sum = object1.methodA()
# object2 = ClassB(object1)
# print (sum)
