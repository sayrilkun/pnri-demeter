import cv2
import numpy as np
import random
import os
import webbrowser
import tkinter as tk
import pyrebase
import qrcode
import string
import threading

import utils.frame_capture as frame_capture
import utils.frame_draw as frame_draw
import utils.dialog as dialog
# import utils.camruler as camruler

from kivy.clock import mainthread
from kivymd.toast import toast
from datetime import datetime
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.button import MDIconButton, MDFloatingActionButton
from tkinter import filedialog
from kivy.properties import StringProperty

from kivymd.app import MDApp
from kivy.lang.builder import Builder
from kivy.uix.screenmanager import Screen
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.core.window import Window
from pyzbar.pyzbar import decode
from kivymd.uix.list import IRightBodyTouch, OneLineListItem, OneLineIconListItem, OneLineAvatarIconListItem
from kivymd.uix.list import IconLeftWidget
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.tab import MDTabsBase
from kivymd.utils import asynckivy
from kivymd.uix.button import MDFlatButton, MDRaisedButton, MDRoundFlatButton
from kivymd.uix.dialog import MDDialog
from kivy.uix.image import AsyncImage
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.label import MDLabel
from kivy.uix.boxlayout import BoxLayout
from kivymd.uix.textfield import MDTextField

