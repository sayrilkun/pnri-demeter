import pyrebase
import os
from yutils.configg import config

firebase = pyrebase.initialize_app(config)
storage= firebase.storage()
db= firebase.database()