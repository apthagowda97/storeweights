import os
import sys

def gdrive():
    try:
        from google.colab import drive

        drive.mount(os.path.join(os.getcwd(), "gdrive"))
    except:
        sys.exit('Google Colab is not detected')
