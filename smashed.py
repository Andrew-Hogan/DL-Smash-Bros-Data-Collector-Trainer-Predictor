import threading
import time
import numpy as np
import cv2
import os
from keras.models import load_model
import h5py
from keras import __version__ as keras_version
import argparse
from PIL import ImageGrab
import ctypes


model = None

##set up run
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AI for Smash')
    parser.add_argument(
        'model',
        type=str,
        help='Path to model h5 file. Model should be on the same path.'
    )
    args = parser.parse_args()

    f = h5py.File(args.model, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    model = load_model(args.model)

control_letters = 'wasdxertfgvbijkluonm12345678' 
control_array = list(control_letters)

##Track letter location and which is pressed
values = dict()
for index, letter in enumerate(control_array):
   values[letter] = index

pressed_array = dict()
for index, letter in enumerate(control_array):
    pressed_array[letter] = 0

##set up controls
prediction = np.zeros((0,len(values)))
#prev_move = 0

HEX_CODE = {'a':0x1E,
           'b':0x30,
           'c':0x2E,
           'd':0x20,
           'e':0x12,
           'f':0x21,
           'g':0x22,
           'h':0x23,
           'i':0x17,
           'j':0x24,
           'k':0x25,
           'l':0x26,
           'm':0x32,
           'n':0x31,
           'o':0x18,
           'p':0x19,
           'q':0x10,
           'r':0x13,
           's':0x1F,
           't':0x14,
           'u':0x16,
           'v':0x2F,
           'w':0x11,
           'x':0x2D,
           'y':0x15,
           'z':0x2C
           }
number_moves = '12345678'
special_moves = list(number_moves)

SendInput = ctypes.windll.user32.SendInput
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

# directx scan codes http://www.gamespp.com/directx/directInputKeyboardScanCodes.html
def press(*arg):
    '''
    one press, one release.
    accepts as many arguments as you want. e.g. press('left_arrow', 'a','b').
    '''
    for i in args:
        PressKey(HEX_CODE[i])
        time.sleep(.001)
        ReleaseKey(HEX_CODE[i])
        
def predicts():
    threading.Timer(.2, predicts).start()
    global model
    global values
    global pressed_array
    global prediction
    global control_array
    #global prev_move
    global special_moves

    img = np.array(ImageGrab.grab(bbox=(630,250,1270,700))) ##640x450
    image = cv2.resize(img,(128, 128), interpolation=cv2.INTER_NEAREST)

    prediction = model.predict(image[None, :, :, :], batch_size=1)
    move = np.argmax(prediction)
    print (control_array[move])
    #if move != prev_move:
    #keyboard.release(control_array[prev_move])
    #keyboard.press(control_array[move])
    #keyboard.release(control_array[move])
    if control_array[move] in special_moves:
        print ("Trying to do a direction attack but drew is lazy!")
        key = HEX_CODE['s']
        PressKey(key)
        time.sleep(.1)
        ReleaseKey(key)
    else:
        key = HEX_CODE[control_array[move]]
        PressKey(key)
        time.sleep(.1)
        ReleaseKey(key)

    ##for future
    #prev_move = move

predicts()
    
