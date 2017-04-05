import numpy as np
from pynput import keyboard
from pynput.keyboard import Controller
import string
from PIL import ImageGrab, Image
import random

control_letters = 'wasdxertfgvbijkluonm12345678' ##12, 13, 14, 15 = i, j, k, l
control_array = list(control_letters)

##Track letter location for Button
global values
values = dict()
for index, letter in enumerate(control_array):
   values[letter] = index

##Track currently pressed keys
global pressed_array
pressed_array = dict()
for index, letter in enumerate(control_array):
    pressed_array[letter] = 0

##Array to be saved
global Button
Button = np.zeros((0,len(values)))

##updates array which is saved at the end
def update_array(key):
    global pressed_array
    global values
    global Button

    Valid_Prediction = np.zeros((1,len(values)))
    if values[key] == 2:##COMBO BUTTONS
       if pressed_array['i'] == 1:
          Valid_Prediction[0, values['1']] = 1
       elif pressed_array['k'] == 1:
          Valid_Prediction[0, values['2']] = 1
       elif pressed_array['j'] == 1:
          Valid_Prediction[0, values['3']] = 1
       elif pressed_array['l'] == 1:
          Valid_Prediction[0, values['4']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    elif values[key] == 8:
       if pressed_array['i'] == 1:
          Valid_Prediction[0, values['5']] = 1
       elif pressed_array['k'] == 1:
          Valid_Prediction[0, values['6']] = 1
       elif pressed_array['j'] == 1:
          Valid_Prediction[0, values['7']] = 1
       elif pressed_array['l'] == 1:
          Valid_Prediction[0, values['8']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    elif values[key] == 15:
       if pressed_array['i'] == 1:
          Valid_Prediction[0, values['o']] = 1
       elif pressed_array['k'] == 1:
          Valid_Prediction[0, values['m']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    elif values[key] == 13:
       if pressed_array['i'] == 1:
          Valid_Prediction[0, values['u']] = 1
       elif pressed_array['n'] == 1:
          Valid_Prediction[0, values['m']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    elif values[key] == 12:
       if pressed_array['j'] == 1:
          Valid_Prediction[0, values['u']] = 1
       elif pressed_array['l'] == 1:
          Valid_Prediction[0, values['o']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    elif values[key] == 14:
       if pressed_array['j'] == 1:
          Valid_Prediction[0, values['n']] = 1
       elif pressed_array['l'] == 1:
          Valid_Prediction[0, values['m']] = 1
       else:
          Valid_Prediction[0, values[key]] = 1
    else:
       Valid_Prediction[0,values[key]] = 1
    
    Button = np.append(Valid_Prediction, Button, axis=0)
    return

numsamples = 0
def capscreen():
   global numsamples
   img = ImageGrab.grab(bbox=(630,250,1270,700)) ##640x450 at the default location (on my computer)
   img.save('./training_data/training_' + str(int(numsamples)) + '.jpg')
   numsamples += 1
   return
    
def on_press(key):
    keyboard = Controller()
    global pressed_array
    try:
        print('alphanumeric key {0} pressed'.format(
            key.char))
        if key.char in pressed_array:
           if pressed_array[key.char] == 0:
              pressed_array[key.char] = 1
              update_array(key.char)
              capscreen()
           elif pressed_array[key.char] == 1:
              if random.randint(1, 10) > 9:
                 update_array(key.char)
                 capscreen()
    except AttributeError:
        print('special key {0} pressed'.format(
            key))
        return

def on_release(key):
    global pressed_array
    global Button
    if key == keyboard.Key.esc:
        # Stop listener
        np.savetxt('training_controls.txt', Button, newline='\r\n', delimiter=',', fmt='%1.1f')
        print (Button.shape)
        return False
    try:
        print('alphanumeric key {0} released'.format(
            key.char))
        if key.char in pressed_array:
           if pressed_array[key.char] == 1:
              pressed_array[key.char] = 0
    except AttributeError:
        print('special key {0} released'.format(
            key))
    return

# Collect events until escape key is pressed
with keyboard.Listener(
        on_press=on_press,
        on_release=on_release) as listener:
    listener.join()
