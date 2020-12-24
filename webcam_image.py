import cv2

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    cv2.imshow('Input', frame)

    c = cv2.waitKey(1)
    if c == 27:
        cv2.imwrite('webcam_input.jpeg',frame)
        break

cap.release()
cv2.destroyAllWindows()


#!pip install opencv-python-headless==3.4.9.31 (Uncomment in colab)
#!pip install keras-tqdm (Uncomment in colab)
#!pip install pyspellchecker (Uncomment in colab)
#!pip install gTTS (Uncomment in colab)
import cv2
import numpy as np
#import os
import pandas as pd
from matplotlib import pyplot as plt
from gtts import gTTS #Import Google Text to Speech
from IPython.display import Audio #Import Audio method from IPython's Display Class
#from google.colab.patches import cv2_imshow (Uncomment in colab)


######change 1#######
#image = cv2.imread('C:/Users/VISHVA/Desktop/Handwriting/test.jpeg')


#print('Original Image')
#cv2_imshow(image)
#plt.imshow(image)

from keras.preprocessing.sequence import pad_sequences

from keras.layers import Dense, LSTM, Reshape, BatchNormalization, Input, Conv2D, MaxPool2D, Lambda, Bidirectional
from keras.models import Model
from keras.activations import relu, sigmoid, softmax
import keras.backend as K
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
#from keras-tqdm import TQDMNotebookCallback

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf

#ignore warnings in the output
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from tensorflow.python.client import device_lib

# Check all available devices if GPU is available
#print(device_lib.list_local_devices())
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

max_label_len = 0

char_list = "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz" 

# string.ascii_letters + string.digits (Chars & Digits)
# or 
# "!\"#&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

#print(char_list, len(char_list))

def encode_to_labels(txt):
    # encoding each output word into digits
    dig_lst = []
    for index, chara in enumerate(txt):
        dig_lst.append(char_list.index(chara))
        
    return dig_lst


#MODEL

# input with shape of height=32 and width=128 
inputs = Input(shape=(32,128,1))
 
# convolution layer with kernel size (3,3)
conv_1 = Conv2D(64, (3,3), activation = 'relu', padding='same')(inputs)
# poolig layer with kernel size (2,2)
pool_1 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_1)
 
conv_2 = Conv2D(128, (3,3), activation = 'relu', padding='same')(pool_1)
pool_2 = MaxPool2D(pool_size=(2, 2), strides=2)(conv_2)
 
conv_3 = Conv2D(256, (3,3), activation = 'relu', padding='same')(pool_2)
 
conv_4 = Conv2D(256, (3,3), activation = 'relu', padding='same')(conv_3)
# poolig layer with kernel size (2,1)
pool_4 = MaxPool2D(pool_size=(2, 1))(conv_4)
 
conv_5 = Conv2D(512, (3,3), activation = 'relu', padding='same')(pool_4)
# Batch normalization layer
batch_norm_5 = BatchNormalization()(conv_5)
 
conv_6 = Conv2D(512, (3,3), activation = 'relu', padding='same')(batch_norm_5)
batch_norm_6 = BatchNormalization()(conv_6)
pool_6 = MaxPool2D(pool_size=(2, 1))(batch_norm_6)
 
conv_7 = Conv2D(512, (2,2), activation = 'relu')(pool_6)
 
squeezed = Lambda(lambda x: K.squeeze(x, 1))(conv_7)
 
# bidirectional LSTM layers with units=128
blstm_1 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(squeezed)
blstm_2 = Bidirectional(LSTM(256, return_sequences=True, dropout = 0.2))(blstm_1)
 
outputs = Dense(len(char_list)+1, activation = 'softmax')(blstm_2)

# model to be used at test time
act_model = Model(inputs, outputs)

#LOAD WEIGHTS
#LOAD WEIGHTS

######change 2#3###
act_model.load_weights('C:/Users/VISHVA/Desktop/Handwriting/adamo-100000r-30e-77006t-8557v.hdf5')



def process_image(img):
    """
    Converts image to shape (32, 128, 1) & normalize
    """
    w, h = img.shape
    
#     _, img = cv2.threshold(img, 
#                            128, 
#                            255, 
#                            cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Aspect Ratio Calculation
    new_w = 32
    new_h = int(h * (new_w / w))
    img = cv2.resize(img, (new_h, new_w))
    w, h = img.shape
    
    img = img.astype('float32')
    
    # Converts each to (32, 128, 1)
    if w < 32:
        add_zeros = np.full((32-w, h), 255)
        img = np.concatenate((img, add_zeros))
        w, h = img.shape
    
    if h < 128:
        add_zeros = np.full((w, 128-h), 255)
        img = np.concatenate((img, add_zeros), axis=1)
        w, h = img.shape
        
    if h > 128 or w > 32:
        dim = (128,32)
        img = cv2.resize(img, dim)
    
   # img = cv2.subtract(255, img)
    
    img = np.expand_dims(img, axis=2)
    
    # Normalize 
    img = img / 255
    
    return img


def pred(img):
  pxmin = np.min(img)
  pxmax = np.max(img)
  imgContrast = (img - pxmin) / (pxmax - pxmin) * 255
  wi=''
  worda=[]
# increase line width
  kernel = np.ones((3, 3), np.uint8)
  imgMorph = cv2.erode(imgContrast, kernel, iterations = 1)
  p=process_image(imgMorph)
  p=p.reshape((1,32,128,1))
  prediction = act_model.predict(p)
  out = K.get_value(K.ctc_decode(prediction, input_length=np.ones(prediction.shape[0])*prediction.shape[1],
                         greedy=True)[0][0])
  for x in out:
   # print("original_text =  ", valid_original_text[i])
    #print("predicted text = ", end = '')
    for p in x:  
        if int(p) != -1:
           worda.append(char_list[int(p)])   
    wi=wi.join(worda)   
  return wi




#import sys (COLAB)
#sys.path.append('/content/drive/MyDrive/Colab Notebooks/common_modules') COLAB
a=[]
import page
import words
from PIL import Image
import cv2

# User input page image

#####change 3######
image = cv2.cvtColor( cv2.imread('C:/Users/VISHVA/Desktop/Handwriting/webcam_input.jpeg'), cv2.COLOR_BGR2RGB)
#cv2_imshow(image)
speech=[]

# Crop image and get bounding boxes
#crop = page.detection(image)
io=image.copy()
boxes = words.detection(io)
lines = words.sort_words(boxes)

# Saving the bounded words from the page image in sorted way
i = 0
for line in lines:
    text = io.copy()
    for (x1, y1, x2, y2) in line:
        roi = text[y1:y2, x1:x2]
        # save = Image.fromarray(text[y1:y2, x1:x2])
        # print(i)
        a.append(roi)
        #.save("segmented/segment" + str(i) + ".png")
        i += 1

for i in range(len(a)):
  img=a[i]
  #print('Actual word')

  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  # print(gray.shape)
  t=pred(gray)
  #print("Predicted Text: ",t)
  #cv2_imshow(gray)
  speech.append(t)
from spellchecker import SpellChecker

spell = SpellChecker()

# find those words that may be misspelled
misspelled = spell.unknown(speech)
for i in range(len(speech)):
  if speech[i].lower() in misspelled:
    speech[i]=spell.correction(speech[i])

 
  


separator = ' '
stri=separator.join(speech)
tts = gTTS(stri) #Provide the string to convert to speech
tts.save('1.wav') #save the string converted to speech as a .wav file
sound_file = '1.wav'
Audio(sound_file, autoplay=True)
print('The String is: ' , stri)
print('DONE!')

