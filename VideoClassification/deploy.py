import os
import cv2
import time
import numpy as np
import numpy as np
import os
import cv2
import keras
import sklearn
import pandas
from time import time
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.utils import to_categorical
from keras.models import load_model
from keras.layers import *
from keras import layers
from keras import Model
from keras.callbacks import TensorBoard
from keras import optimizers
import matplotlib.pyplot as plt
from keras.applications import *
from sklearn.metrics import classification_report
import time
from keras.models import load_model
from sklearn.metrics import confusion_matrix
from keras.applications.resnet50 import preprocess_input
from keras.applications import *
from keras import regularizers
import cv2
import math
import os
import time

import os
import glob
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 




video_path = r"F:\violent_video_tagging\violence-video-classification\data\test_video\\"
frames_output_path = r"F:\violent_video_tagging\violence-video-classification\data\frames_output\\"

print("+++ Removing all the frames from the sourece folder... +++")
files = glob.glob(r"F:\violent_video_tagging\violence-video-classification\data\frames_output\\*")
for f in files:
    os.remove(f)



def video_to_frames(input_loc, output_loc,video_name):

    count = 0
    cap = cv2.VideoCapture(input_loc)   # capturing the video from the given path
    frameRate = cap.get(5) #frame rate
    print (frameRate)

    while(cap.isOpened()):
        frameId = cap.get(10) #current frame number
        # print(frameId)
        ret, frame = cap.read()
        
        if (ret != True):
            break
        if (frameId % math.floor(frameRate) == 0):
            frame = cv2.resize(frame, (224,224))
            filename = video_name + "frame%d.jpg" % count;count+=1
            cv2.imwrite(output_loc + filename, frame)
            
    cap.release()
    print ("Done!")


def create_seq_single_video(frames):
    
    print('+++ Creating Sequence... +++')
    vid = []
#     frames = np.array(frames)
    i = 0
    while i < len(frames):
        vid.append(frames[i:i+30])
        i = i+30
    vid = np.asarray(vid)
    
    return vid
    

def preprocess_lstm(features, frames):
    frames = np.asarray(frames)
    v_features = features[0:30*math.floor(frames.shape[0]/30)]
    
    print("Video Features Shape: ", v_features.shape)
    
    vid = create_seq_single_video(v_features)
    
    test_x = vid
    print("Total data: ", test_x.shape)
    test_x = np.reshape(test_x, (test_x.shape[0],test_x.shape[1],np.prod(test_x.shape[2:])))
    print("(LSTM) After Rehshape: ", test_x.shape)
    
    return test_x


def feature_extract(frames, model):
    
    print("+++ Extracting feature... +++")
    
    frames = np.asarray(frames)
    
    print("Before Feaure extraction: ")
    print(frames.shape)
    
    all_data = frames
    print("Adding all data: ", all_data.shape)
    
    #creates feature descriptors
    all_data = all_data.astype('float64')
    desc = preprocess_input(all_data)
    if(model == 'resnet50'):
        loaded_model = resnet50.ResNet50(input_shape=(224,224,3), include_top=False)
    elif(model == 'vgg19'):
        loaded_model = VGG19(input_shape=(224,224,3), include_top=False)
    elif(model == 'vgg16'):
        loaded_model = VGG16(input_shape=(224,224,3), include_top=False)
    else:
        print("Please give model name - 'resnet50', 'vgg19', 'vgg16'")
        
    loaded_model = Model(loaded_model.input,loaded_model.output)
    features = loaded_model.predict(desc,batch_size=10,verbose=1)

    print ("After Feature extraction: ", features.shape)
    
    return features


videos = [vfile for vfile in os.listdir(video_path)]

for video in videos:
    video_path=os.path.join(video_path,video)
    print("Converting Current Videofile Name to Frames: ",video)
    video_to_frames(video_path,frames_output_path,video)


v_path = r"F:\violent_video_tagging\violence-video-classification\data\frames_output\\"



out_frames = []

for frame in os.listdir(v_path):
    frame = cv2.imread(os.path.join(v_path,frame))
    out_frames.append(frame)


print("++++++++++ Resnet50 Features Extraction ++++++++")
features = feature_extract(out_frames, 'resnet50')

print("++++++++++ Preprocesing LSTM +++++++++++++++++++")
test_x = preprocess_lstm(features, out_frames)

model = Sequential()
model.add(LSTM(50, input_shape=(test_x.shape[1],test_x.shape[2]), return_sequences=False, kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1,activation='sigmoid'))
model.load_weights('resnet_LSTM_v1.h5')
model.summary()
optimizer = optimizers.Adam(lr=0.001,decay=0.004)
model.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=["accuracy"])

print("+++ Generating result... +++")
    
pred = model.predict(test_x)

final_out = [1 if x > 0.5 else 0 for x in pred ]

import pandas as pd
print("+++ Final Output... +++")
print(pd.DataFrame(final_out)[0].value_counts())


df = pd.DataFrame(final_out)
try:
	print("Percentage of non-violent sequence of frames : {0}".format(list(df.value_counts(normalize=True)[1].values*100)[0]))
	print("Percentage of violent sequence of frames : {0}".format(list(df.value_counts(normalize=True)[0].values*100)[0]))
except:
	pass

try:
	if list(df.value_counts(normalize=True)[1].values*100)[0] > 80:
		print("+++++++++++++++++++++ VIDEO IS NON-VIOLENT ++++++++++++++++++++++++++")
	else:
		print("+++++++++++++++++++++ VIDEO IS VIOLENT ++++++++++++++++++++++++++++++")
except:
	print("+++++++++++++++++++++ VIDEO IS VIOLENT ++++++++++++++++++++++++++++++")