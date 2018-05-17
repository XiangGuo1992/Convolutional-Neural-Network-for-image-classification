# -*- coding: utf-8 -*-
"""
Created on Tue May 15 23:45:35 2018

@author: Xiang Guo
"""
# from https://www.kaggle.com/sentdex/full-classification-example-with-convnet/code

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion


os.chdir('C:\\github\\action_detection_test\\frames\\test')
directory = 'C:\\github\\action_detection_test\\frames\\test'
subdirectory = [x[0] for x in os.walk(directory)]
del subdirectory[0]

TRAIN_DIR = subdirectory
TEST_DIR = 'C:\\github\\action_detection_test\\frames\\validation\\'
IMG_SIZE = 150
LR = 1e-3

MODEL_NAME = 'action-{}-{}.model'.format(LR, '6conv-basic') # just so we remember which saved model is which, sizes must match


def label_img(sub_label):
    word_label = sub_label.split('\\')[-1]


    if word_label == 'None': return [1,0,0,0]

    elif word_label == 'tie_a_knot': return [0,1,0,0]
    
    elif word_label == 'Save': return [0,0,1,0]
    
    elif word_label == 'Cut': return [0,0,0,1]
    
    
def create_train_data():
    training_data = []
    for sub_label in tqdm(TRAIN_DIR):
        os.chdir(sub_label)
        for img in os.listdir():
            label = label_img(sub_label)

            image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
            training_data.append([np.array(image),np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    return training_data


def process_test_data():
    testing_data = []
    os.chdir(TEST_DIR)
    for sub_label in os.listdir():
        path = os.path.join(TEST_DIR,sub_label)
        for img in os.listdir(path):
            
            img_num = sub_label
            image = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
            testing_data.append([np.array(image), img_num])

        
    shuffle(testing_data)
    np.save('test_data.npy', testing_data)
    return testing_data




train_data = create_train_data()


os.chdir('C:\\github\\action_detection_test\\frames\\method_2')

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf
tf.reset_default_graph()

convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)

convnet = fully_connected(convnet, 4, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')



if os.path.exists('{}.meta'.format(MODEL_NAME)):
    model.load(MODEL_NAME)
    print('model loaded!')
    
train = train_data[:-50]
test = train_data[-50:]


X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
Y = [i[1] for i in train]

test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
test_y = [i[1] for i in test]



model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), 
    snapshot_step=50000, show_metric=True, run_id=MODEL_NAME)



model.save(MODEL_NAME)





#Test
import matplotlib.pyplot as plt

# if you need to create the data:
test_data = process_test_data()
# if you already have some saved:
#test_data = np.load('test_data.npy')

fig=plt.figure()

for num,data in enumerate(test_data[:12]):

    
    img_num = data[1]
    img_data = data[0]
    
    y = fig.add_subplot(3,4,num+1)
    orig = img_data
    data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
    #model_out = model.predict([data])[0]
    model_out = model.predict([data])[0]
    
    if np.argmax(model_out) == 0: str_label='None'
    elif np.argmax(model_out) == 1: str_label='tie_a_knot'
    elif np.argmax(model_out) == 2: str_label='Save'
    elif np.argmax(model_out) == 3: str_label='Cut'
        
    y.imshow(orig,cmap='gray')
    plt.title(str_label)
    y.axes.get_xaxis().set_visible(False)
    y.axes.get_yaxis().set_visible(False)
plt.show()



with open('result.csv','w') as f:
    f.write('id,label\n')
            
with open('result.csv','a') as f:
    for data in tqdm(test_data):
        img_num = data[1]
        img_data = data[0]
        orig = img_data
        data = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
        model_out = model.predict([data])[0]
        f.write('{},{}\n'.format(img_num,model_out[1]))
