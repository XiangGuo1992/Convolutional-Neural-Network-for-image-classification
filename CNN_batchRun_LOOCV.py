# -*- coding: utf-8 -*-
"""
Created on Sun May 20 20:15:43 2018

@author: Xiang Guo
"""

# from https://www.kaggle.com/sentdex/full-classification-example-with-convnet/code

import cv2                 # working with, mainly resizing, images
import numpy as np         # dealing with arrays
import os                  # dealing with directories
from random import shuffle # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm      # a nice pretty percentage bar for tasks. Thanks to viewer Daniel Bühler for this suggestion


os.chdir('C:\\github\\action_detection_test\\frames\\Labeled_data')
folders = os.listdir()
IMG_SIZE = 150
LR = 1e-3
result = []
result2 = []
labels = ['None','tie_a_knot','Push_down','Cut','Transition']

'''
directory = 'C:\\github\\action_detection_test\\frames\\test'
subdirectory = [x[0] for x in os.walk(directory)]
del subdirectory[0]
TRAIN_DIR = subdirectory
'''


def label_img(sub_label):
    word_label = sub_label.split('\\')[-1]


    if word_label == 'None': return [1,0,0,0,0]

    elif word_label == 'tie_a_knot': return [0,1,0,0,0]
    
    elif word_label == 'Push_down': return [0,0,1,0,0]
    
    elif word_label == 'Cut': return [0,0,0,1,0]
    
    elif word_label == 'Transition': return [0,0,0,0,1]

#all validation set
def create_train_data():
    training_data = []
    folders2 = os.listdir('C:\\github\\action_detection_test\\frames\\Labeled_data')
    folders2.remove(subject)
    for sub_folder in folders2:
        TRAIN_FOLD = os.path.join(os.getcwd(),sub_folder)
        for sub_label in os.listdir(TRAIN_FOLD):
            os.chdir(os.path.join(TRAIN_FOLD,sub_label))
            imglist = os.listdir()
            for img in imglist:
                label = label_img(sub_label)
                image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
                training_data.append([np.array(image),np.array(label),img])
        os.chdir('C:\\github\\action_detection_test\\frames\\Labeled_data')
    
    shuffle(training_data)
    np.save('C:\\github\\action_detection_test\\frames\\result2\\{}_train_data.npy'.format(subject), training_data)
    return training_data



#LOOCV validation set
def create_train_data2():
    training_data2 = []
    for sub_label in tqdm(os.listdir(TRAIN_DIR)):
        os.chdir(os.path.join(TRAIN_DIR,sub_label))
        imglist = os.listdir()
        for img in imglist:
            label = label_img(sub_label)
            image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
            training_data2.append([np.array(image),np.array(label),img])
    shuffle(training_data2)
    np.save('C:\\github\\action_detection_test\\frames\\result3\\{}_train_data.npy'.format(subject), training_data2)
    return training_data2



#build networks
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

import tensorflow as tf





for subject in tqdm(folders): 
    os.chdir('C:\\github\\action_detection_test\\frames\\Labeled_data')
    TRAIN_DIR = os.path.join(os.getcwd(),subject)


    #TEST_DIR = os.path.join(os.getcwd(),'testing')
    
    
    MODEL_NAME = 'No-{}-{}-{}.model'.format(subject,LR, '6conv-basic') # just so we remember which saved model is which, sizes must match
    
        
    
    
    '''
    def process_test_data():
        testing_data = []
        for sub_label in os.listdir(TEST_DIR):
            os.chdir(os.path.join(TEST_DIR,sub_label))
            for img in os.listdir():
                
                img_num = sub_label
                image = cv2.imread(img,cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (IMG_SIZE,IMG_SIZE))
                testing_data.append([np.array(image), img_num,img])
    
            
        shuffle(testing_data)
        np.save('C:\\github\\action_detection_test\\frames\\method_2_2_lessTraining\\test_data.npy', testing_data)
        return testing_data
    '''
    
    
    
    train_data = create_train_data()
    
    os.chdir('C:\\github\\action_detection_test\\frames\\result2')
    
    #os.chdir('C:\\github\\action_detection_test\\frames\\method_2_2_lessTraining')
    
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
    
    convnet = fully_connected(convnet, 5, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
    
    model = tflearn.DNN(convnet, tensorboard_dir='C:\\github\\action_detection_test\\frames\\result2\\tensorboard\\{}\\log'.format(subject))
    
    
    if os.path.exists('{}.meta'.format(MODEL_NAME)):
        model.load(MODEL_NAME)
        print('model loaded!')
    
    
    train = train_data[:int(0.7*len(train_data))]
    test = train_data[int(0.7*len(train_data)):]
    
    
    X = np.array([i[0] for i in train]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    Y = [i[1] for i in train]
    
    
    test_x = np.array([i[0] for i in test]).reshape(-1,IMG_SIZE,IMG_SIZE,1)
    test_y = [i[1] for i in test]
    
    
    model.fit({'input': X}, {'targets': Y}, n_epoch=50, validation_set=({'input': test_x}, {'targets': test_y}), 
        snapshot_step=50000, show_metric=True, run_id=MODEL_NAME)
    
    
    
    model.save('C:\\github\\action_detection_test\\frames\\result2\\model\\'+ subject + '\\'+MODEL_NAME)
    
    
    '''
    model_out = model.predict(test_x)
    
    
    labels = ['None','tie_a_knot','Push_down','Cut','Transition']
    labels[np.argmax(model_out,axis=1)]
    '''
    
    
    #Test
    
    
    
    
    #import matplotlib.pyplot as plt
    
    # if you need to create the data:
    test_data = train_data
    test_data2 = create_train_data2()
    # if you already have some saved:
    #test_data = np.load('test_data.npy')
    '''
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
        elif np.argmax(model_out) == 2: str_label='Push_down'
        elif np.argmax(model_out) == 3: str_label='Cut'
        elif np.argmax(model_out) == 4: str_label='Transition'
            
        y.imshow(orig,cmap='gray')
        plt.title(str_label)
        y.axes.get_xaxis().set_visible(False)
        y.axes.get_yaxis().set_visible(False)
    plt.show()
    '''
    
    
    os.chdir('C:\\github\\action_detection_test\\frames\\result3')
    #write result of the same data sets
    with open('{}_result.csv'.format(subject),'w') as f:
        f.write('label,prediction,file\n')
                
    with open('{}_result.csv'.format(subject),'a') as f:
        for data in tqdm(test_data):
            img_num = labels[np.argmax(data[1])]
            img_data = data[0]
            filename = data[2]
            orig = img_data
            data2 = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
            model_out = model.predict([data2])
            f.write('{},{},{}\n'.format(img_num,labels[np.argmax(model_out)],filename))
    
    
    #write result of LOOCV data set
    with open('{}_result_LOOCV.csv'.format(subject),'w') as f:
        f.write('label,prediction,file\n')
                
    with open('{}_result_LOOCV.csv'.format(subject),'a') as f:
        for data in tqdm(test_data2):
            img_num = labels[np.argmax(data[1])]
            img_data = data[0]
            filename = data[2]
            orig = img_data
            data2 = img_data.reshape(IMG_SIZE,IMG_SIZE,1)
            model_out = model.predict([data2])
            f.write('{},{},{}\n'.format(img_num,labels[np.argmax(model_out)],filename))    
    

    import pandas as pd
    df = pd.read_csv('{}_result.csv'.format(subject))
    df2 = pd.read_csv('{}_result_LOOCV.csv'.format(subject))
    
    
    
    
    result.append([subject,np.mean(df['label']==df['prediction'])])
    result2.append([subject,np.mean(df2['label']==df2['prediction'])])
    print('Accuracy for {} is {}'.format(subject,np.mean(df['label']==df['prediction'])))
    print('LOOCV Accuracy for {} is {}'.format(subject,np.mean(df2['label']==df2['prediction'])))


os.chdir('C:\\github\\action_detection_test\\frames\\result3')
csv_out = pd.DataFrame(result)
csv_out.columns = ['No-Subject','Accuracy']
csv_out.to_csv('Accuracy_result.csv',index=False)

csv_out2 = pd.DataFrame(result2)
csv_out2.columns = ['No-Subject','Accuracy']
csv_out2.to_csv('Accuracy_result_LOOCV.csv',index=False)

