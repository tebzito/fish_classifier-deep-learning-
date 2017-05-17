# For example, here's several helpful packages to load in 
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra                
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
				    
# Input data files are available in the "../input/" directory.        
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
							      
from subprocess import check_output                                                                             
print(check_output(["ls", "inputs"]).decode("utf8"))
						    
# Any results you write to the current directory are saved as output.
				
np.random.seed(2017)                         
				  
import os                                                            
import glob                         
import cv2                                           
import datetime                                
import time                                  
import warnings                                      
warnings.filterwarnings("ignore")                        
				    
from keras.models import Sequential, Model 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, ELU, SReLU, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta
from keras.regularizers import l2, activity_l2, l1
from keras.utils import np_utils
#import keras.utils.np_utils as utils
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras import __version__ as keras_version             

"""
Reading and loading data
"""

def get_im_cv2(path):                      
    img = cv2.imread(path)                     
    resized = cv2.resize(img, (64, 64), interpolation = cv2.INTER_LINEAR)                        
    return resized           

def load_train():
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
    for fld in folders:
	index = folders.index(fld)
	print('Load folder {} (Index: {})'.format(fld, index))
	path = os.path.join('inputs', 'train', fld, '*.jpg')
	files = glob.glob(path)
	for fl in files:
	    flbase = os.path.basename(fl)
	    img = get_im_cv2(fl)
	    X_train.append(img)
	    X_train_id.append(flbase)
	    y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id

def load_test():                                   
    path = os.path.join('inputs', 'test_stg1', '*.jpg')
    files = sorted(glob.glob(path))
								      
    X_test = []                                      
    X_test_id = []                                            
    for fl in files:                                                                                            
	flbase = os.path.basename(fl)
	img = get_im_cv2(fl)                       
	X_test.append(img)                          
	X_test_id.append(flbase)
					      
    return X_test, X_test_id     


def read_and_normalize_train_data():               
    train_data, train_target, train_id = load_train()

    print('Convert to numpy...')                                      
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)     
														
    print('Reshape...')
    train_data = train_data.transpose((0, 3, 1, 2))
						    
    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, 8)          

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id
				  
def read_and_normalize_test_data():
    start_time = time.time()       
    test_data, test_id = load_test()

    test_data = np.array(test_data, dtype=np.uint8)  
    test_data = test_data.transpose((0, 3, 1, 2))                                
				
    test_data = test_data.astype('float32')
    test_data = test_data / 255          
					      
    print('Test shape:', test_data.shape)
    print(test_data.shape[0], 'test samples')
    print('Read and process test data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return test_data, test_id
  
# Load data
train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

"""
THE MODEL
"""

nb_classes=8
#imgsize = 96
# frame size
nrows = 64
ncols = 64
batch_size = 64
wr = 0.0000001
dp = 0.000
activation = 'relu'
optimizer = 'adagrad'

# video frame in grayscale
frame_in = Input(shape=(3,nrows,ncols), name='img_input')

# Convolutional for each input...but I want weights to be the same
seq = Sequential()
conv1 = Convolution2D(64,3,3, border_mode='same', activation=activation, W_regularizer=l1(wr), init='lecun_uniform')
conv_l1 = conv1(seq, frame_in)
Econv_l1 = LeakyReLU()(conv_l1)
#pool_l1 = MaxPooling2D(pool_size=(2,2))(Econv_l1)
drop_l1 = Dropout(0.75)(Econv_l1)

conv2 = Convolution2D(32,3,3,border_mode='same', activation='relu', W_regularizer=l1(wr), init='lecun_uniform')
conv_l2 = conv2(drop_l1)
Econv_l2 = LeakyReLU()(conv_l2)
#pool_l2 = MaxPooling2D(pool_size=(2,2))(Econv_l2)
drop_l2 = Dropout(0.75)(Econv_l2)


#if params['choice']['layers'] == 'three':
conv3 = Convolution2D(128,2,2,border_mode='same', activation='relu', W_regularizer=l1(wr), init='lecun_uniform')
conv_l3 = conv3(drop_l2)
Econv_l3 = LeakyReLU()(conv_l3)
#pool_l3 = MaxPooling2D(pool_size=(2,2))(Econv_l3)
drop_l3 = Dropout(0.75)(Econv_l3)

'''
conv4 = Convolution2D(64,2,2,border_mode='same', activation=activation, W_regularizer=l1(wr), init='lecun_uniform')
conv_l4 = conv4(drop_l3)
Econv_l4 = ELU()(conv_l4)
#pool_l4 = MaxPooling2D(pool_size=(2,2))(Econv_l4)
drop_l4 = Dropout(0.25)(Econv_l4)
'''

flat = Flatten()(drop_l3)

D1 = Dense(64)(flat)
ED1 = LeakyReLU()(D1)
drop_l5 = Dropout(0.5)(ED1)
D2 = Dense(300)(drop_l5)    
ED2 = ELU()(D2)
drop_l6 = Dropout(0.5)(ED2)
#D3 = Dense(500)(drop_l6)
#ED3 = ELU()(D3)
#drop_l7 = Dropout(0.85)(ED3)

# Adding output layer
imgs = Dense(nb_classes, activation=activation)(drop_l6)

model = Model(input=[frame_in], output=[imgs])

#SGD = SGD(lr=0.0001)
#adam = Adam(lr=0.0001)
#nadam = Nadam(lr=0.0001)
#adagrad = Adagrad(lr=0.001)
#adamax = Adamax(lr=0.01)
adadelta = Adadelta(lr=10)

print("Compiling model..")
model.compile(loss='binary_crossentropy',
		optimizer=adadelta,
		metrics=['accuracy'])

model.summary()

#X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.15, random_state=42)
#Y_train = np_utils.to_categorical(y_train, nb_classes)
#Y_test = np_utils.to_categorical(y_test,nb_classes)

early_stopping = EarlyStopping(monitor='val_loss', patience=10)

#weights = open('net-specialists2.pickle', 'rb')
h = model.fit(train_data, train_target, 
		batch_size=batch_size,
		nb_epoch=10,
		verbose=1, 
		validation_split=0.1, 
		shuffle=True,
		callbacks=[early_stopping])

score,acc = model.evaluate(X_test, Y_test, verbose=0)
print('Test accuracy:', acc)
print('Test loss:', score)
#print("Predicting model..")
preds = model.predict(test_data, batch_size=32)
#acc = mean_squared_error(y_test, preds)
#print('MSE:', acc)
#sys.stdout.flush()
#print(score, acc)
print('Returning loss..')
print(score, type(score))
print(acc, type(acc))

# RMSE
np.sqrt(0.0036) * 48    

"""
Submission process
"""
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

