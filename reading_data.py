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
				
np.random.seed(2016)                         
				  
import os                                                            
import glob                         
import cv2                                           
import datetime                                
import time                                  
import warnings                                      
warnings.filterwarnings("ignore")                        
				    
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD                                                 
from keras.callbacks import EarlyStopping    
from keras.utils import np_utils           
from keras import __version__ as keras_version             


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
  
"""
THE MODEL
"""
def model(params):
    nb_classes=8
    imgsize = 96
    # frame size
    nrows = 96
    ncols = 96
    batch_size = 128
    wr = 0.00000
    dp = 0.0

    # video frame in grayscale
    frame_in = Input(shape=(1,nrows,ncols), name='img_input')
    print('Params testing: ', params)
      
    # Convolutional for each input...but I want weights to be the same
    print("Adding first layer..")
    conv1 = Convolution2D(params['conv1'],3,3, border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
    conv_l1 = conv1(frame_in)
    Econv_l1 = ELU()(conv_l1)
    #pool_l1 = MaxPooling2D(pool_size=(2,2))(Econv_l1)
    drop_l1 = Dropout(params['drop_l1'])(Econv_l1)
    
    
    print("Adding second layer..")
    conv2 = Convolution2D(params['conv2'],3,3,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
    conv_l2 = conv2(drop_l1)
    Econv_l2 = ELU()(conv_l2)
    #pool_l2 = MaxPooling2D(pool_size=(2,2))(Econv_l2)
    drop_l2 = Dropout(params['drop_l2'])(Econv_l2)
    
    print("Adding three layer..")
    #if params['choice']['layers'] == 'three':
    #print("continuing third layer..")
    conv3 = Convolution2D(params['conv3'],3,3,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
    conv_l3 = conv3(drop_l2)
    Econv_l3 = ELU()(conv_l3)
    #pool_l3 = MaxPooling2D(pool_size=(2,2))(Econv_l3)
    drop_l3 = Dropout(params['drop_l3'])(Econv_l3)
    
    print("Adding fourth layer..")
    conv4 = Convolution2D(params['conv4'],2,2,border_mode='same', W_regularizer=l1(wr), init='lecun_uniform')
    conv_l4 = conv4(drop_l3)
    Econv_l4 = ELU()(conv_l4)
    #pool_l4 = MaxPooling2D(pool_size=(2,2))(Econv_l4)
    drop_l4 = Dropout(params['drop_l3'])(Econv_l4)
    #model = Model(input=[frame_in], output=[drop_l4])
    #model.summary()
    
    '''
    flat = Flatten()(drop_l4)
    #Rs = Reshape((1,9216))(flat)
    #M = merge([flat], mode='concat',concat_axis=1)
    #Rs = Reshape((1,256+2))(M)
    
    
    #A1 = Dense(32, activation='sigmoid')(D3)
    S1 = Dense(32)(drop_l6)
    ES1 = ELU()(S1)
    '''
    print("flatten layers..")
    flat = Flatten()(drop_l4)
    
    print("hidden layers..")
    D1 = Dense(params['conv5'])(flat)
    ED1 = ELU()(D1)
    D2 = Dense(params['conv6'])(ED1)    
    ED2 = ELU()(D2)
    drop_l5 = Dropout(params['drop_l5'])(ED2)
    D3 = Dense(params['conv7'])(drop_l5)
    ED3 = ELU()(D3)
    drop_l6 = Dropout(params['drop_l6'])(ED3)
    
    print("Adding output layer..")
    imgs = Dense(nb_classes, activation=params['activations'])(drop_l6)

    model = Model(input=[frame_in], output=[imgs])

    sgd = SGD(lr=0.01)
    adam = Adam(lr=0.01)
    nadam = Nadam(lr=0.001)

    print("Compiling model..")
    model.compile(loss=['mse'],
		      optimizer=params['optimizers'],
		      metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)

    #X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.15, random_state=42)
    #Y_train = np_utils.to_categorical(y_train, nb_classes)
    #Y_test = np_utils.to_categorical(y_test,nb_classes)
    print("Fitting model..")
    h = model.fit(X_train, Y_train, 
		  batch_size=params['batch_size'],
		  nb_epoch=params['nb_epochs'],
		  verbose=1, 
		  validation_split=0.1, 
		  shuffle=True,
		  callbacks=[early_stopping])
    score, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test accuracy:', acc)
    print('Test loss:', score)
    #print("Predicting model..")
    #preds = model.predict(X_test, batch_size=params['batch_size'])
    #acc = mean_squared_error(y_test, preds)
    #print('MSE:', acc)
    #sys.stdout.flush()
    #print(score, acc)
    print('Returning loss..')
    print(score, type(score))
    print(acc, type(acc))
    #for a in acc.itervalues():
    return {'loss': -acc, 'status': STATUS_OK, 'model': model}

    
train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

