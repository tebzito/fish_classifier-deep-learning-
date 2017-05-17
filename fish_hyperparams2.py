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
				    
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD                                                 
from keras.callbacks import EarlyStopping    
from keras.utils import np_utils           
from keras import __version__ as keras_version

from hyperopt import Trials, STATUS_OK, STATUS_RUNNING, tpe, hp, fmin, partial, mix, rand, anneal
from hyperas import optim
from hyperas.distributions import choice, uniform

"""
Loading and reading data
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

train_data, train_target, train_id = read_and_normalize_train_data()
test_data, test_id = read_and_normalize_test_data()

"""
THE MODEL
"""
def model(train_data, train_target, test_data):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), dim_ordering='th'))
    model.add(Convolution2D({{choice([32, 64, 128])}}, 3, 3, activation='relu', dim_ordering='th', W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D({{choice([32, 64, 128])}}, 3, 3, activation={{choice(['relu','tanh','sigmoid'])}}, dim_ordering='th', W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D({{choice([32, 64, 128])}}, 3, 3, activation={{choice(['relu','tanh','sigmoid'])}}, dim_ordering='th', W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(64, 3, 3, activation={{choice(['relu','tanh','sigmoid'])}}, dim_ordering='th', W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense({{choice([32, 64, 128])}}, activation={{choice(['relu','tanh','sigmoid'])}}, W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    model.add(Dropout({{uniform(0,1)}}))
    #model.add(Dense({{choice([32, 64, 128])}}, activation={{choice(['relu','tanh','sigmoid'])}}, W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))
    #model.add(Dropout({{uniform(0,1)}}))
    model.add(Dense(8, activation='softmax', W_regularizer=l2({{choice([0.1,0.01,0.001,0.0001])}}), init='lecun_uniform'))

    #sgd = SGD(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    #adam = Adam(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #nadam = Nadam(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #adagrad = Adagrad(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    #adamax = Adamax(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    adadelta = Adadelta(lr=10)

    model.compile(optimizer={{choice(['adadelta', 'sgd', 'adam'])}}, loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)

    #weights = open('net-specialists2.pickle', 'rb')
    h = model.fit(train_data, train_target, 
		    batch_size={{choice([16,32,64,128])}},
		    nb_epoch=10,
		    verbose=1, 
		    validation_split=0.1, 
		    shuffle=True,
		    callbacks=[early_stopping])
    #score, acc = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test accuracy:', acc)
    #print('Test loss:', score)
    print("Predicting model..")
    preds = model.predict(test_data, batch_size={{choice([16,32,64,128])}})
    #acc = mean_squared_error(y_test, preds)
    #print('MSE:', acc)
    #sys.stdout.flush()
    #print(score, acc)
    print('Returning loss..')
    loss = h.history['val_loss'][-1]
    loss = loss.astype('float32')
    print(loss, type(loss))
    #for a in acc.itervalues():
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

data = train_data, train_target,test_data

if __name__ == '__main__':
    best_run, best_model = optim.minimize(model,
			      data=data,
			      #space,
			      algo=partial(mix.suggest,
				p_suggest=[
				(.1, rand.suggest),
				(.2, anneal.suggest),
				(.7, tpe.suggest),]),
			      max_evals=50,
			      trials=Trials())
    print("Evaluation of best performing model:")    
    #print(best_model.evaluate(X_test, Y_test))
    print(best_run)
    print(best_model)
    
def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

