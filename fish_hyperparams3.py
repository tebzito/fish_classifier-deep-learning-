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
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta
from keras.callbacks import EarlyStopping    
from keras.regularizers import l2, activity_l2, l1, l1l2
from keras.utils import np_utils           
from keras import optimizers
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
    path = os.path.join('inputs', 'test_stg2', '*.jpg')
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

# hyperparameter tuning
space = {'choice':


hp.choice('num_layers',
    [
                    {'layers':'two',
                     
                                                    
                    },
		     
		    {'layers':'three',
                     
                                                    
                    }, 
		     
		    {'layers':'four',
                     
                                                    
                    },
		     
		    {'layers':'five',
                     
                                                    
                    },
		    
		    {'layers':'six',
                     
                                                    
                    },
		     
		    {'layers':'seven',
                     
                                                    
                    },
		    
		    {'layers':'eight',
                     
                                                    
                    },
		     
		    {'layers':'nine',
                     
                                                    
                    },
		    
		    {'layers':'ten',
                     
                                                    
                    },
		     
		    {'layers':'eleven',
                     
                                                    
                    },
		    
		    {'layers':'twelve',
                     
                    },
		    
		    {
		      
                      'conv14': hp.choice('conv14', [32, 64])
                      #'drop_l4': hp.choice('drop_l4', [0.25,0.5])
                                
                    }
		     
        
    
    ]),
    
    'conv1': hp.choice('conv1', [4, 8]),
    'conv2': hp.choice('conv2', [4, 8]),
    'conv3': hp.choice('conv3', [4, 8]),
    'conv4': hp.choice('conv4', [4, 8]),
    'conv5': hp.choice('conv5', [8, 16]),
    'conv6': hp.choice('conv6', [8, 16]),
    'conv7': hp.choice('conv7', [8, 16]),
    'conv8': hp.choice('conv8', [8, 16]),
    'conv9': hp.choice('conv9', [16, 32]),
    'conv10': hp.choice('conv10', [16, 32]),
    'conv11': hp.choice('conv11', [16, 32]),
    'conv12': hp.choice('conv12', [16, 32]),
    'conv13': hp.choice('conv13', [16, 32]),
    
    #'drop_l1': hp.choice('drop_l1', [0.25,0.5]),
    #'drop_l2': hp.choice('drop_l2', [0.25,0.5]),
    #'drop_l3': hp.choice('drop_l3', [0.25,0.5]),
    #'drop_l4': hp.choice('drop_l4', [0.25,0.5]),
    #'drop_l5': hp.choice('drop_l5', [0.25,0.5]),
    #'drop_l6': hp.choice('drop_l6', [0.25,0.5]),
    
    'batch_size' : hp.choice('batch_size', [8,16]),
    #'nb_epochs' : 1,
    'nb_epochs' :  hp.choice('nb_epochs', [50,100,200]),
    #'optimizer': 'adam',
    #'activations': 'relu'
    'loss': hp.choice('loss', ['categorical_crossentropy']),
    'optimizers': hp.choice('optimizers', ['adam','sgd','adagrad']),
    'activations': hp.choice('activations', ['relu','tanh']),
    'lambda': hp.choice('lambda', [1e-05, 1e-06, 1e-07, 1e-08, 1e-09])
	  
    }


def model(params):
    #wr = 0.0001
    print('Params testing: ', params)
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), dim_ordering='th'))
    print("Adding first layer..")
    model.add(Convolution2D(params['conv1'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding second layer..")
    model.add(Convolution2D(params['conv2'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l1']))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding third layer..")
    model.add(Convolution2D(params['conv3'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding fourth layer..")
    model.add(Convolution2D(params['conv4'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l2']))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding firth layer..")
    model.add(Convolution2D(params['conv5'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding sixth layer..")
    model.add(Convolution2D(params['conv6'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l3']))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding seventh layer..")
    model.add(Convolution2D(params['conv7'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding eighth layer..")
    model.add(Convolution2D(params['conv8'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l3']))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding nineth layer..")
    model.add(Convolution2D(params['conv9'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding tenth layer..")
    model.add(Convolution2D(params['conv10'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l3']))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding eleventh layer..")
    model.add(Convolution2D(params['conv11'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding twelvth layer..")
    model.add(Convolution2D(params['conv12'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l3']))
    
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    print("Adding thirteenth layer..")
    model.add(Convolution2D(params['conv13'], 3, 3, activation='relu', dim_ordering='th', W_regularizer=l1l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(params['drop_l3']))
    
    model.add(Flatten())
    
    """
    print("Adding hidden layer..")
    model.add(Dense(params['hidden1'], activation='relu', W_regularizer=l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(Dropout(params['drop_l4']))
    model.add(Dense(params['hidden2'], activation='relu', W_regularizer=l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    model.add(Dropout(params['drop_l5']))
    #model.add(Dense(params['hidden3'], activation='relu', W_regularizer=l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))
    #model.add(Dropout(params['drop_l6']))
    """
    
    print("Adding output layer..")
    model.add(Dense(8, activation='softmax', W_regularizer=l2(params['lambda']), activity_regularizer=activity_l2(params['lambda']), init='lecun_uniform'))

    #sgd = SGD(lr=1.0, decay=1e-2, momentum=0.9, nesterov=True)
    #adam = Adam(lr=1.0, decay=1e-2)
    #nadam = Nadam(lr=1.0, schedule_decay=1e-2)
    #nadam = Nadam(lr=0.01, schedule_decay=1e-6)
    #adagrad = Adagrad(lr=1.0, decay=1e-2)
    #adamax = Adamax(lr=1e-2, decay=1e-6, momentum=0.9)
    #adadelta = Adadelta(lr=10)

    model.compile(optimizer=params['optimizers'], loss=params['loss'], metrics=['accuracy'])

    model.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    #weights = open('net-specialists2.pickle', 'rb')
    h = model.fit(train_data, train_target, 
		    batch_size=params['batch_size'],
		    nb_epoch=params['nb_epochs'],
		    verbose=1, 
		    validation_split=0.1, 
		    shuffle=True,
		    callbacks=[early_stopping])
    #score, acc = model.evaluate(X_test, Y_test, verbose=0)
    #print('Test accuracy:', acc)
    #print('Test loss:', score)
    print("Predicting model..")
    preds = model.predict(test_data, batch_size=params['batch_size'])
    #acc = mean_squared_error(y_test, preds)
    #print('MSE:', acc)
    #sys.stdout.flush()
    #print(score, acc)
    print('Returning loss..')
    loss = h.history['val_loss'][-1]
    loss = loss.astype('float32')
    print(loss, type(loss))
    #print(model)
    #for a in acc.itervalues():
    return {'loss': loss, 'status': STATUS_OK, 'model': model}

#data = train_data, train_target,test_data

if __name__ == '__main__':
    best_run, best_model = optim.fmin(model,
			      #data=data,
			      space,
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
    
"""
SUBMISSION STAGE
"""

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

