# For example, here's several helpful packages to load in 
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra                
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
				    
# Input data files are available in the "../input/" directory.        
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
							      
from subprocess import check_output                                                                             
#print(check_output(["ls", "inputs"]).decode("utf8"))
						    
# Any results you write to the current directory are saved as output.
				
np.random.seed(2017)                         
				  
import os                                                            
import glob                         
import cv2                                           
import datetime                                
import time                                  
import warnings                                      
warnings.filterwarnings("ignore")                        
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
				    
from keras.models import Sequential, Model 
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, ELU, SReLU, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta
from keras.regularizers import l2, activity_l2, l1, l1l2, activity_l1 
from keras.utils import np_utils

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
    path = os.path.join('inputs', 'test_stg2', '*.jpg')
    files = sorted(glob.glob(path))

    X_test = []                                      
    X_test_id = []                                            
    for fl in files:                                                                                            
	flbase = os.path.basename(fl)
	img = get_im_cv2(fl)                       
	X_test.append(img)                      
	#X_test_id.append(flbase)
	X_test_id.append('test_stg2/' + flbase)

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
train_data, train_target, train_id = shuffle(train_data, train_target, train_id)
test_data, test_id = shuffle(test_data, test_id)

### image dimensions
imageSize = (64, 64)
img_width, img_height = imageSize[0], imageSize[1]
num_channels = 3

early_stopping = EarlyStopping(monitor='val_loss', patience=10)
batch_size = 8
nb_epoch = 25
    
"""
THE MODEL
"""
def build_model():
    wr1 = 1e-09
    wr2 = 1e-08
    wr3 = 1e-07
    wr4 = 1e-06
    wr5 = 1e-05

    dp = 0.000
    activation = 'relu'
    optimizer = 'sgd'
    
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 64, 64), dim_ordering='th'))
    model.add(Convolution2D(2, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr1), activity_regularizer=activity_l1(wr1), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(2, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr1), activity_regularizer=activity_l1(wr1), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(2, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr2), activity_regularizer=activity_l1(wr2), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(2, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr2), activity_regularizer=activity_l1(wr2), init='lecun_uniform'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr3), activity_regularizer=activity_l1(wr3), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(0.85))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr4), activity_regularizer=activity_l1(wr4), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(0.95))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(16, 3, 3, activation=activation, dim_ordering='th', W_regularizer=l1(wr5), activity_regularizer=activity_l1(wr5), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), dim_ordering='th'))
    #model.add(Dropout(0.95))

    model.add(Flatten())

    """
    model.add(Dense(64, activation=activation, W_regularizer=l1(wr3), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(Dense(32, activation=activation, W_regularizer=l1(wr3), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(Dense(16, activation=activation, W_regularizer=l1(wr3), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(Dense(8, activation=activation, W_regularizer=l1(wr3), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    """

    model.add(Dense(8, activation='softmax', W_regularizer=l1(wr1), init='lecun_uniform'))

    sgd = SGD(lr=1e-1, decay=1e-6, momentum=0.9)
    #adam = Adam(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    #nadam = Nadam(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    adagrad = Adagrad(lr=1.0, decay=1e-2)
    #adamax = Adamax(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True)
    #adadelta = Adadelta(lr=10)

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)

    return model

def run_train(n_folds=5):
    num_fold = 5
    sum_score = 0
    models = []   
    callbacks = [
        early_stopping
    ]
    
    ### if we just want to train a single model without cross-validation, set n_folds to 0 or None
    if not n_folds:
        model = build_model()
        
        #X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=val_split)
        print('Training...')
        #print('Size of train split: ', len(X_train), len(Y_train))
        #print('Size of validation split: ', len(X_valid), len(Y_valid))
              
        h = model.fit(train_data, train_target, 
	    batch_size=batch_size,
	    nb_epoch=nb_epoch,
	    verbose=1, 
	    validation_split=0.1, 
	    shuffle=True,
	    callbacks=[early_stopping])


        predictions_valid = model.predict(test_data, batch_size=batch_size, verbose=1)
        score = h.history['loss'][-1]
        print('Loss: ', score)
        sum_score += score
        models.append(model)
                     
    else:
        kf = KFold(len(train_id), n_folds=n_folds, shuffle=True, random_state=7)

        for train_index, test_index in kf:
            model = build_model()
            X_train = train_data[train_index]
            Y_train = train_target[train_index]
            X_valid = train_data[test_index]
            Y_valid = train_target[test_index]

            num_fold += 1
            print('Training on fold {} of {}...'.format(num_fold, n_folds))
            print('Size of train split: ', len(X_train), len(Y_train))
            print('Size of validation split: ', len(X_valid), len(Y_valid))

            h = model.fit(train_data,
                      train_target,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      shuffle=True,
                      verbose=1,
                      validation_split=0.1,
                      callbacks=[early_stopping])
                      #class_weight=class_weight)

            predictions_valid = model.predict(test_data, batch_size=batch_size, verbose=1)
            score = h.history['loss'][-1]
            print('Loss for fold {0}: '.format(num_fold),score)
            sum_score += score*len(test_index)
            models.append(model)
        score = sum_score/len(train_data)
        
    print("Average loss across folds: ", score)
    
    info_string = "loss-{0:.2f}_{1}fold_{2}x{3}_{4}epoch_patience".format(score, n_folds, img_width, img_height, nb_epoch)
    return info_string, models

print(model)

# RMSE
np.sqrt(0.000156) * 48 

""""""
SAVING AND LOADING WEIGHTS
""""""

# loading weights
###weights = open('net-specialists2.pickle', 'rb')

### saving the model's weights
# serialize model to JSON
model_json = model.to_json()
with open("model_relu_50epochs50patience(newdata).json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model_relu_50epochs50patience(newdata).h5")
print("Saved model to disk")

# load the previous weights' networks
f = h5py.File('model_relu_500epochs.h5')
for k in range(f.attrs['nb_layers']):
    if k >= len(model.layers):
	# we don't look at the last (fully-connected) layers in the savefile
	break
    g = f['layer_{}'.format(k)]
    weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
    model.layers[k].set_weights(weights)
f.close()

# load json and create model
json_file = open('model10.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model_relu_220epochs100patients.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
#score = loaded_model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

preds = loaded_model.predict(test_data, batch_size=8, verbose=1)
    
# compile the model with a SGD/momentum optimizer
#model.compile(loss='categorical_crossentropy',
#	      optimizer=optimizers.SGD(lr=1e-4, momentum=0.9))


""""""
VALIDATION & SUBMISSION STAGE
""""""
def create_submission(predictions, test_id, info):
    #result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result2 = pd.DataFrame(predictions)
    #result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    result2.loc[:, 'image'] = pd.Series(test_id, index=result2.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result2.to_csv(sub_file, index=False)

def merge_several_folds_mean(data, nfolds):
    a = np.array(data[0])
    for i in range(1, nfolds):
        a += np.array(data[i])
    a /= nfolds
    return a.tolist()

def ensemble_predict(info_string, models):
    num_fold = 0
    yfull_test = []
    test_id = []
    n_folds = len(models)

    for i in range(n_folds):
        model = models[i]
        num_fold += 1
        print('Predicting on fold {} of {}'.format(num_fold, n_folds))
        test_data, test_id = read_and_normalize_test_data()
        test_prediction = model.predict(test_data, batch_size=batch_size, verbose=2)
        yfull_test.append(test_prediction)

    preds = merge_several_folds_mean(yfull_test, n_folds)
    create_submission(preds, test_id, info_string)
    
if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5
    info_string, models = run_train()
    #info_string, models = run_cross_validation_create_models(num_folds)
    ensemble_predict(info_string, models)
    