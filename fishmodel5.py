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
import h5py
import warnings                                      
from sklearn.utils import shuffle
from sklearn.cross_validation import KFold
from sklearn.metrics import log_loss, confusion_matrix
warnings.filterwarnings("ignore")                        

from keras.models import Sequential, Model , model_from_json
from keras.layers.core import Dense, Dropout, Flatten, Reshape
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers import Input, ELU, SReLU, LeakyReLU
from keras.optimizers import SGD, Adam, RMSprop, Nadam, Adagrad, Adamax, TFOptimizer, Adadelta
from keras.regularizers import l2, activity_l2, l1, l1l2, activity_l1 
from keras.utils.np_utils import convert_kernel
from keras.utils import np_utils
from keras import optimizers

from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint 
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
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
train_data, train_target, train_id = shuffle(train_data, train_target, train_id)
test_data, test_id = shuffle(test_data, test_id)

### 
NEW MODEL
###

### path for preloaded vgg16 weights and bottleneck model (once trained)
weights_path = 'weights/vgg16_weights.h5'
bottleneck_model_weights_path = 'weights/model_relu_50epochs50patience(local_optima).h5'  # these need to be in local directory

### settings for keras early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=1, mode='auto')

### other hyperparameters
n_folds = 5
batch_size = 8
nb_epoch = 5
bottleneck_epoch = 3 # used when training bottleneck model
val_split = .15  # if not using kfold cv
classes = ["ALB", "BET", "DOL", "LAG", "NoF", "OTHER", "SHARK", "YFT"]
num_classes = len(classes)
class_weight = None


### image dimensions
imageSize = (64, 64)
img_width, img_height = imageSize[0], imageSize[1]
num_channels = 3

"""
# Flipping tensorflow VGG16/19 to theano dim_ordering 
for layer in model.layers:
   if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
      original_w = K.get_value(layer.W)
      converted_w = convert_kernel(original_w)
      K.set_value(layer.W, converted_w)
"""

def build_model():
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(num_channels, img_width, img_height), dim_ordering='tf'))

    model.add(Convolution2D(4, 3, 3, activation='relu', name='conv1_1', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', name='conv1_2', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(4, 3, 3, activation='relu', name='conv2_1', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(4, 3, 3, activation='relu', name='conv2_2', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', name='conv3_1', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', name='conv3_2', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', name='conv3_3', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', name='conv4_1', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', name='conv4_2', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', name='conv4_3', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv5_1', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv5_2', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', name='conv5_3', dim_ordering='tf', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.summary()

    # load the weights of the VGG16 networks
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        if k.__class__.__name__ in ['Convolution1D', 'Convolution2D']:
	    original_w = K.get_value(k.W)
	    converted_w = convert_kernel(original_w)
	    K.set_value(k.W, converted_w)
	    model.layers[k].set_weights(weights)
    f.close()
    
    # build a classifier model to put on top of the convolutional model
    bottleneck_model = Sequential()
    bottleneck_model.add(Flatten(input_shape=model.output_shape[1:]))
    #bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(4, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Reshape((None, None, None)))
    #bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(8, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(8, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(8, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.5))
    
    bottleneck_model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))

    bottleneck_model.add(Dense(64, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #bottleneck_model.add(Dropout(0.75))
    
    bottleneck_model.add(Dense(8, activation='softmax', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    bottleneck_model.summary()
    
    # load weights from bottleneck model
    #bottleneck_model.load_weights(bottleneck_model_weights_path)
    ops = []
    for layer in bottleneck_model.layers:
	if layer.__class__.__name__ in ['Convolution1D', 'Convolution2D', 'Convolution3D', 'AtrousConvolution2D']:
	    original_w = K.get_value(layer.W)
	    converted_w = convert_kernel(original_w)
	    ops.append(tf.assign(layer.W, converted_w).op)

    # add the model on top of the convolutional base
    model.add(bottleneck_model)

    # set the first 25 layers (up to the last conv block)
    # to non-trainable (weights will not be updated)
    for layer in model.layers[:25]:
        layer.trainable = False
        
    # compile the model with a SGD/momentum optimizer
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-2, momentum=0.9, decay=1e-2), metrics=['accuracy'])
    
    return model

def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1./255)

    # build the VGG16 network
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, img_width, img_height)))

    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name='conv1_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name='conv1_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.5))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name='conv2_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th', name='conv2_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', name='conv3_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', name='conv3_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(16, 3, 3, activation='relu', dim_ordering='th', name='conv3_3', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv4_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv4_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv4_3', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv5_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv5_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 3, 3, activation='relu', dim_ordering='th', name='conv5_3', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))
    
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th', name='conv6_1', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th', name='conv6_2', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))
    model.add(Convolution2D(64, 3, 3, activation='relu', dim_ordering='th', name='conv6_3', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    #model.add(Dropout(0.75))
    
    model.summary()
    
    # load the weights of the VGG16 networks
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Model loaded.')
    
    # create validation split
    train_data, train_target, _ = normalize_train_data()
    X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=val_split)

    # create generator for train data
    generator = datagen.flow(
            X_train,
            Y_train,
            batch_size=batch_size,
            shuffle=False)
    
    # save train features to .npy file
    bottleneck_features_train = model.predict_generator(generator, X_train.shape[0])
    np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

    # create generator for validation data
    generator = datagen.flow(
            X_valid,
            Y_valid,
            batch_size=batch_size,
            shuffle=False)
    
    # save validation features to .npy file
    bottleneck_features_validation = model.predict_generator(generator, X_valid.shape[0])
    np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)
    return Y_train, Y_valid

def train_bottleneck_model():
    train_labels, validation_labels = save_bottleneck_features()

    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(4, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    
    model.add(Dense(8, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.5))
    
    
    model.add(Dense(8, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    
    model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    
    model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.85))
    
    model.add(Dense(16, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    
    model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.85))
    
    model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.75))
    
    model.add(Dense(32, activation='relu', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))
    #model.add(Dropout(0.85))
    
    model.add(Dense(8, activation='softmax', W_regularizer=l1(1e-07), activity_regularizer=activity_l1(1e-07), init='lecun_uniform'))

    model.compile(optimizer=optimizers.Adagrad(lr=0.1, decay=1e-2), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(train_data,
              train_labels,
              nb_epoch=bottleneck_epoch,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels),
              callbacks=[early_stopping],
              class_weight=class_weight,
              verbose=2)
    
    model.save_weights(bottleneck_model_weights_path)
    return model
  
def run_train(n_folds=n_folds):
    num_fold = 0
    sum_score = 0
    models = []   
    callbacks = [
        early_stopping
    ]
    
    ### if we just want to train a single model without cross-validation, set n_folds to 0 or None
    if not n_folds:
        model = build_model()
        
        X_train, X_valid, Y_train, Y_valid = train_test_split(train_data, train_target, test_size=val_split)
        print('Training...')
        print('Size of train split: ', len(X_train), len(Y_train))
        print('Size of validation split: ', len(X_valid), len(Y_valid))
              
        model.fit(X_train,
          Y_train,
          batch_size=batch_size,
          nb_epoch=nb_epoch,
          shuffle=True,
          verbose=1,
          validation_data=(X_valid, Y_valid),
          callbacks=callbacks,
          class_weight=class_weight)

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
        score = log_loss(Y_valid, predictions_valid)
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

            model.fit(X_train,
                      Y_train,
                      batch_size=batch_size,
                      nb_epoch=nb_epoch,
                      shuffle=True,
                      verbose=1,
                      validation_data=(X_valid, Y_valid),
                      callbacks=callbacks,
                      class_weight=class_weight)

            predictions_valid = model.predict(X_valid.astype('float32'), batch_size=batch_size, verbose=2)
            score = log_loss(Y_valid, predictions_valid)
            print('Loss for fold {0}: '.format(num_fold), score)
            sum_score += score*len(test_index)
            models.append(model)
        score = sum_score/len(train_data)
        
    print("Average loss across folds: ", score)
    
    info_string = "loss-{0:.2f}_{1}fold_{2}x{3}_{4}epoch_patience_vgg16".format(score, n_folds, img_width, img_height, nb_epoch)
    return info_string, models
    
"""
VALIDATION & SUBMISSION STAGE
"""

def create_submission(predictions, test_id, info):
    result1 = pd.DataFrame(predictions, columns=['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT'])
    result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
    now = datetime.datetime.now()
    sub_file = 'submission_' + info + '_' + str(now.strftime("%Y-%m-%d-%H-%M")) + '.csv'
    result1.to_csv(sub_file, index=False)

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

"""
def run_cross_validation_create_models(nfolds=10):
    # input image dimensions
    batch_size = 8
    nb_epoch = 15
    random_state = 51
    first_rl = 96

    train_data, train_target, train_id = read_and_normalize_train_data()

    yfull_train = dict()
    kf = KFold(len(train_id), n_folds=nfolds, shuffle=True, random_state=random_state)
    num_fold = 0
    sum_score = 0
    models = []
    for train_index, test_index in kf:
        #model = create_model()
        X_train = train_data[train_index]
        Y_train = train_target[train_index]
        X_valid = train_data[test_index]
        Y_valid = train_target[test_index]

        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, verbose=0),
        ]
        
        model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
             shuffle=True, verbose=2, validation_data=(X_valid, Y_valid),
             callbacks=callbacks)

        # load weights into new model
	#loaded_model.load_weights("model_relu_500epochs.h5")
	#print("Loaded model from disk")

        predictions_valid = model.predict(X_valid.astype('float32'), batch_size=8, verbose=1)
        score = log_loss(Y_valid, predictions_valid)
        print('Score log_loss: ', score)
        sum_score += score*len(test_index)

        # Store valid predictions
        for i in range(len(test_index)):
            yfull_train[test_index[i]] = predictions_valid[i]

        models.append(model)

    score = sum_score/len(train_data)
    print("Log_loss train independent avg: ", score)

    info_string = 'loss_' + str(score) + '_folds_' + str(nfolds) + '_ep_' + str(nb_epoch) + '_fl_' + str(first_rl)
    return info_string, models


def run_cross_validation_process_test(info_string, models):
    batch_size = 8
    num_fold = 0
    yfull_test = []
    test_id = []
    nfolds = len(models)

    for i in range(nfolds):
        model = models[i]
        num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        test_data, test_id = read_and_normalize_test_data()
        predictions = model.predict(test_data, batch_size=batch_size, verbose=1)
        yfull_test.append(predictions)

    test_res = merge_several_folds_mean(yfull_test, nfolds)
    info_string = 'loss_' + info_string \
                + '_folds_' + str(nfolds)
    create_submission(test_res, test_id, info_string)
    create_submission(predictions, test_id, info_string)

    
def plot_confusion_matrix(cls_pred):
    # This is called from print_validation_accuracy() below.

    # cls_pred is an array of the predicted class-number for
    # all images in the validation set.

    # Get the true classifications for the test-set.
    cls_true = [classes[np.argmax(x)] for x in labels_valid]
    
    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred,
                          labels=classes)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.matshow(cm)

    # Make various adjustments to the plot.
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')

    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()

    
def print_validation_accuracy(show_example_errors=False,
                              show_confusion_matrix=False):
    
    test_batch_size = 4
    
    # Number of images in the validation set.
    num_test = len(labels_valid)
    
    cls_pred = np.zeros(shape=num_test, dtype=np.int)
    
    i = 0
    # iterate through batches and create list of predictions
    while i < num_test:
        # The ending index for the next batch is denoted j.
        j = min(i + test_batch_size, num_test)

        # Get the images from the test-set between index i and j.
        images = sample_valid[i:j, :]

        # Calculate the predicted class using TensorFlow.
        cls_pred[i:j] = [np.argmax(x) for x in model.predict(images)]

        # Set the start-index for the next batch to the
        # end-index of the current batch.
        i = j
    
    # Convenience variable for the true class-numbers of the validation set.
    cls_pred = np.array([classes[x] for x in cls_pred])
    cls_true = np.array([classes[np.argmax(x)] for x in labels_valid])

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on validation set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)
"""

if __name__ == '__main__':
    print('Keras version: {}'.format(keras_version))
    num_folds = 5
    info_string, models = run_train()
    ensemble_predict(info_string, models)
    #plot_confusion_matrix(cls_pred)
    #info_string, models = run_cross_validation_create_models(num_folds)
    #print_validation_accuracy(show_example_errors=False, show_confusion_matrix=True)

    #info_string, models = run_cross_validation_process_test(info_string, models)
