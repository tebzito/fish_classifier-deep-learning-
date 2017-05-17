import warnings
warnings.filterwarnings("ignore")
import multiprocessing
import os, glob
os.environ['THEANO_FLAGS'] = "floatX=float32,openmp=True" 
os.environ['OMP_NUM_THREADS'] = str(multiprocessing.cpu_count())
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from sklearn.model_selection import GridSearchCV, KFold
import pandas as pd
import numpy as np
np.random.seed(2017)
import cv2

train = pd.DataFrame([[i.split('/')[3],i.split('/')[4],i] for i in glob.glob('train/*/*.jpg')])
train.columns = ['type','image','path']

train_data = []
train_target = []
folders = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']

# Normalizing the data
for i in range(len(train)):
    img = cv2.imread(train['path'][i])
    img = cv2.resize(img, (40, 40), cv2.INTER_LINEAR)
    train_data.append(img)
    train_target.append(folders.index(train['type'][i]))
train_data = np.array(train_data, dtype=np.uint8)
train_target = np.array(train_target, dtype=np.uint8)
train_data = train_data.transpose((0, 3, 1, 2))
train_data = train_data.astype('float32') / 255

def create_model(lr=1e-2, decay=1e-6, momentum=0.9, nesterov=True):
    model = Sequential()
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 40, 40), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(4, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(ZeroPadding2D((1, 1), dim_ordering='th'))
    model.add(Convolution2D(8, 3, 3, activation='relu', dim_ordering='th'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), dim_ordering='th'))

    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(8, activation='softmax'))

    sgd = SGD(lr=lr, decay=decay, momentum=momentum, nesterov=nesterov)
    model.compile(optimizer=sgd, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, nb_epoch=5, batch_size=20, verbose=2)
lrs=[1e-2]
decays=[1e-6]
momentums=[0.8] 
nesterovs=[True]
epochs = np.array([5])
batches = np.array([20])
param_grid = dict(lr=lrs, decay=decays, momentum=momentums, nesterov=nesterovs, nb_epoch=epochs, batch_size=batches)
grid = GridSearchCV(estimator=model, cv=KFold(2), param_grid=param_grid, verbose=20)
grid_result = grid.fit(train_data, train_target)

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
for params, mean_score, scores in grid_result.grid_scores_:
    print("%f (%f) with: %r" % (scores.mean(), scores.std(), params))

files = sorted(glob.glob('../input/test_stg1/*.jpg'))
test_data = []
test_id = []
for fl in files:
    flbase = os.path.basename(fl)
    img = cv2.imread(fl)
    img = cv2.resize(img, (40, 40), cv2.INTER_LINEAR)
    test_data.append(img)
    test_id.append(flbase)
test_data = np.array(test_data, dtype=np.uint8)
test_data = test_data.transpose((0, 3, 1, 2))
test_data = test_data.astype('float32')  / 255

test_prediction = grid_result.predict_proba(test_data)
result1 = pd.DataFrame(test_prediction, columns=folders)
result1.loc[:, 'image'] = pd.Series(test_id, index=result1.index)
result1.to_csv('submission.csv', index=False)
