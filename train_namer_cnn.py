# ------------------------------------------------------------------------------
#
# train_namer_cnn.py
#
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# import libraries
# ------------------------------------------------------------------------------
import os
import scipy.io as sio
import numpy as np
import keras
from keras.layers import Conv2D, Activation, BatchNormalization
from keras.models import Sequential
from keras.optimizers import Adam
import datetime
from keras.callbacks import ModelCheckpoint
import tensorflow
from keras.backend.tensorflow_backend import set_session


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ------------------------------------------------------------------------------
# class initialization
# ------------------------------------------------------------------------------
class SaveNetworkProgress(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.epoch_ind = []
        self.losses = []
        self.val_losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.epoch_ind.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        sio.savemat(tmp_progress_filename, dict([('val_losses', self.val_losses), ('losses',self.losses), ('epoch_ind', self.epoch_ind)]))


# ------------------------------------------------------------------------------
# script initialization
# ------------------------------------------------------------------------------

# set tensorflow environment to limit GPU memory usage, and select GPU
config = tensorflow.ConfigProto()
config.gpu_options.allow_growth = True
set_session(tensorflow.Session(config=config))

# intialize hardcoded variables
kernel_size = [3, 3]
output_size = 64
num_layers = 27
patch_size = 51

nepochs = 40
nbatch = 100 

learning_rate = .0001 

exp_name = '_n100_lr0001_'
data_fn = 'training_data.mat'  # email mhaskell@fas.harvard.edu for training data

# initialize paths and filenames (fn abbreviation) and variable names (vn abbreviation)
cur_path = os.getcwd()
data_path = cur_path
datestring = datetime.date.today().strftime("%Y-%m-%d")
tmp_progress_filename = './convergence_curves/' + datestring + exp_name + 'progress'

# load ground truth, training, and test data
tmd = sio.loadmat(data_fn) # temp mat data
print("Loaded mat data.")

x_test1 = tmd['x_test_pt1']
x_test2 = tmd['x_test_pt2']
y_test1 = tmd['y_test_pt1']
y_test2 = tmd['y_test_pt2']
x_test = np.concatenate((x_test1, x_test2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)
print("Did test data concatenation.")

x_train1 = tmd['x_train_pt1']
x_train2 = tmd['x_train_pt2']
x_train3 = tmd['x_train_pt3']
x_train4 = tmd['x_train_pt4']
x_train5 = tmd['x_train_pt5']
x_train6 = tmd['x_train_pt6']
x_train7 = tmd['x_train_pt7']
y_train1 = tmd['y_train_pt1']
y_train2 = tmd['y_train_pt2']
y_train3 = tmd['y_train_pt3']
y_train4 = tmd['y_train_pt4']
y_train5 = tmd['y_train_pt5']
y_train6 = tmd['y_train_pt6']
y_train7 = tmd['y_train_pt7']
x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7), axis=0)
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7), axis=0)
print("Did training data concatenation.")


# ------------------------------------------------------------------------------#
#                                 setup cnn                                     #
# ------------------------------------------------------------------------------#

model = Sequential()

# layer 1
model.add(Conv2D(output_size, kernel_size, input_shape=(patch_size, patch_size, 2), padding='same'))
model.add(Activation('relu'))

# mid layers
for layers in range(1, num_layers - 1):
    model.add(Conv2D(output_size, kernel_size, padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

# last layer
model.add(Conv2D(2, kernel_size, padding='same'))

adam_opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mse', optimizer=adam_opt, metrics=['accuracy'])
model.summary()
model.save(cur_path + '/models/' + datestring + exp_name + 'init_model.h5')

# ------------------------------------------------------------------------------
# % train cnn
# ------------------------------------------------------------------------------

save_progress = SaveNetworkProgress()
filepath = cur_path + '/model_weights/' + datestring + exp_name + 'weights-{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, save_progress]

hist = model.fit(x_train, y_train, epochs=nepochs, callbacks=callbacks_list, batch_size=nbatch, shuffle=True,
                 validation_data=(x_test, y_test))

# save
model.save(cur_path + '/models/' + datestring + exp_name + 'trained_model.h5')



