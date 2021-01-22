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
import tensorflow as tf
from tensorflow.python.keras.backend import set_session


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
        #
    def on_epoch_end(self, epoch, logs={}):
        self.epoch_ind.append(epoch)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        sio.savemat(tmp_progress_filename, dict([('val_losses', self.val_losses), ('losses',self.losses), ('epoch_ind', self.epoch_ind)]))


# ------------------------------------------------------------------------------
# script initialization
# ------------------------------------------------------------------------------

# set tensorflow environment to limit GPU memory usage, and select GPU
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.compat.v1.Session(config=config))

'''UHN HPC Msg Raised
2021-01-22 01:03:18.686798: I tensorflow/core/platform/cpu_feature_guard.cc:142] This Tenso                                                                                              rFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN)to use the follow                                                                                              ing CPU instructions in performance-critical operations:  SSE4.1 SSE4.2 AVX AVX2 AVX512F FM                                                                                              A
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2021-01-22 01:03:18.809414: I tensorflow/core/platform/profile_utils/cpu_utils.cc:104] CPU                                                                                               Frequency: 2095074999 Hz
2021-01-22 01:03:18.813278: I tensorflow/compiler/xla/service/service.cc:168] XLA service 0                                                                                              x560d7d628f00 initialized for platform Host (this does not guarantee that XLA will be used)                                                                                              . Devices:
2021-01-22 01:03:18.813317: I tensorflow/compiler/xla/service/service.cc:176]   StreamExecu                                                                                              tor device (0): Host, Default Version
2021-01-22 01:03:18.813571: I tensorflow/core/common_runtime/process_util.cc:146] Creating                                                                                               new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads f                                                                                              or best performance.
'''

# intialize hardcoded variables
kernel_size = [3, 3]
output_size = 64
num_layers = 27
patch_size = 51

nepochs = 40
nbatch = 100

learning_rate = .0001

exp_name = r'_n100_lr0001_'
main_path = r'/cluster/projects/uludag/Brian'
data_path = main_path + r'/data/namer'
save_path = main_path + r'/moco-sigpy/namer'
data_fn = data_path + r'/training_data.mat'  # email mhaskell@fas.harvard.edu for training data

# initialize paths and filenames (fn abbreviation) and variable names (vn abbreviation)
# cur_path = os.getcwd()
# data_path = cur_path
datestring = datetime.date.today().strftime("%Y-%m-%d")
tmp_progress_filename = './convergence_curves/' + datestring + exp_name + 'progress'

# load ground truth, training, and test data
tmd = sio.loadmat(data_fn) # temp mat data
print("Loaded mat data.")

x_test1 = tmd['x_test_pt1']
x_test2 = tmd['x_test_pt2']
y_test1 = tmd['y_test_pt1']
y_test2 = tmd['y_test_pt2']

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

del tmd

x_test = np.concatenate((x_test1, x_test2), axis=0)
del x_test1, x_test2
y_test = np.concatenate((y_test1, y_test2), axis=0)
del y_test1, y_test2
print("Did test data concatenation.")

x_train = np.concatenate((x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7), axis=0)
del x_train1, x_train2, x_train3, x_train4, x_train5, x_train6, x_train7
y_train = np.concatenate((y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7), axis=0)
del y_train1, y_train2, y_train3, y_train4, y_train5, y_train6, y_train7
print("Did training data concatenation.")


# ------------------------------------------------------------------------------#
#                                 setup cnn                                     #
# ------------------------------------------------------------------------------#

model = Sequential()

''' UHN HPC Raise Msg
2021-01-22 01:38:09.863642: I tensorflow/core/common_runtime/process_util.cc:146] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
'''

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
model.save(save_path + '/models/' + datestring + exp_name + 'init_model.h5')

# ------------------------------------------------------------------------------
# % train cnn
# ------------------------------------------------------------------------------

save_progress = SaveNetworkProgress()
filepath = save_path + '/model_weights/' + datestring + exp_name + 'weights-{epoch:02d}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint, save_progress]

hist = model.fit(x_train, y_train, epochs=nepochs, callbacks=callbacks_list, batch_size=nbatch, shuffle=True,
                 validation_data=(x_test, y_test))

# save
model.save(save_path + '/models/' + datestring + exp_name + 'trained_model.h5')
