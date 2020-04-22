# %% [code]
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split

# %% [code]
import keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import Adam
from keras.regularizers import l2
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.callbacks import LearningRateScheduler, ModelCheckpoint

from genmodel import *
from getdata import *
from utils import *

def main_seg():
    IMAGE_LIB = '../input/2d_images/'
    MASK_LIB = '../input/2d_masks/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2

    write_seg = True

    # get train/test data
    x_train, x_val, y_train, y_val = getdata_seg(IMAGE_LIB, MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    # get model
    model = genmodel_seg(x_train.shape[1:])
    #model.summary()

    # train model
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef])

    # train setup
    weight_saver = ModelCheckpoint('lung_seg.h5', monitor='val_dice_coef', save_best_only=True, save_weights_only=True)
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

    # train
    hist = model.fit_generator(my_generator(x_train, y_train, 8),
                               steps_per_epoch = 200,
                               validation_data = (x_val, y_val),
                               epochs=10, verbose=2,
                               callbacks = [weight_saver, annealer])

    # model train summary
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()
    plt.plot(hist.history['dice_coef'], color='b')
    plt.plot(hist.history['val_dice_coef'], color='r')
    plt.show()

    # testing
    model.load_weights('lung_seg.h5')
    #plt.imshow(model.predict(x_train[10].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')

    # test results
    y_hat = model.predict(x_val)
    fig, ax = plt.subplots(1,3,figsize=(12,6))
    ax[0].imshow(x_val[0,:,:,0], cmap='gray')
    ax[1].imshow(y_val[0,:,:,0])
    ax[2].imshow(y_hat[0,:,:,0])
    print(len(y_hat),len(y_hat[0]))

    # TODO
    #if write_seg:
    #    for i in range(len(y_hat)):
    #        imwrite('segmented_image.png', y_hat)

def main_reg():
    MASK_LIB = '../input/2d_masks/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2
    EPOCH = 100
    p_feat_id = 3
    # get train/test data
    x_train, x_val, y_train, y_val = getdata_reg(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    # get model
    model = genmodel_reg(x_train.shape[1:])
    #model.summary()

    # train model
    model.compile(optimizer=Adam(2e-4), loss='mean_squared_error')#, metrics=[dice_coef])
    #model.compile(optimizer=Adam(2e-4), loss='mean_absolute_percentage_error')#, metrics=[dice_coef])
    #loss = keras.losses.huber_loss(delta=1.0)
    #model.compile(optimizer=Adam(2e-4), loss=loss)#, metrics=[dice_coef])

    # train setup
    weight_saver = ModelCheckpoint('lung_reg.h5', save_best_only=True, save_weights_only=True)
    #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

    # train
    hist = model.fit(x_train, y_train[:,p_feat_id], batch_size=1, epochs=EPOCH, validation_data = (x_val, y_val[:,p_feat_id]),
                        verbose=2, callbacks=[weight_saver])#, annealer])

    # model train summary
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.show()

    # testing
    model.load_weights('lung_reg.h5')
    #plt.imshow(model.predict(x_train[10].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')

    # test results
    y_hat = model.predict(x_val)
    e = [abs(y_val[i,p_feat_id]-y_hat[i])*100/y_val[i,p_feat_id] for i in range(len(y_val[:,p_feat_id]))]

    print("\nAverage % error : ",sum(e)/len(e))
    #plt.plot(hist.history['loss'], color='b')
    #plt.plot(hist.history['val_loss'], color='r')
    #plt.show()

    print("Yhat")
    print(y_hat)
    print("Yval")
    print(y_val[:,p_feat_id])

    print("Percentage error : ",get_percent_error(y_hat,y_val[:,p_feat_id].reshape((y_val.shape[0],1))))

#main_seg()
main_reg()
