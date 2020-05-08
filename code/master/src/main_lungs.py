# %% [code]
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve,r2_score
from datetime import datetime
import tensorflow as tf
#from sklearn.metrics import r2_score

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
import shutil

def main_seg(segnet="unet"):
    IMAGE_LIB = '../input/2d_images/'
    MASK_LIB = '../input/2d_masks/'
    WRITE_DIR = './'+segnet+'_segmentation_results/'
    PLOT_DIR = './'+segnet+'_segmentation_results/plots/'
    IMG_HEIGHT, IMG_WIDTH = 64, 64
    TEST_RATIO = 0.2

    print("Running Segmentation")

    write_seg = True

    # get train/test data
    x_train, x_val, y_train, y_val, visualize_x_data, visualize_y_data = getdata_seg(IMAGE_LIB, MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    print("Data reading completed")
    # get model
    if segnet == "unetplus":
        print("Using UNet+ Model")
        model = genmodel_seg_unetplus(x_train.shape[1:])
    elif segnet == "unetplusplus":
        print("Using UNet++ Model")
        model = genmodel_seg_unetplusplus(x_train.shape[1:])
    elif segnet == "unete":
        print("Using UNet-Ensemble Model")
        model = genmodel_seg_unete(x_train.shape[1:])
    elif segnet == "updownnet":
        print("Using Up-DownSample Model")
        model = genmodel_seg_updownnet(x_train.shape[1:])
    elif segnet == "fconvnet":
        print("Using Fully Convolutional Network Model")
        model = genmodel_seg_fconvnet(x_train.shape[1:])
    else:
        print("Using UNet Model")
        model = genmodel_seg_unet(x_train.shape[1:])

    #model.summary()

    # train model
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=[dice_coef, IoU])

    # train setup
    weight_saver = ModelCheckpoint(segnet+'.h5', monitor='val_dice_coef', save_best_only=True, save_weights_only=True)
    annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

    # train
    hist = model.fit_generator(my_generator(x_train, y_train, 8),
                               steps_per_epoch = 200,
                               validation_data = (x_val, y_val),
                               epochs=100, verbose=2,
                               callbacks = [weight_saver, annealer])
    print("Training completed")
    # testing
    model.load_weights(segnet+'.h5')

    # test results
    y_hat = model.predict(x_val)

    print("Testing done")
    # Visualize data
    visualize_y_hat = model.predict(visualize_x_data)

    #threshold
    #visualize_y_hat = [0 if val < 0.5 else 1 for val in visualize_y_hat]
    visualize_y_hat[visualize_y_hat < 0.5] = 0
    visualize_y_hat[visualize_y_hat >= 0.5] = 1
    if write_seg:
        if os.path.isdir(WRITE_DIR):
            shutil.rmtree(WRITE_DIR)
        os.mkdir(WRITE_DIR)
        for i in range(len(visualize_y_hat)):
            write_file_res = WRITE_DIR+'segmented_image_'+str(i)+'.png'
            write_file_gt = WRITE_DIR+'gt_image_'+str(i)+'.png'
            res_image = np.array(255*visualize_y_hat[i], dtype = 'uint8')
            gt_image = np.array(255*visualize_y_data[i], dtype = 'uint8')
            cv2.imwrite(write_file_res, res_image)
            cv2.imwrite(write_file_gt, gt_image)

    print("Segmentation results written")

    os.mkdir(PLOT_DIR)
    # model train summary
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.xlabel('No of epochs')
    plt.ylabel('Loss')
    plt.legend(['Loss', 'Validation Loss'])
    plt.savefig(PLOT_DIR+'loss.png')
    plt.clf()
    #plt.show()
    plt.plot(hist.history['dice_coef'], color='b')
    plt.plot(hist.history['val_dice_coef'], color='r')
    plt.xlabel('No of epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend(['Dice Coefficient', 'Validation Dice Coefficient'])
    #plt.show()
    plt.savefig(PLOT_DIR+'dice.png')
    plt.clf()
    plt.plot(hist.history['IoU'], color='b')
    plt.plot(hist.history['val_IoU'], color='r')
    plt.xlabel('No of epochs')
    plt.ylabel('IoU')
    plt.legend(['IoU', 'Validation IoU'])
    plt.savefig(PLOT_DIR+'iou.png')
    plt.clf()
    #plt.show()


def main_reg():
    MASK_LIB = '../input/2d_masks/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2
    EPOCH = 100
    p_feat_id = 5
    # get train/test data
    x_train, x_val, y_train, y_val = getdata_reg(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    # get model
    model = genmodel_reg_f4(x_train.shape[1:])
    #model.summary()

    # train model
    model.compile(optimizer=Adam(2e-4), loss='mean_squared_error')#, metrics=[r2_keras])
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
    plt.xlabel('No of epochs')
    plt.ylabel('Loss')
    plt.legend(['Loss', 'Validation Loss'])
    plt.title("Regression Learning Curve ")
    #plt.savefig(PLOT_DIR+'loss.png')
    plt.show()
    plt.clf()
    
    # testing
    model.load_weights('lung_reg.h5')
    #plt.imshow(model.predict(x_train[10].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')

    # test results
    y_hat = model.predict(x_val)
    #e = [abs(y_val[i,p_feat_id]-y_hat[i])*100/y_val[i,p_feat_id] for i in range(len(y_val[:,p_feat_id]))]
    print("R2 score  :",r2_score(y_val[:,p_feat_id], y_hat))

    #print("\nAverage % error : ",sum(e)/len(e))
    #plt.plot(hist.history['loss'], color='b')
    #plt.plot(hist.history['val_loss'], color='r')
    #plt.show()

    #print("Yhat")
    #print(y_hat)
    #print("Yval")
    #print(y_val[:,p_feat_id])

    #print("Percentage error : ",get_percent_error(y_hat,y_val[:,p_feat_id].reshape((y_val.shape[0],1))))
'''
def main_reg_all():
    MASK_LIB = '../input/2d_masks/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2
    EPOCH = 400
    n_f = 6
    #p_feat_id = 3
    # get train/test data
    x_train, x_val, y_train, y_val = getdata_reg(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    y_hat = np.zeros(y_val.shape)
    #print(y_val.shape)
    #print(y_hat.shape)

    #return

    model = []
    for p_feat_id in range(n_f):
        print("Processing model : ",p_feat_id)
        # get model
        model.append(genmodel_reg(x_train.shape[1:]))
        #model.summary()

        # train model
        model[p_feat_id].compile(optimizer=Adam(2e-4), loss='mean_squared_error')#, metrics=[dice_coef])
        #model.compile(optimizer=Adam(2e-4), loss='mean_absolute_percentage_error')#, metrics=[dice_coef])
        #loss = keras.losses.huber_loss(delta=1.0)
        #model.compile(optimizer=Adam(2e-4), loss=loss)#, metrics=[dice_coef])

        # train setup
        weight_saver = ModelCheckpoint("lung_reg_"+str(p_feat_id)+ ".h5", save_best_only=True, save_weights_only=True)
        #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

        # train
        hist = model[p_feat_id].fit(x_train, y_train[:,p_feat_id], batch_size=1, epochs=EPOCH, validation_data = (x_val, y_val[:,p_feat_id]),
                            verbose=2, callbacks=[weight_saver])#, annealer])

        # model train summary
        #plt.plot(hist.history['loss'], color='b')
        #plt.plot(hist.history['val_loss'], color='r')
        #plt.show()

        # testing
        model[p_feat_id].load_weights("lung_reg_"+str(p_feat_id)+ ".h5")
        #plt.imshow(model.predict(x_train[10].reshape(1,IMG_HEIGHT, IMG_WIDTH, 1))[0,:,:,0], cmap='gray')

        # test results

        #temp = model[p_feat_id].predict(x_val).reshape(y_hat.shape[0])
        #print(temp.shape)
        #print(y_hat[:,p_feat_id].shape)
        #return
        y_hat[:,p_feat_id] = model[p_feat_id].predict(x_val).reshape(y_hat.shape[0])
        #e = [abs(y_val[i,p_feat_id]-y_hat[i])*100/y_val[i,p_feat_id] for i in range(len(y_val[:,p_feat_id]))]

    #print("\nAverage % error : ",sum(e)/len(e))
    #plt.plot(hist.history['loss'], color='b')
    #plt.plot(hist.history['val_loss'], color='r')
    #plt.show()

    #print("Yhat")
    #print(y_hat)
    #print("Yval")
    #print(y_val[:,p_feat_id])

    print("Percentage error : ",get_percent_error(y_hat,y_val))
'''
#main_seg("updownnet")
main_reg()
#main_reg_all()
