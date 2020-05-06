# %% [code]
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from datetime import datetime
import tensorflow as tf

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

def main_seg(segnet="unet"):
    IMAGE_LIB = '../input/2d_images/'
    MASK_LIB = '../input/2d_masks/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2

    print("Running Segmentation")
    write_seg = True

    # get train/test data
    x_train, x_val, y_train, y_val = getdata_seg(IMAGE_LIB, MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

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

    # model train summary
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.legend(['Loss', 'Validation Loss'])
    plt.show()
    plt.plot(hist.history['dice_coef'], color='b')
    plt.plot(hist.history['val_dice_coef'], color='r')
    plt.legend(['Dice Coefficient', 'Validation Dice Coefficient'])
    plt.show()
    plt.plot(hist.history['IoU'], color='b')
    plt.plot(hist.history['val_IoU'], color='r')
    plt.legend(['IoU', 'Validation IoU'])
    plt.show()

    # testing
    model.load_weights(segnet+'.h5')

    # test results
    y_hat = model.predict(x_val)
    #fig, ax = plt.subplots(1,3,figsize=(12,6))
    #ax[0].imshow(x_val[0,:,:,0], cmap='gray')
    #ax[1].imshow(y_val[0,:,:,0])
    #ax[2].imshow(y_hat[0,:,:,0])
    print(y_hat.shape)
    print("Testing done")
    # TODO
    #if write_seg:
    #    for i in range(len(y_hat)):
    #        imwrite('segmented_image.png', y_hat)

def main_classification():
    MASK_LIB = '../input/2d_images/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2
    EPOCH = 100

    # get train/test data
    x_train, x_val, y_train, y_val = getdata_classification(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)
    print("Data reading completed")

    # get model
    model = genmodel_classification(x_train.shape[1:])
    #model.summary()

    # compile model
    model.compile(optimizer=Adam(2e-4), loss='binary_crossentropy', metrics=['accuracy'])
    #model.compile(optimizer=Adam(2e-4), loss='mean_absolute_percentage_error')#, metrics=[dice_coef])
    #loss = keras.losses.huber_loss(delta=1.0)
    #model.compile(optimizer=Adam(2e-4), loss=loss)#, metrics=[dice_coef])

    # train setup
    weight_saver = ModelCheckpoint('covid19_classification.h5', save_best_only=True, save_weights_only=True)
    #annealer = LearningRateScheduler(lambda x: 1e-3 * 0.8 ** x)

    # train model
    hist = model.fit(x_train, y_train, batch_size=1, epochs=EPOCH, validation_data = (x_val, y_val),
                        verbose=2, callbacks=[weight_saver])#, annealer])

    print("Training completed")
    # model train summary
    plt.plot(hist.history['loss'], color='b')
    plt.plot(hist.history['val_loss'], color='r')
    plt.xlabel('No of epochs')
    plt.ylabel('Loss')
    plt.legend(['Training Loss', 'Validation Loss'])
    plt.show()

    # testing
    model.load_weights('covid19_classification.h5')

    # test results
    y_hat = model.predict(x_val)
    # convert probability to labels
    y_hat = [0 if val < 0.5 else 1 for val in y_hat]

    print("Classification Report : ")
    print(classification_report(y_val, y_hat))
    print("Confusion Matrix : ")
    print(confusion_matrix(y_val, y_hat))
    precision, recall, _ = precision_recall_curve(y_val, y_hat)
    print("Precision = ", precision)
    print("Recall = ", recall)
    # PR curve
    plt.plot(recall, precision, marker='.', label='PR Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.legend()
    plt.show()
    # ROC Curve
    false_positive_rate, true_positive_rate, _ = roc_curve(y_val, y_hat)
    print("False Positive Rate = ", false_positive_rate)
    print("True Positive Rate = ", true_positive_rate)
    plt.plot(false_positive_rate, true_positive_rate, marker='*', label='ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

    print("Testing done")


def main_classification_vgg16():
    MASK_LIB = '../input/2d_images/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2
    EPOCH = 100

    # get train/test data
    x_train, x_val, y_train, y_val = getdata_classification(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)

    # get model
    model = genmodel_classification_vgg16(x_train.shape[1:])
    opt = SGD(lr=0.001, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

    weight_saver = ModelCheckpoint('covid19_classification_vgg16.h5', save_best_only=True, save_weights_only=True)

    # train
    hist = model.fit(x_train, y_train, batch_size=1, epochs=EPOCH, validation_data = (x_val, y_val),
                        verbose=2, callbacks=[weight_saver])
    y_hat = model.predict(x_val)
    print(y_hat)


def extract_feature_histogram(image):
    #grayscale histogram calculation
    hist = cv2.calcHist(image, [0], None, [256], [0, 256])

    return hist.flatten()

def main_classification_svm():
    IMAGE_LIB = '../input/2d_images/'
    IMG_HEIGHT, IMG_WIDTH = 32, 32
    TEST_RATIO = 0.2

    feature_x_train = []
    label_y_train = []
    feature_x_val = []
    label_y_val = []
    x_train, x_val, y_train, y_val = getdata_classification(IMAGE_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO)
    # get training features
    for i in range(x_train.shape[0]):
        image = x_train[i]
        label = y_train[i]

        hist = extract_feature_histogram(image)
        feature_x_train.append(hist)
        label_y_train.append(label)
    # get validation features
    for i in range(x_val.shape[0]):
        image = x_val[i]
        label = y_val[i]

        hist = extract_feature_histogram(image)
        feature_x_val.append(hist)
        label_y_val.append(label)

    weight_saver = ModelCheckpoint('covid19_classification_svm.h5', save_best_only=True, save_weights_only=True)
    svm_model = LinearSVC(penalty='l2', loss='squared_hinge')
    svm_model.fit(feature_x_train, label_y_train)

    y_prediction = svm_model.predict(feature_x_val)
    print("Classification Report : ")
    print(classification_report(label_y_val, y_prediction))
    print("Confusion Matrix : ")
    print(confusion_matrix(label_y_val, y_prediction))
    #plt.plot(hist.history['loss'], color='b')
    #plt.plot(hist.history['val_loss'], color='r')
    #plt.show()


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
#main_seg("unetplusplus")
main_classification()
#main_classification_svm()
#main_classification_vgg16()
#main_reg()
#main_reg_all()
