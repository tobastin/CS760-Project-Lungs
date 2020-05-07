import csv
import os
import numpy as np # linear algebra
import cv2
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.model_selection import train_test_split

def getdata_seg(IMAGE_LIB, MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO):
    visualize_idx = [0, 100, 200, 300, 400]
    all_images = [x for x in sorted(os.listdir(IMAGE_LIB)) if x[-4:] == '.png']

    visualize_x_data = np.empty((len(visualize_idx), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    visualize_y_data = np.empty((len(visualize_idx), IMG_HEIGHT, IMG_WIDTH), dtype='float32')

    x_data = np.empty((len(all_images)-len(visualize_idx), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    count = 0
    count_1 = 0
    for i, name in enumerate(all_images):
        im = cv2.imread(IMAGE_LIB + name, 0).astype("int16").astype('float32')
        im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_LANCZOS4)
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
        if i in visualize_idx:
            visualize_x_data[count] = im
            count = count+1
        else:
            x_data[count_1] = im
            count_1 = count_1+1

    y_data = np.empty((len(all_images)-len(visualize_idx), IMG_HEIGHT, IMG_WIDTH), dtype='float32')
    count = 0
    count_1 = 0
    for i, name in enumerate(all_images):
        im = cv2.imread(MASK_LIB + name, 0).astype('float32')/255.
        im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_NEAREST)
        if i in visualize_idx:
            visualize_y_data[count] = im
            count = count+1
        else:
            y_data[count_1] = im
            count_1 = count_1+1
    #print(visualize_x_data.shape)
    #print(visualize_y_data.shape)
    #assert(0)
    # %% [code]
    #fig, ax = plt.subplots(1,2, figsize = (8,4))
    #ax[0].imshow(x_data[0], cmap='gray')
    #ax[1].imshow(y_data[0], cmap='gray')
	#plt.show()

    x_data = x_data[:,:,:,np.newaxis]
    y_data = y_data[:,:,:,np.newaxis]
    visualize_x_data = visualize_x_data[:,:,:,np.newaxis]
    visualize_y_data = visualize_y_data[:,:,:,np.newaxis]
    x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = TEST_RATIO)
    return x_train, x_val, y_train, y_val, visualize_x_data, visualize_y_data

def getdata_classification(MASK_LIB, IMG_HEIGHT, IMG_WIDTH, TEST_RATIO):
    with open('../input/classification_gt.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        x_data = []
        y_data = []
        for row in readCSV:
            # read mask image
            if row[0] == 'img_id':
                continue
            #print(MASK_LIB + row[0] + ".png")
            #im = cv2.imread(MASK_LIB + row[0] + ".png", 0).astype('float32')/255.0
            im = cv2.imread(MASK_LIB + row[0] + ".png", 0).astype("int16").astype('float32')
            #cv2.imshow('', im)

            im = cv2.resize(im, dsize=(IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
            x_data.append(im)

            y = []
            #for i in range(1,7):
            #    y.append(np.float(row[i]))
            y_data.append(np.float(row[1]))

    x_data = np.array(x_data)
    y_data = np.array(y_data)
    x_data = x_data[:,:,:,np.newaxis]
    #y_data = y_data[:,:,np.newaxis]
    print(x_data.shape)
    print(y_data.shape)
    #assert(0)
    #print(y_data)
    #x_train, x_val, y_train, y_val = train_test_split(x_data, y_data, test_size = 0.5)
    return train_test_split(x_data, y_data, test_size = TEST_RATIO)
