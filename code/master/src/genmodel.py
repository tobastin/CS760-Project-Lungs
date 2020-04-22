from keras.layers import *
from keras.models import Model

def genmodel_seg(input_shape):

    input_layer = Input(shape=input_shape)
    c1 = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)
    l = MaxPool2D(strides=(2,2))(c1)
    c2 = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l)
    l = MaxPool2D(strides=(2,2))(c2)
    c3 = Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='same')(l)
    l = MaxPool2D(strides=(2,2))(c3)
    c4 = Conv2D(filters=32, kernel_size=(1,1), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(c4), c3], axis=-1)
    l = Conv2D(filters=32, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(l), c2], axis=-1)
    l = Conv2D(filters=24, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = concatenate([UpSampling2D(size=(2,2))(l), c1], axis=-1)
    l = Conv2D(filters=16, kernel_size=(2,2), activation='relu', padding='same')(l)
    l = Conv2D(filters=64, kernel_size=(1,1), activation='relu')(l)
    l = Dropout(0.5)(l)
    output_layer = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(l)

    return Model(input_layer, output_layer)



def genmodel_reg(input_shape):

    input_layer = Input(shape=input_shape)
    l = Conv2D(filters=8, kernel_size=(3,3), activation='relu', padding='same')(input_layer)    #32x32 -> 32x32x8
    l = MaxPool2D(strides=(2,2))(l)    #32x32x8 -> 16x16x8
    l = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l) #16x16x8 -> 16x16x16
    l = MaxPool2D(strides=(2,2))(l)     # 16x16x16 -> 8x8x16
    l = Conv2D(filters=16, kernel_size=(3,3), activation='relu', padding='same')(l) # 8x8x16 -> 8x8x16
    l = MaxPool2D(strides=(2,2))(l)     #8x8x16 -> 4x4x16
    l = Flatten()(l)
    l = Dense(units = 256, activation = "relu")(l)
    l = Dense(units = 128, activation = "relu")(l)
    l = Dense(units = 16, activation = "relu")(l)
    output_layer = Dense(units = 1)(l)

    return Model(input_layer, output_layer)