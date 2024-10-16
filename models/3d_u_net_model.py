# -*- coding: utf-8 -*-
"""
# **3D-UNET For Multimodal Brain Tumor Segmentation**
"""

# importing all libraries

import tensorflow
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv3D, MaxPooling3D, UpSampling3D, concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv3D, Conv3DTranspose, MaxPooling3D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dense, Dropout,Maximum

# Convolutional blocks for the U_Net model
def conv_block(input_mat,num_filters,kernel_size,batch_norm):
  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(input_mat)
  X = BatchNormalization()(X)
  X = Activation('leaky_relu')(X)

  X = Conv3D(num_filters,kernel_size=(kernel_size,kernel_size,kernel_size),strides=(1,1,1),padding='same')(X)
  X = BatchNormalization()(X)
  X = Activation('leaky_relu')(X)

  return X

# Define the 3D U-Net model
def Unet_3d(input_img, n_filters=8, dropout=0.2, batch_norm=True):
    # Contracting path
    conv_1 = conv_block(input_img, n_filters, 3, batch_norm)
    pool_1 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_1)
    pool_1 = Dropout(dropout)(pool_1)

    conv_2 = conv_block(pool_1, n_filters * 2, 3, batch_norm)
    pool_2 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_2)
    pool_2 = Dropout(dropout)(pool_2)

    conv_3 = conv_block(pool_2, n_filters * 4, 3, batch_norm)
    pool_3 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_3)
    pool_3 = Dropout(dropout)(pool_3)

    conv_4 = conv_block(pool_3, n_filters * 8, 3, batch_norm)
    pool_4 = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2))(conv_4)
    pool_4 = Dropout(dropout)(pool_4)

    conv_5 = conv_block(pool_4, n_filters * 16, 3, batch_norm)

    # Expansive path
    up_conv_6 = Conv3DTranspose(n_filters * 8, 3, strides=(2, 2, 2), padding='same')(conv_5)
    up_conv_6 = concatenate([up_conv_6, conv_4])
    conv_6 = conv_block(up_conv_6, n_filters * 8, 3, batch_norm)
    conv_6 = Dropout(dropout)(conv_6)

    up_conv_7 = Conv3DTranspose(n_filters * 4, 3, strides=(2, 2, 2), padding='same')(conv_6)
    up_conv_7 = concatenate([up_conv_7, conv_3])
    conv_7 = conv_block(up_conv_7, n_filters * 4, 3, batch_norm)
    conv_7 = Dropout(dropout)(conv_7)

    up_conv_8 = Conv3DTranspose(n_filters * 2, 3, strides=(2, 2, 2), padding='same')(conv_7)
    up_conv_8 = concatenate([up_conv_8, conv_2])
    conv_8 = conv_block(up_conv_8, n_filters * 2, 3, batch_norm)
    conv_8 = Dropout(dropout)(conv_8)

    up_conv_9 = Conv3DTranspose(n_filters, 3, strides=(2, 2, 2), padding='same')(conv_8)
    up_conv_9 = concatenate([up_conv_9, conv_1])
    conv_9 = conv_block(up_conv_9, n_filters, 3, batch_norm)

    # Output layer
    outputs = Conv3D(4, (1, 1, 1), activation='softmax')(conv_9)

    model = Model(inputs=input_img, outputs=outputs)

    return model

#Building the 3D-Unet Model.
input_img = Input((128,128,128,4))
model = Unet_3d(input_img,8,0.1,True)

print(model.input_shape)
print(model.output_shape)

# printing the summary of the model
model.summary()
