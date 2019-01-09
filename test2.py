# -*- coding: utf-8 -*-
"""
Created on Wed Dec 26 18:26:54 2018

@author: ADWANI
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 24 06:56:57 2018

@author: ADWANI
"""

#from skimage import io
import random
import skimage
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras

I1_dir='D:\Research\BBBC006_v1_images_z_11\BBBC006_v1_images_z_11\*.tif'
I2_dir='D:\Research\BBBC006_v1_images_z_12\BBBC006_v1_images_z_12\*.tif'
I3_dir='D:\Research\BBBC006_v1_images_z_13\BBBC006_v1_images_z_13\*.tif'
I4_dir='D:\Research\BBBC006_v1_images_z_14\BBBC006_v1_images_z_14\*.tif'
I5_dir='D:\Research\BBBC006_v1_images_z_15\BBBC006_v1_images_z_15\*.tif'
O_dir='D:\Research\BBBC006_v1_images_z_16\BBBC006_v1_images_z_16\*.tif'
I7_dir='D:\Research\BBBC006_v1_images_z_17\BBBC006_v1_images_z_17\*.tif'
I8_dir='D:\Research\BBBC006_v1_images_z_18\BBBC006_v1_images_z_18\*.tif'
I9_dir='D:\Research\BBBC006_v1_images_z_19\BBBC006_v1_images_z_19\*.tif'
I10_dir='D:\Research\BBBC006_v1_images_z_20\BBBC006_v1_images_z_20\*.tif'



I1=skimage.io.imread_collection(I1_dir)
Out=skimage.io.imread_collection(O_dir)


#To show the image
#for i in I1:
#    plt.imshow(i)
#    plt.show()
#    

#We know that all the images are  696x520 dimensions 
#x dir 696-128= 568
#y dir 520-128= 392  
def random_crop(image):
    x=random.randint(0,568)
    y=random.randint(0,392)
    cropped=image[x:x+128, y:y+128]
    return cropped

def seeImage(image):
    plt.imshow(image)
    plt.show()
    
def crop_together(image1,image2):
    x=random.randint(0,568)
    y=random.randint(0,392)
    c1=image1[x:x+128, y:y+128]
    c2=image2[x:x+128, y:y+128]
    return c1,c2



cropped_images1=[]
for i in I1:
    z=random_crop(i)
    cropped_images1.append(z)
    
CI_1=[]
CI_Out=[]
for i in range(len(Out)):
    crop1,out1=crop_together(I1[i],Out[i])
    CI_1.append(crop1)
    CI_Out.append(out1)          
    
    
input_size=(128,128,1)
#CI_1[0].shape    
#UNET Model
inputs = Input(input_size)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
drop5 = Dropout(0.5)(conv5)
up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
merge6 = concatenate([drop4,up6], axis = 3)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
merge7 = concatenate([conv3,up7], axis = 3)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
merge8 = concatenate([conv2,up8], axis = 3)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
merge9 = concatenate([conv1,up9], axis = 3)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

model = Model(input = inputs, output = conv10)
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
aCI_1 = np.asarray(CI_1)
aCI_Out = np.asarray(CI_Out)
x=CI_1[0].reshape(128,128,1)
y=CI_Out[0].reshape(128,128,1)
model.fit(aCI_1,aCI_Out)