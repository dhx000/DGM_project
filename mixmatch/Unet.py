import tensorflow as tf
from tensorflow.python.keras.layers import BatchNormalization,LeakyReLU,ReLU,Activation,Dense,Conv2D,Input,UpSampling2D,MaxPooling2D,Concatenate
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizers import Adam,RMSprop
import numpy as np
from Layers import InstanceNormalization
import utils
import config
class Unet(object):
    def __init__(self,lamda=10.0):
        self.img_shape=config.img_shape
        self.gt_shape=config.img_shape
        self.unet=self.build_unet()
        '''
        self.patch_size=30
        self.generator=self.build_generator()
        self.discriminator=self.build_discriminator()
        self.optimpizer=Adam(0.0002,beta_1=0.5)
        self.discriminator.compile(optimizer=self.optimpizer,loss='mse')
        self.lamda=lamda
        self.critie_model=self.build_model()
        '''
    def Conv3x3(self,input,filter_num):
        x = Conv2D(filter_num,3,2,padding='same')(input)
        return x
    def build_unet(self):
        input_img = Input(shape=self.img_shape)

        # (N,240,240,1) -> (N,120,120,64)
        conv1 = Conv2D(32, 3, 1, padding='same')(input_img)
        conv1 = InstanceNormalization()(conv1)
        conv1 = ReLU()(conv1)
        conv1 = Conv2D(32, 3, 1, padding='same')(conv1)
        conv1 = InstanceNormalization()(conv1)
        conv1 = ReLU()(conv1)
        pool1 = MaxPooling2D()(conv1)

        # (N,120,120,64) -> (N,60,60,128)
        conv2 = Conv2D(64, 3, 1, padding='same')(pool1)
        conv2 = InstanceNormalization()(conv2)
        conv2 = ReLU()(conv2)
        conv2 = Conv2D(64, 3, 1, padding='same')(conv2)
        conv2 = InstanceNormalization()(conv2)
        conv2 = ReLU()(conv2)
        pool2 = MaxPooling2D()(conv2)

        # (N,60,60,128) -> (N,30,30,256)
        conv3 = Conv2D(128, 3, 1, padding='same')(pool2)
        conv3 = InstanceNormalization()(conv3)
        conv3 = ReLU()(conv3)
        conv3 = Conv2D(128, 3, 1, padding='same')(conv3)
        conv3 = InstanceNormalization()(conv3)
        conv3 = ReLU()(conv3)
        pool3 = MaxPooling2D()(conv3)

        # (N,30,30,128) -> (N,15,15,256)
        conv4 = Conv2D(256, 3, 1, padding='same')(pool3)
        conv4 = InstanceNormalization()(conv4)
        conv4 = ReLU()(conv4)
        conv4 = Conv2D(256, 3, 1, padding='same')(conv4)
        conv4 = InstanceNormalization()(conv4)
        conv4 = ReLU()(conv4)
        pool4 = MaxPooling2D()(conv4)

        # (N,15,15,256) -> (N,15,15,512)
        conv5 = Conv2D(512, 3, 1, padding='same')(pool4)
        conv5 = InstanceNormalization()(conv5)
        conv5 = ReLU()(conv5)
        conv5 = Conv2D(512, 3, 1, padding='same')(conv5)
        conv5 = InstanceNormalization()(conv5)
        conv5 = ReLU()(conv5)

        # (N,15,15,512) -> (N,30,30,256)
        up1 = UpSampling2D(size=(2, 2))(conv5)
        conv6 = Concatenate(axis=-1)([up1, conv4])
        conv6 = Conv2D(256, 3, 1, padding='same')(conv6)
        conv6 = InstanceNormalization()(conv6)
        conv6 = ReLU()(conv6)
        conv6 = Conv2D(256, 3, 1, padding='same')(conv6)
        conv6 = InstanceNormalization()(conv6)
        conv6 = ReLU()(conv6)

        # (N,30,30,256) -> (N,60,60,128)
        up2 = UpSampling2D(size=(2, 2))(conv6)
        conv7 = Concatenate(axis=-1)([up2, conv3])
        conv7 = Conv2D(128, 3, 1, padding='same')(conv7)
        conv7 = InstanceNormalization()(conv7)
        conv7 = ReLU()(conv7)
        conv7 = Conv2D(128, 3, 1, padding='same')(conv7)
        conv7 = InstanceNormalization()(conv7)
        conv7 = ReLU()(conv7)

        # (N,60,60,128) -> (N,120,120,64)
        up3 = UpSampling2D(size=(2, 2))(conv7)
        conv8 = Concatenate(axis=-1)([up3, conv2])
        conv8 = Conv2D(64, 3, 1, padding='same')(conv8)
        conv8 = InstanceNormalization()(conv8)
        conv8 = ReLU()(conv8)
        conv8 = Conv2D(64, 3, 1, padding='same')(conv8)
        conv8 = InstanceNormalization()(conv8)
        conv8 = ReLU()(conv8)

        # (N,120,120,64) -> (N,240,240,32)
        up4 = UpSampling2D(size=(2, 2))(conv8)
        conv9 = Concatenate(axis=-1)([up4, conv1])
        conv9 = Conv2D(32, 3, 1, padding='same')(conv9)
        conv9 = InstanceNormalization()(conv9)
        conv9 = ReLU()(conv9)
        conv9 = Conv2D(32, 3, 1, padding='same')(conv9)
        conv9 = InstanceNormalization()(conv9)
        conv9 = ReLU()(conv9)

        output = Conv2D(1, 1, 1, padding='same', activation='sigmoid')(conv9)

        self.unet = Model(input_img,output)
        self.unet.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = [utils.f1])
        return self.unet
    def get_unet(self):
        return self.unet
    def get_mask(self,Image):
        return self.unet.predict(Image)
    def save(self,name='unet.h5'):
        self.unet.save_weights(name)
if __name__ == '__main__':
    unet=Unet()
    model=unet.get_unet()
    model.summary()

