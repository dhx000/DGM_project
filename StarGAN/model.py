import tensorflow as tf
from tensorflow.python.keras.layers import Dense,Input,Reshape,Flatten, Dropout, Concatenate
from tensorflow.python.keras.layers import BatchNormalization, Activation, ZeroPadding2D,LeakyReLU
from tensorflow.python.keras.layers import Conv2D,UpSampling2D,ZeroPadding2D,Add,Lambda
from tensorflow.python.keras.models import Sequential,Model
from tensorflow.python.keras.optimizers import Adam,RMSprop
from tensorflow.python.keras.initializers import RandomNormal
from tensorflow.python.keras import backend as K
from keras.preprocessing import image
from Layers import InstanceNormalization
import matplotlib.pyplot as plt
import numpy as np
import random
import  cv2
import config
import os
from DataSet import Dataset
class CycleGAN():
    def __init__(self):
        # Input shape
        self.class_num=config.class_num
        self.img_shape = config.img_shape
        self.img_width =  config.img_width
        self.img_height = config.img_height
        self.img_channel = config.img_channel
        self.gf = 32
        self.df = 64
        self.patch = int(config.img_height // (2 ** 4))
        self.patch_size = (self.patch, self.patch, 1)
        self.s1=Dataset('./t1ce','./tice_label')
        self.s2=Dataset('./t2','./t2_label')
        self.target=Dataset('./OpenBayes','./Openbayes_label',need_resize=1)
        optimizer = RMSprop(0.0002)
        self.D=self.build_discriminator()
        self.G=self.build_generator()
        self.G.trainable=False
        real_img=Input(shape=config.img_shape)
        real_src,real_cls=self.D(real_img)
        fake_cls=Input(shape=(self.class_num,))
        fake_img=self.G([real_img,fake_cls])
        fake_src,fake_output=self.D(fake_img)
        self.Train_D=Model([real_img,fake_cls],[real_src,real_cls,fake_src,fake_output])
        self.Train_D.compile(loss=['mse',self.classification_loss,'mse',self.classification_loss],optimizer=optimizer,loss_weights=[1.0,1.0,1.0,1.0])

        self.G.trainable=True
        self.D.trainable=False
        real_x=Input(shape=self.img_shape)
        now_label=Input(shape=(self.class_num,))
        target_label=Input(shape=(self.class_num,))
        fake_x=self.G([real_x,target_label])
        fake_out_src,fake_out_cls=self.D(fake_x)
        x_rec=self.G([fake_x,now_label])
        self.train_G=Model([real_x,now_label,target_label],[fake_out_src,fake_out_cls,x_rec])
        self.train_G.compile(loss=['mse',self.classification_loss,'mae'],
                             optimizer=optimizer, loss_weights = [1.0, 1.0,1.0])

        '''
        self.AdataSet=Dataset('./done', './done_label',need_resize=1)
        
            to get different model
            change image path 
            for example t1ce to t1
        
        self.BdataSet=Dataset('./Data/2/t1','./Data/2/label')
        # Number of filters in the first layer of G and D

        self.patch = int(config.img_height //(2**4))
        self.patch_size = (self.patch,self.patch,1)
        # Loss weights
        self.lambda_cycle = 10.0                    # Cycle-consistency loss
        self.lambda_id = 0.1 * self.lambda_cycle    # Identity loss

        optimizer = RMSprop(0.0002)

        # Build and compile the discriminators
        self.d_B = self.build_discriminator()
        self.d_A = self.build_discriminator()

        self.conv_init = RandomNormal(0, 0.02)  # for convolution kernel
        self.gamma_init = RandomNormal(1., 0.02)

        self.d_A.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])
        self.d_B.compile(loss='mse',
            optimizer=optimizer,
            metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generators
        #-------------------------

        # Build the generators
        self.g_AB = self.resnet_6blocks()
        self.g_BA = self.resnet_6blocks()

        # Input images from both domains
        img_A = Input(shape=self.img_shape)
        img_B = Input(shape=self.img_shape)

        # Translate images to the other domain
        fake_B = self.g_AB(img_A)
        fake_A = self.g_BA(img_B)
        # Translate images back to original domain
        reconstr_A = self.g_BA(fake_B)
        reconstr_B = self.g_AB(fake_A)

        # Identity mapping of images
        img_A_id = self.g_BA(img_A)
        img_B_id = self.g_AB(img_B)

        # For the combined model we will only train the generators
        self.d_A.trainable = False
        self.d_B.trainable = False

        # Discriminators determines validity of translated images
        valid_A = self.d_A(fake_A)
        valid_B = self.d_B(fake_B)

        # Combined model trains generators to fool discriminators
        self.combined = Model(inputs=[img_A, img_B],
                              outputs=[ valid_A, valid_B,
                                        reconstr_A, reconstr_B,
                                        img_A_id, img_B_id ])
        self.combined.compile(loss=['mse', 'mse',
                                    'mae', 'mae',
                                    'mae', 'mae'],
                            loss_weights=[  1, 1,
                                            self.lambda_cycle, self.lambda_cycle,
                                            self.lambda_id, self.lambda_id ],
                            optimizer=optimizer)
        '''
    def classification_loss(self, Y_true, Y_pred) :
        return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=Y_true, logits=Y_pred))
    def normalize(self, **kwargs):
        # return batchnorm()#axis=get_filter_dim()
        return InstanceNormalization()
    def get_onehot_label(self,label):
        now_label=[0.]*self.class_num
        now_label[label]=1.
        now_label=np.array(now_label).reshape([1,self.class_num])
        return now_label
    def resnet_block(self, input, dim, ks=(3, 3), strides=(1, 1)):
        x = ZeroPadding2D((1, 1))(input)
        x = Conv2D(dim, ks, strides=strides, kernel_initializer=self.conv_init)(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        x = ZeroPadding2D((1, 1))(x)
        x = Conv2D(dim, ks, strides=strides, kernel_initializer=self.conv_init)(x)
        x = InstanceNormalization()(x)
        res = Add()([input, x])
        return res

    def resnet_6blocks(self, ngf=64, **kwargs):

        input = Input(config.img_shape)
        x = ZeroPadding2D((3, 3))(input)
        x = Conv2D(ngf, (7, 7), kernel_initializer=self.conv_init)(x)
        x = InstanceNormalization()(x)
        x = Activation('relu')(x)

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            x = Conv2D(ngf * mult * 2, (3, 3),
                       padding='same', strides=(2, 2),
                       kernel_initializer=self.conv_init)(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)

        mult = 2 ** n_downsampling
        for i in range(6):
            x = self.resnet_block(x, ngf * mult)

        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            f = int(ngf * mult / 2)
            x = UpSampling2D()(x)
            x = InstanceNormalization()(x)
            x = Activation('relu')(x)

        x = ZeroPadding2D((3, 3))(x)
        x = Conv2D(config.img_channel, (7, 7), kernel_initializer=self.conv_init)(x)
        x = Activation('tanh')(x)

        model = Model(inputs=input, outputs=[x])
        print('Model resnet 6blocks:')
        model.summary()
        return model

    def build_generator(self):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            d = InstanceNormalization()(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = InstanceNormalization()(u)

            u = Concatenate()([u, skip_input])
            return u

        # Image input
        input_img = Input(shape=self.img_shape)
        inp_c = Input(shape=(self.class_num,))
        c = Lambda(lambda x: K.repeat(x, self.img_width *self.img_height))(inp_c)
        c = Reshape((self.img_width, self.img_height, self.class_num))(c)
        d0 = Concatenate()([input_img, c])
        # Downsampling
        d1 = conv2d(d0, self.gf)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)

        # Upsampling
        u1 = deconv2d(d4, d3, self.gf*4)
        u2 = deconv2d(u1, d2, self.gf*2)
        u3 = deconv2d(u2, d1, self.gf)

        u4 = UpSampling2D(size=2)(u3)
        output_img = Conv2D(self.img_channel, kernel_size=4, strides=1, padding='same', activation='tanh')(u4)

        return Model([input_img,inp_c], output_img)

    def build_discriminator(self):

        def d_layer(layer_input, filters, f_size=4, normalization=True):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)

            if normalization:
                d = InstanceNormalization()(d)

            return d

        img = Input(shape=self.img_shape)

        d1 = d_layer(img, self.df, normalization=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8)

        validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        class_cls= Conv2D(self.class_num, kernel_size=15, strides=1, padding='valid')(d4)
        class_cls= Reshape((self.class_num, ))(class_cls)
        model = Model(img,[validity,class_cls])
        return model
    def train(self, epochs, batch_size=1, sample_interval=50):

        #start_time = datetime.datetime.now()
        # Adversarial loss ground truths
        valid = np.ones((batch_size,) + self.patch_size)
        fake = np.zeros((batch_size,) + self.patch_size)
        self.train_G.summary()
        self.Train_D.summary()
        for epoch in range(epochs):
            for batch_i in range(0,1000):
                rand_number=random.randint(0,1)
                target_label=2
                target_label=self.get_onehot_label(target_label)
                now_img=None
                now_label=None
                if (rand_number==0):
                    now_img,_=self.s1.get_one_data()
                    now_label=self.get_onehot_label(0)
                if (rand_number==1):
                    now_img,_=self.s2.get_one_data()
                    now_label=self.get_onehot_label(1)
                if (rand_number==2):
                    now_img,_=self.target.get_one_data()
                    now_label=self.get_onehot_label(2)
                #print(now_label.shape)
                #print(target_label.shape)
                #print(self.D.output)
                d_loss=self.Train_D.train_on_batch([now_img,target_label],[valid,now_label,fake,target_label],class_weight=[1.0,1.0,1.0,1.0])
                g_loss=self.train_G.train_on_batch([now_img,now_label,target_label],[valid,target_label,now_img],class_weight=[1.0,1.0,1.0])
                print("[Epoch %d/%d][Batch %d/%d]"%(epoch,epochs,batch_i,1000))
                print("[d loss: %.3f][real dloss: %.3f][real cl: %.6f][fake dloss: %.3f][fake cl:%.6f]"%(d_loss[0],d_loss[1],d_loss[2],d_loss[3],d_loss[4]))
                print("[g loss: %.3f][g dloss: %.3f][g cl loss: %.6f][rec loss: %.3f]"%(g_loss[0],g_loss[1],g_loss[2],g_loss[3]))





                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    '''
                        change parameter path to save image to different dir
                    '''
                    self.sample_images(epoch, batch_i,path='sample_t1ce')
        '''
            remember to save model with different name
        '''
        self.G.save_weights('g_300.h5')

    def sample_images(self, epoch, batch_i,path='sample'):
        os.makedirs(path+'/', exist_ok=True)
        r, c = 2, 2
        imgs_A, label_A = self.s1.get_one_data()
        imgs_B, label_B = self.s2.get_one_data()
        target_label=self.get_onehot_label(2)
        fake_A  = self.G.predict([imgs_A,target_label])
        fake_B  = self.G.predict([imgs_B,target_label])
        gen_imgs = np.concatenate([imgs_A, fake_A, imgs_B, fake_B])
        gen_imgs = np.reshape(gen_imgs,(r*c,config.img_width,config.img_height))
        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5
        for x in range(r*c):
            now_image=gen_imgs[x,:,:]
            now_image=now_image[:,:,np.newaxis]
            now_image=image.array_to_img(now_image,scale=True)
            now_image.save(os.path.join(path, 'generated_' \
                                      + str(epoch) + '_' + str(batch_i)+'_'+str(x) + '_.png'))

        '''
        titles = ['Original', 'Translated', 'Reconstructed']
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt],cmap='gray')
                axs[i, j].set_title(titles[j])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("samples/%d_%d.png" % (epoch, batch_i))
        plt.close()
        '''


if __name__ == '__main__':
    gan = CycleGAN()
    gan.train(epochs=300,batch_size=1,sample_interval=50)
    #g=gan.build_generator()
    #g.summary()
    #d=gan.build_discriminator()
    #d.summary()

