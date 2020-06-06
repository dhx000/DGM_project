import numpy as np
import cv2
import config
import random
from testData import TestSet
from keras.layers import Input
from keras.models import Model
import keras.backend as K
from keras.losses import binary_crossentropy
from mixmatch_model import MixMatchModel
from DataSet import DataSet
import utils
from Unet import Unet
from keras.optimizers import RMSprop,Adam
def augment(x,flip_code=0):
    h_flip=cv2.flip(x,flip_code)
    h_flip=h_flip[:,:,np.newaxis]
    return h_flip
def augment_labelData(image,label):
    n=image.shape[0]
    return_image=[]
    return_label=[]
    for x in range(0,n):
        now_n=random.randint(0,2)-1
        now_image=augment(image[x],now_n)
        now_label=augment(label[x],now_n)
        return_image.append(now_image)
        return_label.append(now_label)
    return np.array(return_image),np.array(return_label)
def augment_unlabelData(image,K=1):
    n = image.shape[0]
    return_image =  [[None for col in range(K)] for row in range(n)]
    for x in range(0, n):
        now_n = np.random.choice(config.filp_num+1,K,replace=False)
        for y in range(0,K):
            now_image=augment(image[x],now_n[y])
            return_image[x][y]=now_image
    # return N K 240 240
    return np.array(return_image)

def guess_label_with_logit(x_image,model,K=1):
    x_aug=model.predict(x_image)
    x_aug=x_aug.reshape(config.img_width,config.img_height,1)
    return x_aug
def guess_label(x_image,x_label,model,K=2):
    # x_aug [K,240,240,2]
    x_aug=np.concatenate((x_image,x_label),axis=-1)
    # logits [K,30,30,1]
    logits=model.predict(x_aug)
    # [900,K]
    logits=np.reshape(logits,(K,-1)).T
    # [900,1]
    logits_bagging=np.sum(logits,axis=1)
    logits_bagging=logits_bagging / K
    return logits_bagging
def expand_ulabel(image,model,K=1):
    # image [batch,K,240,240,1]
    n = image.shape[0]
    return_label = [[None for col in range(K)] for row in range(n)]
    for x in range(0,n):
        now_label = guess_label_with_logit(image[x], model)
        for y in range(0,K):
            return_label[x][y]=now_label
    return np.array(return_label)
def sharpen(p,T=0.5):
    return np.power(p,1/T)/(np.power(p,1/T)+np.power(1-p,1/T))
def mixup(x1,x2,y1,y2,alpha=0.75):
    beta = np.random.beta(alpha, alpha)
    beta = np.maximum(beta, 1 - beta)
    #beta=0.5
    x = beta * x1 + (1 - beta) * x2
    y = beta * y1 + (1 - beta) * y2
    #label_x1=np.round(np.clip(y1,0,1))
    #label_x2=np.round(np.clip(y2,0,1))
    #y=np.clip(label_x1+label_x2,0,1)
    return x, y
def split_X(XU,XUy,batch_size):
    s=np.arange(batch_size)
    x_image=np.take(XU,s,axis=0)
    x_label=np.take(XUy,s,axis=0)
    return x_image,x_label
def split_U(XU,XUy,batch_size,K):
    s = np.arange(batch_size)+batch_size
    u_label=np.take(XUy,s,axis=0)
    u_image=np.take(XU, s,axis=0)
    return u_image,u_label
def get_right_label(now_label):
    now_label=now_label.reshape((config.img_width,config.img_height))
    return now_label
def mixmatch(model,x,y,u,T=0.5,K=1,alpha=0.75):
    batch_size= x.shape[0]
    x_aug,y_aug=augment_labelData(x,y)
    u_aug = augment_unlabelData(u)
    u_label=expand_ulabel(u_aug,model,K)
    u_label=sharpen(u_label,T)
    U = np.concatenate((u_aug),axis=0)
    U_label=np.concatenate((u_label),axis=0)
    XU = np.concatenate((x_aug,U),axis=0)
    XUy = np.concatenate((y_aug,U_label),axis=0)
    indices=np.arange(XU.shape[0])
    np.random.shuffle(indices)
    W = np.take(XU,indices,axis=0)
    Wy = np.take(XUy,indices,axis=0)
    XU,XUy=mixup(XU,W,XUy,Wy,alpha)
    X_image,X_label=split_X(XU,XUy,batch_size)
    U_image,U_label=split_U(XU,XUy,batch_size,K)
    return X_image,X_label,U_image,U_label
def hard_label_to_soft_label(now_label):
    eps=(1e-5)*np.random.normal(0,1.0,now_label.shape)
    now_label=np.abs(now_label-eps)
    return now_label

if __name__ == '__main__':
    unet=Unet().build_unet()
    cirite_model=MixMatchModel(unet).build_model()
    cirite_model.summary()
    #discriminator=vgan.discriminator
    #ldataSet=DataSet(image_path='./done',label_path='./done_label',need_resize=1)
    #udataSet=TestSet(image_path='./t1ce',label_path='./unet2_label')
    ldataSet=DataSet(image_path='./unet2_image',label_path='./unet2_label')
    udataSet=TestSet(image_path='./OpenBayes',label_path='./Openbayes_label',need_resize=1)
    ans_dice=[]
    best_acc=0
    for epoch in range(0,config.epochs):
        round = int(config.cycle_num/config.batch_size) * 2
        for x in range(0,round):
            x_image, x_label = ldataSet.get_train_dataBatch(batchsize=config.batch_size//2)
            u_image, u_label = udataSet.get_DataBatch(batchsize=config.batch_size//2)
            X_image, X_label, U_image, U_label = mixmatch(unet, x_image, x_label, u_image)

            loss=cirite_model.train_on_batch([X_image,U_image],[X_label,U_label])



            #dataSet.visualize(X_image[0])
            #dataSet.visualize(np.uint8(X_label[0]*255),label=True)



            '''
            y_image, y_label = dataSet.get_label_dataBatch(batch_size=config.batch_size)
            y_label=to_categorical(y_label)
            z_image, z_label = dataSet.get_label_dataBatch(batch_size=config.batch_size)
            z_label=to_categorical(z_label)
            mix_image, mix_label = mixup(z_image, y_image, z_label, y_label)

            #dataSet.visualize(mix_image[0]*0.5+0.5)
            #dataSet.visualize(mix_image[0])
            #dataSet.visualize(np.uint8(mix_label[0,:,:,1] * 255), label=True)

            valid = np.ones((config.batch_size,config.patch_size,config.patch_size,1))
            fake = np.zeros((config.batch_size,config.patch_size,config.patch_size,1))
            #train true


            mix_label = hard_label_to_soft_label(mix_label)
            mix_input=np.concatenate((mix_image,mix_label),axis=-1)
            d_valid_loss=discriminator.train_on_batch(mix_input,valid)

            #train_fake

            input_image=np.concatenate((X_image,U_image),axis=0)
            input_label=np.concatenate((X_label,U_label),axis=0)
            input_label=hard_label_to_soft_label(input_label)
            now_input=np.concatenate((input_image,input_label),axis=-1)
            d_fake_loss=discriminator.train_on_batch(now_input,fake)

            d_loss=0.5*np.mean(d_valid_loss+d_fake_loss)

            # train vgan
            g_loss = cirite_mode.train_on_batch([X_image, U_image], [valid[0:config.batch_size//2],valid[config.batch_size//2:config.batch_size],X_label, U_label])
            '''
            x_gt=unet.predict(x_image)
            u_gt=unet.predict(u_image)

            x_dice=utils.get_dice(x_label,x_gt)
            u_dice=utils.get_dice(u_label,u_gt)


            print(
                "[Epoch %d/%d] [Batch %d/%d]  [seg loss: %05f, seg_label: %05f seg_unlabel : %05f] [xdice : %05f udice : %05f] " \
                % (epoch, config.epochs,
                   x, round,
                   loss[0],
                   loss[1],
                   loss[2],
                   x_dice,
                   u_dice
                   ))
        test_dice = []
        for x in range(0, config.cycle_num):
            now_image, now_label, now_name = udataSet.get_test_data_obo()
            now_predict = unet.predict(now_image)
            dice = utils.get_dice(now_label, now_predict)
            test_dice.append(dice)

        test_dice = np.mean(test_dice)
        if (test_dice>best_acc):
            best_acc=test_dice
            unet.save('cycle_mixmatch_best.h5')
        ans_dice.append(test_dice)
        print("[Epoch %d/%d] [test_dice : %05f]" % (epoch, config.epochs, test_dice))
    ans_dice=np.array(ans_dice)
    np.save('cycle_mixmatch_best.npy',ans_dice)







