import numpy as np
import math
import config
from tensorflow.python.keras import backend as K


def tp_sum(y_true,y_pred):
    y_label=K.round(K.clip(y_pred, 0, 1))

    return K.mean(K.sum(y_true*y_label,axis=1))
def label_sum(y_true,y_pred):
    y_label=K.round(K.clip(y_pred, 0, 1))

    return K.mean(K.sum(y_label,axis=1))
def true_sum(y_true,y_pred):
    return K.mean(K.sum(y_true,axis=1))
def get_recall(y_true,y_pred):
    y_tp = np.round(np.clip(y_true * y_pred, 0, 1))
    y_tp_sum = np.sum(y_tp)
    y_label_sum=np.sum(y_true)
    if (y_label_sum==0):
        return 0
    return y_tp_sum/y_label_sum
def get_prec(y_true,y_pred):
    y_tp = np.round(np.clip(y_true * y_pred, 0, 1))
    y_tp_sum = np.sum(y_tp)
    y_label_sum = np.sum(y_pred)
    if (y_label_sum == 0):
        return 0
    return y_tp_sum / y_label_sum
def get_IOU(y_true,y_pred):
    smooth = 1.0
    y_true = np.reshape(y_true, (-1, config.img_height * config.img_width))
    y_label = np.reshape(y_pred, (-1, config.img_width * config.img_height))
    y_label = np.round(np.clip(y_label, 0, 1))
    y_tp = np.round(np.clip(y_true * y_label, 0, 1))

    y_tp_sum = np.sum(y_tp, axis=1)
    y_true_sum = np.sum(y_true, axis=1)
    y_label_sum = np.sum(y_label, axis=1)
    IOU_array = (y_tp_sum+smooth) / (y_label_sum + y_true_sum  - y_tp_sum +smooth)
    IOU = np.mean(IOU_array)
    return IOU
def get_dice(y_true,y_pred):
    smooth = 1.0
    y_true = np.reshape(y_true, (-1, config.img_height * config.img_width))
    y_label = np.reshape(y_pred, (-1, config.img_width * config.img_height))
    y_label = np.round(np.clip(y_label, 0, 1))


    y_tp = np.round(np.clip(y_true * y_label, 0, 1))

    y_tp_sum = np.sum(y_tp, axis=1)
    y_true_sum = np.sum(y_true, axis=1)
    y_label_sum = np.sum(y_label, axis=1)
    dice_array = (y_tp_sum * 2 + smooth) / (y_label_sum + y_true_sum + smooth)
    dice = np.mean(dice_array)
    return dice
def f1(y_true, y_pred):
    smooth=1.0

    y_true = K.reshape(y_true,(-1,config.img_height*config.img_width))
    y_label = K.reshape(y_pred,(-1,config.img_width*config.img_height))

    y_tp = K.round(K.clip(y_true * y_label, 0, 1))

    y_tp_sum=K.sum(y_tp,axis=1)
    y_true_sum=K.sum(y_true,axis=1)
    y_label_sum=K.sum(y_label,axis=1)
    y_tp_sum=K.cast(y_tp_sum,np.float32)
    y_label_sum=K.cast(y_label_sum,np.float32)
    y_true_sum=K.cast(y_true_sum,np.float32)
    dice_array=(y_tp_sum*2+smooth)/(y_label_sum+y_true_sum+smooth)
    dice = K.mean(dice_array)
    return dice

def dice_coef(y_true, y_pred):
    smooth=1.0
    intersection = K.sum(y_true * y_pred, axis=(1, 2))
    union = K.sum(y_true, axis=(1, 2)) + K.sum(y_pred, axis=(1, 2))
    sample_dices = (2. * intersection + smooth) / (union + smooth)
    dices = K.mean(sample_dices, axis=0)
    return K.mean(dices)

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true,y_pred)


