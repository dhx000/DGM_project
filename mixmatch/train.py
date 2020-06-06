from DataSet import DataSet
from testData import TestSet
import utils
from Unet import Unet
import config
import numpy as np
import math
def TrainGenerator(dataset,batch_size):
    while True:
        batch_images, batch_labels = dataset.get_train_dataBatch(batchsize=batch_size)
        batch_labels=batch_labels[:,:,:,np.newaxis]
        yield batch_images,batch_labels
def ValGenerator(dataset,batch_size):
    while True:
        batch_images, batch_labels,name = dataset.get_test_data_obo()
        batch_labels=batch_labels[:,:,:,np.newaxis]
        yield batch_images,batch_labels
if __name__ == '__main__':
    train_dataSet=DataSet('./sl','./sl_label')
    udataSet = TestSet(image_path='./OpenBayes', label_path='./Openbayes_label', need_resize=1)
    unetClass=Unet()
    unet=unetClass.get_unet()
    round=int(config.train_num/config.batch_size)

    trainGenerator = TrainGenerator(train_dataSet, config.batch_size)
    valGenerator = ValGenerator(udataSet, config.batch_size)

    # Fcn32.fit_generator(trainGenerator,steps_per_epoch=math.ceil(config.train_num / config.batch_size),epochs=config.epochs,validation_data=valGenerator,validation_steps=8)

    # unet=Unet().build_model()
    unet.fit_generator(trainGenerator, steps_per_epoch=math.ceil(config.train_num / config.batch_size),
                     epochs=30, validation_data=valGenerator, validation_steps=8)
    unet.save_weights('unet_transformed.h5')

