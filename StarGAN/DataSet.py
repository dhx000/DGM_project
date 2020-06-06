import numpy as np
import cv2
import os
import random
import config
class Dataset:
    def __init__(self,img_path,label_path,need_resize=0):
        self.img_path=img_path
        self.label_path=label_path
        self.img_list=self.get_file_list(self.img_path)
        self.label_list=self.get_file_list(self.label_path)
        assert self.img_list.shape[0] == self.label_list.shape[0]
        self.N= len(self.img_list)
        self.need_resize=need_resize
        self.loc=0
    def preprocess_image(self,image):
        image=((image*1.0)/255.0)*2-1
        image=image.astype('float')
        return image
    def get_one_data(self,debug=0):
        x = random.randint(0,self.N-1)
        img_name = self.img_list[x]
        label_name = self.label_list[x]
        now_image = self.get_image(self.img_path + '/' + img_name)
        now_label = self.get_label(self.label_path + '/' + label_name)
        #return_image = self.get_image_without_label(now_image, now_label)
        #return_label = self.get_image_with_label(now_image, now_label)
        return_image=now_image
        return_label=now_label
        if (debug==1):
            self.visualize(return_image)
            self.visualize(return_label,label=1)
        return_image = return_image[np.newaxis,:,:,np.newaxis]
        return_label = return_label[np.newaxis,:,:,np.newaxis]
        return self.preprocess_image(return_image),return_label
    def get_data_batch(self,batch_size=8):
        end = min(self.loc + batch_size, self.N)
        start = self.loc
        img_batch_list=self.img_list[start:end]
        label_batch_list=self.label_list[start:end]
        if (end - start != batch_size):
            need = batch_size - end + start
            img_batch_list = np.concatenate([img_batch_list, self.img_list[0:need]])
            label_batch_list = np.concatenate([label_batch_list, self.label_list[0:need]])
        self.loc = (self.loc + batch_size) % self.N
        train_image = []
        train_label = []
        for x in range(0, batch_size):
            now_image = self.get_image(self.img_path + '/' + img_batch_list[x])
            now_label = self.get_label(self.label_path + '/' + label_batch_list[x])
            return_image = self.get_image_without_label(now_image,now_label)
            return_label = self.get_image_with_label(now_image,now_label)
            train_image.append(return_image)
            train_label.append(return_label)
        # print(train_image.shape)
        train_image=train_image[:,:,:,np.newaxis]
        return train_image, train_label
    def get_file_list(self, path):
        path_list = os.listdir(path)
        return np.array(path_list)
    def get_image_with_label(self,image,label):
        image_out=image*label
        return image_out
    def get_image_without_label(self,image,label):
        image_out=image*(1-label)
        return image_out
    def get_image(self,img_path):
        image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if (self.need_resize==1):
            image=cv2.resize(image,(config.img_width,config.img_height))
        return image
    def get_label(self,label_path):
        label=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        if (self.need_resize==1):
            label=cv2.resize(label,(config.img_width,config.img_height))
        label[label!=0]=1
        return label
    def visualize(self,image,name='image',label=False):
        '''
            image a numpy array can be get from train/val list
            input dim should be [512,512,3] for image or [512,512] for label
            because label value is set to be 1
            to visualize it I set it to be 128
            output a picture
            if you want to visualize a label image set parameter label=True
        '''
        if (label==True):
            image[image==1]=128
            #image=image.reshape([config.img_width,config.img_height])
        img_GaussianBlur = cv2.GaussianBlur(image, (3, 3), 0)
        cv2.imshow(name,img_GaussianBlur)
        cv2.waitKey(0)
if __name__ == '__main__':
    #dataSet=Dataset('./t1ce','./tice_label')
    #dataSet=Dataset('./t2','./t2_label')
    #dataSet = Dataset('./ts/train_image', './ts/train_label')
    dataSet = Dataset('./OpenBayes', './Openbayes_label',need_resize=1)

    '''
    imglist,labellist=dataSet.get_data_batch()
    imglist, labellist = dataSet.get_data_batch()
    dataSet.visualize(imglist[6])
    dataSet.visualize(labellist[6])
    '''
    dataSet.get_one_data(1)