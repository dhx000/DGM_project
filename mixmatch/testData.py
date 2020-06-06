import cv2
import numpy as np
import os
import config
class TestSet:
    def __init__(self,image_path='./images',label_path='./label',filepath='',need_resize=0):
        print('load dataset')
        self.image_path=image_path
        self.label_path=label_path
        self.need_resize=need_resize
        self.image_list=self.get_file_list(self.image_path)
        self.label_list=self.get_file_list(self.label_path)
        self.test_loc=0
        self.train_loc=0
        print(self.image_list.shape)
        print('dataset load complete')
    def preprocess_image(self,image):
        image=((image*1.0)/255.0)*2-1
        image=image.astype('float')
        return image
    def get_DataBatch(self,batchsize=4):
        n = min(self.image_list.shape[0], config.train_num)
        end = min(self.train_loc + batchsize, n)
        start = self.train_loc
        batch_list = self.image_list[start:end]
        if (end - start != batchsize):
            need = batchsize - end + start
            batch_list = np.concatenate([batch_list, self.image_list[0:need]])
        self.train_loc = (self.train_loc + batchsize) % n
        train_image = []
        train_label = []
        for x in range(0, batchsize):
            name = str(batch_list[x])
            now_image = self.get_image(self.image_path + '/' + self.image_list[self.train_loc])
            now_label = self.get_label(self.label_path + '/' + self.label_list[self.train_loc])
            train_image.append(now_image)
            train_label.append(now_label)
        train_label = np.array(train_label).reshape([batchsize, config.img_width, config.img_height])
        # train_image=(np.array(train_image)/255.).astype(float)
        train_image = np.array(train_image)
        train_image = self.preprocess_image(train_image)
        train_image = train_image[:, :, :, np.newaxis]
        # print(train_image.shape)
        return train_image, train_label
    def get_test_data_obo(self):
        '''
            get test_data one by one
            use in evluation
            output image,label
            image dim : [1,512,512,3] (to predict,so add a axis in axis=0)
            label dim: [512*512] all value is int belongs to {0,1}
        '''
        n = self.image_list.shape[0]
        self.test_loc+=1
        if (self.test_loc>=n): self.test_loc=0
        assert self.test_loc <= n
        now_name = str(self.image_list[self.test_loc])
        now_image=self.get_image(self.image_path+'/'+self.image_list[self.test_loc])
        #print(self.image_path+'/'+self.image_list[self.test_loc])
        now_label=self.get_label(self.label_path+'/'+self.label_list[self.test_loc])
        now_label=now_label.reshape([1,config.img_width,config.img_height])
        now_image=now_image[np.newaxis,:,:]
        now_image=self.preprocess_image(now_image)
        now_image=now_image[:,:,:,np.newaxis]
        return now_image,now_label,now_name
    def get_test_num(self):
        return self.image_list.shape[0]
    def visualize(self,image,label=False):
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
            image=image.reshape([config.img_width,config.img_height])
        cv2.imshow('image',image)
        cv2.waitKey(0)
    def get_list_fromfile(self,filename):
        f=open(filename,'r')
        returnlist=[]
        line = None
        while True:
            line=f.readline().replace('\n', '')
            if (line is None) or len(line)==0 : break
            returnlist.append(int(line))
        f.close()
        return np.array(returnlist)

    def get_file_list(self,path):
        path_list=os.listdir(path)
        return np.array(path_list)
    def get_image(self,img_path):
        image=cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        if (self.need_resize==1):
            image=cv2.resize(image,(config.img_width,config.img_height))
        return image
    def get_label(self,label_path):
        label=cv2.imread(label_path,cv2.IMREAD_GRAYSCALE)
        #print(label_path)
        #print(label.shape)

        if (self.need_resize == 1):
            label = cv2.resize(label, (config.img_width, config.img_height))
        label[label!=0]=1
        return label
