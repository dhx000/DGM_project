import cv2
import numpy as np
import os
dir_list=os.listdir('./sl')
print(dir_list)
for image in dir_list:
    name=image.split('.')[0]
    label_name=name+'.png'
    path='./unet2_label/'+label_name
    now_img=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    cv2.imwrite('./sl_label/'+label_name,now_img)
