# DGM_project
A improvement for Multi-source Domain Adaptation in Semantic Segmentation based on Medical image processing.
## What I Do
 - Use StarGAN instead of CycleGAN to improve model scalability.Now we can finish this task with only one Generator and Discrminator instead of N G&D in the past.
 - Use Mixmatch (semi-supervised method) to reduce the dependence of the model on the amount of labeled data.
## DataSet
 - Tow Source Domain DataSet is collect from [Brats2018](https://www.med.upenn.edu/sbia/brats2018.html) .I select MRI-T1CE modal as the first source Domain which has 1000 labeled images and MRI-T2 modal as the second one which as 800 labeld images.
 - Target Domain DataSet is collect from OpenBayes,which has 500 images.
## Result
### StarGAN
I transfered two Source Domain images to Target Domain.Here are two samples.

![stargan](https://github.com/dhx000/DGM_project/blob/master/stargan.png)

### Mixmatch 
I compare model performance which contains without DA,DA and DA+mixmatch.The results are shown in the table below.The results show that we get a better performance.

![table](https://github.com/dhx000/DGM_project/blob/master/table.png)

Here are two segmentation samples

![seg](https://github.com/dhx000/DGM_project/blob/master/seg.png)

## RUN

### environment
I implement code based on Tensorflow and Keras,here is my environment setting:
 - Python 3.7
 - Tensorflow 1.14.0
 - Keras 2.3.1
 - numpy 1.17.2
 - opencv 4.1.1.26


### StarGAN
I implement my code based on this [CycleGAN](https://github.com/eriklindernoren/Keras-GAN) code.I use LSGAN and PatchGAN in Discriminator.
#### DataSet setting
initailize the dataset use this function.If your image size is different from image shape in **config.py**, set parameter **need_resize=1**。
```
self.target=Dataset(image_path='./OpenBayes',label_path='./Openbayes_label',need_resize=1)
```
initial your dataset with right path in **__init__** function in **model.py**。

#### Train
start training
```
python model.py
```

### MixMatch
#### DataSet setting
initailize the labeled and unlabeled dataset use this function.If your image size is different from image shape in **config.py**, set parameter **need_resize=1**。
```
    ldataSet=DataSet(image_path='./unet2_image',label_path='./unet2_label')
    udataSet=TestSet(image_path='./OpenBayes',label_path='./Openbayes_label',need_resize=1)
```
#### Train
start training
```
python cycle_mixmatch.py
```
#### evalation
start evalution
```
python evalution.py
```
## Reference
 - [1]. MixMatch: A Holistic Approach to Semi-Supervised Learning.([arxiv](https://arxiv.org/abs/1905.02249v1))
 - [2]. StarGAN: Unified Generative Adversarial Networks
for Multi-Domain Image-to-Image Translation ([full paper](http://openaccess.thecvf.com/content_cvpr_2018/papers/Choi_StarGAN_Unified_Generative_CVPR_2018_paper.pdf))
