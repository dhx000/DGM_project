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
I transfered two Source Domain images to Target Domain.Here are tow samples.
!()[stargan.png]
 
