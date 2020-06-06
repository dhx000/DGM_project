import utils
import numpy as np
from DataSet import DataSet
from testData import TestSet
from Unet import Unet
import config
from sklearn.metrics import roc_auc_score,recall_score,precision_score,accuracy_score
if __name__== '__main__':
    dataSet=DataSet(image_path='./unet2_image',label_path='./unet2_label',need_resize=1)
    testSet2=TestSet(image_path='./OpenBayes',label_path='./Openbayes_label',need_resize=1)
    testSet3 = TestSet(image_path='./unet2_image', label_path='./unet2_label')
    #testSet=TestSet(image_path='./combine2',label_path='./Res_label')
    unet=Unet().build_unet()
    unet.load_weights('unet_transformed.h5')

    #baseline_unet=Unet().build_unet()
    #baseline_unet.load_weights('unet.h5')

    all_unet=Unet().build_unet()
    all_unet.load_weights('cycle_mixmatch_best.h5')

    #mixnet=Unet().build_unet()
    #mixnet.load_weights('cycle_mixmatch_best.h5')
    dice_list=[]
    name_list=[]
    auc_list=[]
    iou_list=[]
    prec_list=[]
    roc_list=[]
    recall_list=[]
    acc_list=[]
    print(dataSet.test_list.shape[0])
    for x in range(0,config.cycle_num):

        source_image,source_gt,source_name=testSet2.get_test_data_obo()
        source_output=np.round(unet.predict(source_image))
        if (np.sum(source_gt)==0): continue
        dice=utils.get_dice(source_gt,source_output)
        IOU=utils.get_IOU(source_gt,source_output)
        source_gt=source_gt.reshape((-1))
        source_output=source_output.reshape((-1))
        sk_recall = utils.get_recall(source_gt, source_output)
        sk_prec = utils.get_prec(source_gt, source_output)
        #sk_dice= utils.get_dice(source_gt,sou)
        sk_auc=roc_auc_score(source_gt,source_output)
        sk_accuary=accuracy_score(source_gt,source_output)


        #print(dice)

        #target_output=np.round(mixnet.predict(source_image))
        #target_dice=utils.get_dice(source_gt,target_output)
        #improve=target_dice-dice
        #print("source: %05f target: %05f improve: %05f name:%s"%(dice,target_dice,improve,source_name))
        #if (target_dice<0.1):
         #   name_list.append(source_name)
        dice_list.append(dice)
        auc_list.append(sk_auc)
        iou_list.append(IOU)
        prec_list.append(sk_prec)
        recall_list.append(sk_recall)
        acc_list.append(sk_accuary)
        print(x)
        print("[acc: %05f][prec: %05f][recall: %05f][IOU: %05f][dice : %05f] [auc : %05f]"%(sk_accuary,sk_prec,sk_recall,IOU,dice,sk_auc))
        #print("[auc  : %05f]"%(sk_dice))
    dice_list=np.array(dice_list)
    np.save('dice.npy',dice_list)

    dice=np.mean(dice_list)
    auc=np.mean(np.array(auc_list))
    prec=np.mean(np.array(prec_list))
    recall=np.mean(np.array(recall_list))
    acc=np.mean(np.array(acc_list))
    iou=np.mean(np.array(iou_list))
    print('final ans')
    print("[acc: %05f][prec: %05f][recall: %05f][IOU: %05f][dice : %05f] [auc : %05f]" % (acc, prec, recall, iou, dice, auc))
    print("%05f\t%05f\t%05f\t%05f\t%05f\t%05f]" % (acc, prec, recall, dice, iou, auc))
    print(name_list)
