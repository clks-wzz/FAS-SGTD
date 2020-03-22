import numpy as np
import sklearn
from sklearn.metrics import roc_curve, auc

differ_thresh=0.01

def hex2dec(string_num):
    return str(int(string_num.upper(), 16))

def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th

def performances(dev_scores, dev_labels, test_scores, test_labels):
    dev_labels[dev_labels<0] = 0
    fpr,tpr,threshold = roc_curve(dev_labels, dev_scores, pos_label=1)
    err, best_th = get_err_threhold(fpr, tpr, threshold)
    #err = 1 - sklearn.metrics.average_precision_score(dev_labels, dev_scores)
    attacks=np.unique(test_labels[test_labels<0])
    #results = [err * 100, 0, 0, 0, 0, 0]
    APCER = np.zeros([2])
    for i in range(attacks.size):
        real_scores=test_scores[test_labels>0]
        attack_scores=test_scores[test_labels==attacks[i]]
        APCER[i] = np.mean(np.array(attack_scores>best_th, np.float32))*100
        BPCER = np.mean(np.array(real_scores<best_th, np.float32))*100

    ''' my test block: only mean err for test set '''
    fpr_test,tpr_test,threshold_test = roc_curve(test_labels, test_scores, pos_label=1)
    err_test, best_th_test = get_err_threhold(fpr_test, tpr_test, threshold_test)
    ''' my test block: only mean err for test set '''
    
    results = [err * 100, err_test * 100, APCER[0], APCER[1], np.max(APCER), BPCER, (np.max(APCER) + BPCER)/2.0]
    return results
