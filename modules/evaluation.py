import numpy as np
from sklearn.metrics import f1_score, accuracy_score, recall_score, roc_auc_score, precision_score, confusion_matrix, average_precision_score
from scikitplot.helpers import binary_ks_curve
import torch 


Tensor = torch.tensor


def eval_auc_roc(pred, target):
    scores = roc_auc_score(target, pred)
    return scores


def eval_auc_pr(pred, target):
    scores = average_precision_score(target, pred)
    return scores


def eval_ks_statistics(target, pred):
    scores = binary_ks_curve(target, pred)[3]
    return scores


def find_best_f1(probs, labels):
    best_f1, best_thre = -1., -1.
    thres_arr = np.linspace(0.05, 0.95, 19)
    for thres in thres_arr:
        preds = np.zeros_like(labels)
        preds[probs > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre


def eval_pred(pred: Tensor, target: Tensor):
    s_pred = pred.cpu().detach().numpy()
    s_target = target.cpu().detach().numpy()
    
    auc_roc = roc_auc_score(s_target, s_pred)
    auc_pr =  average_precision_score(s_target, s_pred)
    ks_statistics = eval_ks_statistics(s_target, s_pred)
    
    best_f1, best_thre = find_best_f1(s_pred, s_target)
    p_labels = (s_pred > best_thre).astype(int)
    accuracy = np.mean(s_target == p_labels)
    recall = recall_score(s_target, p_labels)
    precision = precision_score(s_target, p_labels)
    
    return auc_roc, auc_pr, ks_statistics, accuracy, \
        recall, precision, best_f1, best_thre

