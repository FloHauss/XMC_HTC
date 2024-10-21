#!/usr/bin/env python
# coding:utf-8

import numpy as np


def _precision_recall_f1(right, predict, total):
    """
    :param right: int, the count of right prediction
    :param predict: int, the count of prediction
    :param total: int, the count of labels
    :return: p(precision, Float), r(recall, Float), f(f1_score, Float)
    """
    p, r, f = 0.0, 0.0, 0.0
    if predict > 0:
        p = float(right) / predict
    if total > 0:
        r = float(right) / total
    if p + r > 0:
        f = p * r * 2 / (p + r)
    return p, r, f

def evaluate(epoch_predicts, epoch_labels, id2label, logger, threshold=0.5, top_k=50, as_sample=False):
    """
    :param epoch_labels: List[List[int]], ground truth, label id
    :param epoch_predicts: List[List[Float]], predicted probability list
    :param vocab: data_modules.Vocab object
    :param threshold: Float, filter probability for tagging
    :param top_k: int, truncate the prediction
    :return:  confusion_matrix -> List[List[int]],
    Dict{'precision' -> Float, 'recall' -> Float, 'micro_f1' -> Float, 'macro_f1' -> Float}
    """
    assert len(epoch_predicts) == len(epoch_labels), 'mismatch between prediction and ground truth for evaluation'
    # label2id = vocab.v2i['label']
    # id2label = vocab.i2v['label']
    # epoch_gold_label = list()
    # # get id label name of ground truth
    # for sample_labels in epoch_labels:
    #     sample_gold = []
    #     for label in sample_labels:
    #         assert label in id2label.keys(), print(label)
    #         sample_gold.append(id2label[label])
    #     epoch_gold_label.append(sample_gold)

    epoch_gold = epoch_labels

    # initialize confusion matrix
    # confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    ##right_count_list = [0 for _ in range(len(id2label))]
    ##gold_count_list = [0 for _ in range(len(id2label))]
    ##predicted_count_list = [0 for _ in range(len(id2label))]
    right_count_list = {}
    gold_count_list = {}
    predicted_count_list = {}
    
    for sample_predict, sample_gold in zip(epoch_predicts, epoch_gold):
        if as_sample:
            sample_predict_id_list = sample_predict
        else:
            np_sample_predict = np.array(sample_predict, dtype=np.float32)
            sample_predict_descent_idx = np.argsort(-np_sample_predict)
            sample_predict_id_list = []
            if top_k is None:
                top_k = len(sample_predict)
            for j in range(top_k):
                if np_sample_predict[sample_predict_descent_idx[j]] > threshold:
                    sample_predict_id_list.append(sample_predict_descent_idx[j])

        #for i in range(len(confusion_count_list)):
        #    for predict_id in sample_predict_id_list:
        #        confusion_count_list[i][predict_id] += 1

        # count for the gold and right items
        for gold in sample_gold:
            #gold_count_list[gold] += 1
            gold_count_list[gold] = gold_count_list.get(gold, 0) + 1
            for label in sample_predict_id_list:
                if gold == label:
                    #right_count_list[gold] += 1
                    right_count_list[gold] = right_count_list.get(gold, 0) + 1

        # count for the predicted items
        for label in sample_predict_id_list:
            #predicted_count_list[label] += 1
            predicted_count_list[label] = predicted_count_list.get(label, 0) + 1
            

    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        print(f'i: {i}, label: {label}')
        #label = label + '_' + str(i)
        #precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
        #                                                                                     predicted_count_list[i],
        #                                                                                     gold_count_list[i])
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list.get(i, 0),
                                                                                             predicted_count_list.get(i, 0),
                                                                                             gold_count_list.get(i, 0))
        right_total += right_count_list.get(i, 0)
        gold_total += gold_count_list.get(i, 0)
        predict_total += predicted_count_list.get(i, 0)
        

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total if gold_total > 0 else 0.0
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            'full': [precision_dict, recall_dict, fscore_dict, right_count_list, predicted_count_list, gold_count_list]}


def evaluate_seq2seq(batch_predicts, batch_labels, id2label):
    """_summary_

    Args:
        batch_predicts (_type_): one batch of predicted graph e.g [[0,0,1...],[0,1,...]],index is the corresponding label_id
        batch_labels (_type_): _description_ same as top,but the ground true label
        id2label (_type_): _description_

    Returns:
        _type_: _description_ return de micro,macro,precision and recall
    """
    assert len(batch_predicts) == len(batch_labels), 'mismatch between prediction and ground truth for evaluation'
    np_pred, np_labels = np.array(batch_predicts),np.array(batch_labels)
    np_right = np.bitwise_and(np_pred,np_labels)
    #[1]是True的索引,[0]是batch的索引，使用[1]就足够了
    pred_label_id = np.nonzero(np_pred)[1].tolist()
    labels_label_id = np.nonzero(np_labels)[1].tolist()
    right_label_id = np.nonzero(np_right)[1].tolist()
    
    # initialize confusion matrix
    confusion_count_list = [[0 for _ in range(len(id2label))] for _ in range(len(id2label))]
    right_count_list = [0 for _ in range(len(id2label))]
    gold_count_list = [0 for _ in range(len(id2label))]
    predicted_count_list = [0 for _ in range(len(id2label))]
    
    for x in pred_label_id :predicted_count_list[x]+=1
    for x in labels_label_id :gold_count_list[x]+=1
    for x in right_label_id:right_count_list[x]+=1
    


    precision_dict = dict()
    recall_dict = dict()
    fscore_dict = dict()
    right_total, predict_total, gold_total = 0, 0, 0

    for i, label in id2label.items():
        label = label + '_' + str(i)
        precision_dict[label], recall_dict[label], fscore_dict[label] = _precision_recall_f1(right_count_list[i],
                                                                                             predicted_count_list[i],
                                                                                             gold_count_list[i])
        right_total += right_count_list[i]
        gold_total += gold_count_list[i]
        predict_total += predicted_count_list[i]

    # Macro-F1
    precision_macro = sum([v for _, v in precision_dict.items()]) / len(list(precision_dict.keys()))
    recall_macro = sum([v for _, v in recall_dict.items()]) / len(list(precision_dict.keys()))
    macro_f1 = sum([v for _, v in fscore_dict.items()]) / len(list(fscore_dict.keys()))
    # Micro-F1
    precision_micro = float(right_total) / predict_total if predict_total > 0 else 0.0
    recall_micro = float(right_total) / gold_total
    micro_f1 = 2 * precision_micro * recall_micro / (precision_micro + recall_micro) if (precision_micro + recall_micro) > 0 else 0.0

    return {'precision': precision_micro,
            'recall': recall_micro,
            'micro_f1': micro_f1,
            'macro_f1': macro_f1,
            }

def evaluate_RP(epoch_predicts, epoch_labels):
    r_precision_scores = []

    # calculate R-precision for each sample:
    for predicts, labels in zip(epoch_predicts, epoch_labels):
        R = len(labels)
        set_predictions = set(predicts[:R])
        set_labels = set(labels)

        relevant_retrieved = len(set_predictions & set_labels)
        r_precision = relevant_retrieved / R
        r_precision_scores.append(r_precision)

    # calculate the average R-precision:
    average_r_precision = sum(r_precision_scores) / len(r_precision_scores) if r_precision_scores else 0

    return average_r_precision

def evaluate_PK(epoch_predicts, epoch_labels, K, as_sample=True):
    assert len(epoch_predicts) == len(epoch_labels), 'Mismatch between prediction and ground truth for evaluation'

    #storage:
    precision_at_k = {f'P@{k}': 0 for k in K}
    total_correct_counts = {f'P@{k}': 0 for k in K}

    num_samples = len(epoch_labels)  # Number of samples to average over

    # Same code as the original HBGL, but we use only sample:
    for sample_predict, sample_gold in zip(epoch_predicts, epoch_labels):
        if as_sample:
            sample_predict_id_list = sample_predict  # If predictions are pre-ranked
        else:
            np_sample_predict = np.array(sample_predict, dtype=np.float32)
            sample_predict_descent_idx = np.argsort(-np_sample_predict)  # Sort predictions
            sample_predict_id_list = sample_predict_descent_idx[:K]  # Top K predictions

        true_labels_set = set(sample_gold)  # Set of true labels

        # Calculate precision at each K from 1 to K, K is set in the test file test.py:
        for k in K:
            top_k_predictions = sample_predict_id_list[:k]
            correct_count = len([pred for pred in top_k_predictions if pred in true_labels_set])
            total_correct_counts[f'P@{k}'] += correct_count / k  # Count correct predictions for each K

    # Calculate average P@K over all samples
    for k in K:
        precision_at_k[f'P@{k}'] = total_correct_counts[f'P@{k}'] / num_samples   # normalize by the number of samples

    return precision_at_k