import numpy as np

def model_eval(id2str, batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels):
    # define iou thresholds
    iou_thresholds = np.arange(0.5, 0.96, 0.05).tolist()

    # initialize precision recall dictionary to store results
    precision_recall_dict = {}

    # loop through all iou thresholds to get precision and recall for all classes at each iou_threshold
    for iou_threshold in iou_thresholds:

        # get tp, fp, fn dictionary for all classes
        perf_dict = get_perf_dict(id2str, batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels, iou_threshold)
        
        # create dictionary to store precision and recall for all classes
        metric_dict = {key: {'Precision': 0,
                            'Recall': 0}
                    for key in id2str.keys()}
        
        # derive precision and recall using tp, fp, fn from perf_dict for each class
        for label in list(metric_dict.keys()):
            precision, recall = compute_precision_recall(perf_dict[label]['TP'], perf_dict[label]['FP'], perf_dict[label]['FN'])
            metric_dict[label]['Precision'] = precision
            metric_dict[label]['Recall'] = recall

        # store precision recall dict for iou_threshold value in results dict
        precision_recall_dict[iou_threshold] = metric_dict

    # initialize average precision dictionary
    ap_dict = {key: 0 for key in id2str.keys()}

    # calculate average precision for each label
    for label in list(id2str.keys()):

        # initialize precisions and recalls list for auc calculation
        precisions = []
        recalls = []

        # add precision/recall datapoint for each iou_threshold
        for iou_threshold in iou_thresholds:
            precisions.append(precision_recall_dict[iou_threshold][label]['Precision'])
            recalls.append(precision_recall_dict[iou_threshold][label]['Recall'])

        # need to get unique values of precisions and recalls so that we can sort by recall
        sorted_precisions = []
        sorted_recalls = []
        unique_precisions = []
        unique_recalls = []

        # find all unique values of recall
        for a, recall in enumerate(recalls):
            
            # add recall to unique_recalls if not in list
            if recall not in unique_recalls:
                unique_recalls.append(recall)

                # add corresponding precision for this recall
                unique_precisions.append(precisions[a])

            # if recall is already in the list and the precision is higher than stored precision, update it
            elif recall in unique_recalls and precisions[a] > unique_precisions[unique_recalls.index(recall)]:
                unique_precisions[unique_recalls.index(recall)] = precisions[a]
        
        # if there is more than one datapoint
        if len(unique_recalls) > 1:
            sorted_indices = np.argsort(unique_recalls)
            sorted_recalls = np.array(unique_recalls)[sorted_indices]
            sorted_precisions = np.array(unique_precisions)[sorted_indices]
        
        # only one unique datapoint
        else:
            sorted_recalls = unique_recalls
            sorted_precisions = unique_precisions
        
        # calculate auc using trapezoidal rule
        ap = np.trapz(sorted_precisions, sorted_recalls)
        
        # store average precision in dictionary
        ap_dict[label] = ap

    # calculate mAP
    ap_dict.pop(-1)
    sum = 0
    for key in list(ap_dict.keys()):
        sum += ap_dict[key]
    mAP = sum / len(list(ap_dict.keys()))
    return mAP, ap_dict

# def get_perf_dict(str2id, batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels, iou_threshold=0.5):
#     perf_dict = {key: {'TP': 0,
#                    'FP': 0,
#                    'FN': 0}
#                    for key in str2id.keys()}
    
#     # Iterate through each evaluated image
#     for j, truth_labels in enumerate(batch_truth_labels):

#         truth_boxes = batch_truth_boxes[j]
#         pred_boxes = batch_pred_boxes[j]
#         pred_labels = batch_pred_labels[j]

#         # calculate false negatives
#         for truth in truth_labels:
#             if truth not in pred_labels and truth != 'pad':
#                 perf_dict[truth]['FN'] += 1

#         for i, truth_box in enumerate(truth_boxes):

#             # initialize tracker for best match for ground truth box
#             best_match = -1
#             best_iou = 0

#             for k, pred_box in enumerate(pred_boxes):
                
#                 # if a predicted box does not match any ground truth box, consider it a false positive
#                 if pred_labels[k] not in truth_labels:
#                     perf_dict[pred_labels[k]]['FP'] += 1
                
#                 # there is a match between the predicted box and a ground truth box
#                 else:
#                     # only consider cases when the labels for the boxes match
#                     if pred_labels[k] == truth_labels[i]:

#                         # compute iou
#                         iou = compute_iou(truth_box, pred_box)

#                         # if iou is less than threshold, consider it a false positive
#                         if iou < iou_threshold:
#                             perf_dict[pred_labels[k]]['FP'] += 1

#                         # iou is >= iou_threshold but it's not the best match, also consider it a false positive
#                         elif iou >= iou_threshold and iou < best_iou:
#                             perf_dict[pred_labels[k]]['FP'] += 1
                        
#                         # iou is >= iou_threshold and it's the best match
#                         else:
#                             if best_match != -1:
#                                 perf_dict[pred_labels[k]]['FP'] += 1
#                             else:
#                                 best_iou = iou
#                                 best_match = k
#                                 perf_dict[pred_labels[k]]['TP'] += 1
#     return perf_dict

def get_perf_dict(id2str, batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels, iou_threshold=0.5):
    perf_dict = {key: {'TP': 0, 'FP': 0, 'FN': 0} for key in id2str.keys()}
    
    # Iterate through each batch of images
    for truth_boxes, truth_labels, pred_boxes, pred_labels in zip(batch_truth_boxes, batch_truth_labels, batch_pred_boxes, batch_pred_labels):
        matched = set()
        
        # Iterate over all predictions
        for k, pred_box in enumerate(pred_boxes):
            found_match = False
            for i, truth_box in enumerate(truth_boxes):
                if pred_labels[k] == truth_labels[i]:  # Ensure labels match
                    iou = compute_iou(pred_box, truth_box)
                    if iou >= iou_threshold and (i not in matched):  # Check IoU threshold and unmatched
                        perf_dict[pred_labels[k]]['TP'] += 1
                        matched.add(i)
                        found_match = True
                        break
            if not found_match:
                perf_dict[pred_labels[k]]['FP'] += 1

        # Calculate FNs
        for i, truth_label in enumerate(truth_labels):
            if i not in matched:
                perf_dict[truth_label]['FN'] += 1

    return perf_dict

def compute_iou(boxA, boxB):

    # Determine the coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    
    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # Compute iou
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    return iou

def compute_precision_recall(tp, fp, fn):

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0

    # Recall
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall