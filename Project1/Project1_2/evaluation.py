import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import accuracy_score

def evaluate(pred_rects,pred_scores,pred_labels,gt_rects,gt_labels): 

    preds = [dict(
        boxes=torch.tensor(pred_rects),
        scores=torch.tensor(pred_scores),
        labels=torch.tensor(pred_labels),
        )]

    target = [dict(
        boxes=torch.tensor(gt_rects),
        labels=torch.tensor(gt_labels),
        )]
    
    metric = MeanAveragePrecision()
    metric.update(preds, target)
    val = metric.compute()
    print(val)

    acc = accuracy_score(gt_labels,pred_labels)
    print('Accuracy score:',acc)