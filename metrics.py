import torch

import torch.nn.functional as F

from sklearn.metrics import f1_score as f1_sklearn
from sklearn.metrics import jaccard_score as jaccard_sklearn


# Set the overall THRESHOLD for tp, fp, tn, fn, iou
THRESHOLD = 0.5


#################################################################################################################################
###########################################################  Metrics  ###########################################################
#################################################################################################################################
def f1_score(target: torch.tensor, prediction: torch.tensor, nr_classes: int):
    multiclass = True if nr_classes > 2 else False

    if not multiclass:
        # convert to boolean and back to flaot and int
        prediction = (prediction > THRESHOLD).float()
        target = (target == torch.max(target)).int()

        # flat input tensors
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        f1 = f1_sklearn(target.cpu(), prediction.cpu())

    elif multiclass:
        if prediction.ndim == 4:
            if (target.shape[1] == 1 and target.ndim == 4) or (target.ndim == 3):
                target_argmax = target.cpu()
                target_onehot = (
                    F.one_hot(target_argmax, num_classes=nr_classes).long().cpu()
                )
            else:
                target_onehot = target.long().cpu()
                target_argmax = torch.argmax(target, dim=1).cpu()

            if (prediction.shape[1] == 1 and prediction.ndim == 4) or (
                prediction.ndim == 3
            ):
                prediction_argmax = prediction.cpu()
            else:
                prediction_argmax = torch.argmax(prediction, dim=1).long().cpu()

            target = target_argmax.contiguous().view(-1)
            prediction = prediction_argmax.contiguous().view(-1)

            f1 = f1_sklearn(target, prediction, average="micro")  # "macro"

        if prediction.ndim == 5:
            target_onehot = target.cpu()
            target_argmax = torch.argmax(target_onehot, dim=1).long().cpu()

            prediction_argmax = torch.argmax(prediction, dim=1).long().cpu()

            target = target_argmax.contiguous().view(-1)
            prediction = prediction_argmax.contiguous().view(-1)

            f1 = f1_sklearn(target, prediction, average="macro")

    return f1


def iou_score(target: torch.tensor, prediction: torch.tensor, nr_classes: int):
    multiclass = True if nr_classes > 2 else False

    if not multiclass:
        # convert to boolean and back to flaot and int
        prediction = (prediction > THRESHOLD).float()
        target = (target == torch.max(target)).int()

        # flat input tensors
        prediction = prediction.contiguous().view(-1)
        target = target.contiguous().view(-1)

        # compute intersection and union
        intersection = (prediction * target).sum()
        union = ((prediction + target).sum()) - intersection

        # compute intersection over union
        iou = (intersection + 1) / (union + 1)

    elif multiclass:
        # 2d case
        if prediction.ndim == 4:
            target_argmax = torch.argmax(target, dim=1).long().cpu()
            target_onehot = target.long().cpu()

            prediction_argmax = torch.argmax(prediction, dim=1).long().cpu()

            target = target_argmax.contiguous().view(-1)
            prediction = prediction_argmax.contiguous().view(-1)

            iou = jaccard_sklearn(target, prediction, average="micro")

        # 3d case
        if prediction.ndim == 5:
            target_onehot = target.cpu()
            target_argmax = torch.argmax(target_onehot, dim=1).long().cpu()

            nr_classes = torch.numel(torch.unique(target_argmax))

            prediction_argmax = torch.argmax(prediction, dim=1).long().cpu()

            target = target_argmax.contiguous().view(-1)
            prediction = prediction_argmax.contiguous().view(-1)

            iou = jaccard_sklearn(target, prediction, average="macro")

    return iou