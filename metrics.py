import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F



def resize_labels(labels, size):
    resized_labels = F.interpolate(labels, size=(size, size), mode='bilinear', align_corners=False)
    threshold = 0.5
    resized_labels = (resized_labels > threshold).float()
    return resized_labels


class PixelWiseContrastiveLoss(nn.Module):
    def __init__(self, max, min):
        super(PixelWiseContrastiveLoss, self).__init__()
        self.maxvalue = max
        self.minvalue = min

    def forward(self, output1, output2, label):
        label = resize_labels(label, output1.size(2))
        batch_size = output1.size(0)
        distance = torch.norm((output1 - output2),2,1)
        change_loss = label.mul(torch.where(distance < self.minvalue, 0, distance * 1e1))
        change_loss = torch.mean(change_loss.view(batch_size,-1),dim=1)
        unchange_loss = (1-label).mul(torch.where(distance < self.maxvalue, distance*1e1, 0))
        unchange_loss = torch.mean(unchange_loss.view(batch_size, -1), dim=1)
        loss = (change_loss + unchange_loss).sum() / float(batch_size)
        return loss


def MDD_loss(features, labels):
    features = nn.Softmax(dim=1)(features)
    batch_size = features.size(0)
    if float(batch_size) % 2 != 0:
        raise Exception('Incorrect batch size provided')
    feature_left = features[:int(0.5 * batch_size)]
    feature_right = features[int(0.5 * batch_size):]
    ContrastiveLoss = PixelWiseContrastiveLoss(0.6,0.1)
    label = labels[:int(0.5 * batch_size)]
    loss_c = ContrastiveLoss(feature_left, feature_right, label)
    return loss_c


