from torch import nn
from .loss import LpLoss, H1Loss


CRITERION_DICT = {
    
}



class Criterion(nn.Module):
    def __init__(self, criterion_list):
        super(Criterion, self).__init__()
        pass

    def forward(self, preds, labels):
        loss = self.criterion(preds, labels)
        return loss