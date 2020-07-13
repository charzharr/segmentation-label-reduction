
import sys
import torch, torch.nn as nn

class PixelWiseBCE(nn.Module):

    def __init__(self):
        super(PixelWiseBCE, self).__init__()
        self.softmax = nn.Softmax(dim=1)
        self.BCELoss = nn.BCELoss()

    def forward(self, pred, targ):
        pred_probs = self.softmax(pred)
        return self.BCELoss(pred_probs, targ)
        