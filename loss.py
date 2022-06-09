
import torch
import torch.nn as nn
import torch.nn.functional as F



def loss_cline(input,target,alpha= 0.4,gamma = 2):
    input = torch.squeeze(input)
    BCE_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
    pt = torch.exp(-BCE_loss)
    focal_loss = alpha *  (1-pt) ** gamma  * BCE_loss
    return torch.mean(focal_loss)

def loss_cldice(output_top, target_top, output_bot, target_bot):
    smooth = 0.00001
    tprec = (torch.sum(torch.multiply(target_top, output_bot)[:,1:,...])+smooth)/(torch.sum(output_bot[:,1:,...])+smooth)    
    tsens = (torch.sum(torch.multiply(output_top, target_bot)[:,1:,...])+smooth)/(torch.sum(target_bot[:,1:,...])+smooth)    
    cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
    return cl_dice

def loss_seg(input, target):
    smooth = 0.00001
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))

class CustomLoss(nn.Module):
    def __init__(self, lambda1 = 0.2, lambda2 = 0.5, lambda3 =0.1, threshold = 0.7):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.threshold = threshold
    
    def forward(self, output_top, target_top, output_bot, target_bot):
      
        loss1 = loss_seg(( output_top >self.threshold).type(torch.float64),( target_top >self.threshold ).type(torch.float64))
        loss2 = loss_cline(output_bot, target_bot, 0.4, 2) + loss_cline(1-output_bot, 1-target_bot, 0.6, 2)
        loss3 = loss_cldice(( output_top > self.threshold).type(torch.float64), ( target_top > self.threshold).type(torch.float64), ( output_bot >self.threshold).type(torch.float64), ( target_bot > self.threshold).type(torch.float64))
        
        return self.lambda1* loss1 + self.lambda2 * loss2 + self.lambda3 * loss3