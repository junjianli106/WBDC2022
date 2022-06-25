import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# 支持多分类和二分类
class FocalLoss(nn.Module):
    """
    This is a implementation of Focal Loss with smooth label cross entropy supported which is proposed in
    'Focal Loss for Dense Object Detection. (https://arxiv.org/abs/1708.02002)'
    Focal_Loss= -1*alpha*(1-pt)^gamma*log(pt)
    :param num_class:
    :param alpha: (tensor) 3D or 4D the scalar factor for this criterion
    :param gamma: (float,double) gamma > 0 reduces the relative loss
    for well-classified examples (p>0.5) putting more
    focus on hard misclassified example
    :param smooth: (float,double) smooth value when cross entropy
    :param balance_index: (int) balance class index,
    should be specific when alpha is float
    :param size_average: (bool, optional) By default,
    the losses are averaged over each loss element in the batch.
    """
    def __init__(self, num_class, alpha=None, gamma=2,
                smooth=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.num_class = num_class
        self.alpha = alpha
        self.gamma = gamma
        self.smooth = smooth
        self.size_average = size_average

        if self.alpha is None:
            self.alpha = torch.ones(self.num_class, 1)
        elif isinstance(self.alpha, (list, np.ndarray)):
            assert len(self.alpha) == self.num_class
            self.alpha = torch.FloatTensor(alpha).view(self.num_class, 1)
            self.alpha = self.alpha / self.alpha.sum()
        else:
            raise TypeError('Not support alpha type')
        if self.smooth is not None:
            if self.smooth < 0 or self.smooth > 1.0:
                raise ValueError('smooth value should be in [0,1]')

    def forward(self, input, target):
        logit = F.softmax(input, dim=1)

        if logit.dim() > 2:
            # N,C,d1,d2 -> N,C,m (m=d1*d2*...)
            logit = logit.view(logit.size(0), logit.size(1), -1)
            logit = logit.permute(0, 2, 1).contiguous()
            logit = logit.view(-1, logit.size(-1))
        target = target.view(-1, 1)

        # N = input.size(0)
        # alpha = torch.ones(N, self.num_class)
        # alpha = alpha * (1 - self.alpha)
        # alpha = alpha.scatter_(1, target.long(), self.alpha)
        epsilon = 1e-10
        alpha = self.alpha
        if alpha.device != input.device:
            alpha = alpha.to(input.device)

        idx = target.cpu().long()
        one_hot_key = torch.FloatTensor(target.size(0), self.num_class).zero_()
        one_hot_key = one_hot_key.scatter_(1, idx, 1)
        if one_hot_key.device != logit.device:
            one_hot_key = one_hot_key.to(logit.device)

        if self.smooth:
            one_hot_key = torch.clamp(
                one_hot_key, self.smooth, 1.0 - self.smooth)
        pt = (one_hot_key * logit).sum(1) + epsilon
        logpt = pt.log()

        gamma = self.gamma

        alpha = alpha[idx]
        loss = -1 * alpha * torch.pow((1 - pt), gamma) * logpt

        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
        return loss
    
class HMC(nn.Module):
    """A classifier for Hierarchical multi-label classification"""

    def __init__(self, 
                 feature_size,
                 L1_labels_num,
                 L2_labels_num,
                 L12_table,
                 mlp_hidden_size=None,
                 mask_value=None):
        """Construct a classifier for HMC
        Args:
            feature_size: feature size in input_tensor
            mlp_hidden_size: Number of output dimensions for MLP
            L1_labels_num: Number of labels on the first level
            L2_labels_num: Number of labels on the second level
            L12_table: A list of some lists.
                       For example, L12_table[i][j] is a number, which means 
                       that L12_table[i][j](L2 label number) belongs to i(L1 label)
            mask_value: a float number which make exp(mask_value) close to zero
        """

        super(HMC, self).__init__()

        self.mlp_hidden_size = mlp_hidden_size
        self.mask_value = mask_value
        self.feature_size = feature_size
        self.L1_labels_num = L1_labels_num
        self.L2_labels_num = L2_labels_num
        self.L12_table = L12_table

        assert len(L12_table) == L1_labels_num
        assert self.check_L12_table(L12_table)

        self.fc_L1_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L1_2 = nn.Linear(self.mlp_hidden_size, self.L1_labels_num)
        self.fc_L2_1 = nn.Linear(self.feature_size, self.mlp_hidden_size)
        self.fc_L2_2 = nn.Linear(2*self.mlp_hidden_size, self.L2_labels_num)
        

    def check_L12_table(self, L12_table):
        """check the Legality of L12_table"""

        L2_labels = [num for lst in L12_table for num in lst]
        assert len(L2_labels) == self.L2_labels_num
        for i in range(self.L2_labels_num):
            if i not in L2_labels:
                return False
        return True

    def forward(self,x):
        """forward computation
        Args:
            x: input_tensor, must only have 2 dims
        """

        assert len(x.shape) == 2

        L1 = F.relu(self.fc_L1_1(x))
        L2 = F.relu(self.fc_L2_1(x))
        
        L2 = torch.cat((L1,L2), dim=1)
        L1 = F.relu(self.fc_L1_2(L1))
        L1 = F.softmax(L1, dim=1)
        
        L2 = F.relu(self.fc_L2_2(L2))

        L1_label = L1.argmax(dim=1)
        mask = torch.ones_like(L2) * self.mask_value

        """for-loop may be not a good choice.
        But I don't have any other methods
        """
        for i,element in enumerate(L1_label):
            idx = element.item()
            mask[i,self.L12_table[idx]]=0

        L2 += mask
        L2 = F.softmax(L2,dim=1)

        return L1,L2



