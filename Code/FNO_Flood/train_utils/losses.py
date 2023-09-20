import numpy as np
import torch
import torch.nn.functional as F


class LpLoss(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

class LpLoss2(object):
    '''
    loss function with rel/abs Lp loss
    '''
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1) + 1e-5, self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1) + 1e-5, self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / (y_norms + 1e-5))
            else:
                return torch.sum(diff_norms / (y_norms + 1e-5))

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.abs(x, y)

def GeoPC_loss(input_data, outputH, data_condition, init_condition):

    #data_loss
    h_gt = data_condition[0]

    # loss_d = loss_h + loss_qx + loss_qy
    loss = LpLoss(size_average=True)
    h_c = outputH
    # h_g = torch.unsqueeze(h_gt, dim=0)
    loss_h = loss(h_c, h_gt)
    loss_d = loss_h

    # if i == 0:
    #     _EPSILON = 1e-6
    h_init = init_condition[0]
    h_cc = outputH[:, 0, :, :]
    #     h_c = torch.squeeze(h_c)
    loss_c = loss(h_cc, h_init)

    return loss_d, loss_c