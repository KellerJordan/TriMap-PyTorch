import numpy as np
import torch
from torch import nn
from torch.autograd import Variable


class TriMap(nn.Module):

    def __init__(self, triplets, weights, out_shape, embed_init, use_cuda=False):
        super(TriMap, self).__init__()
        n, num_dims = out_shape
        self.Y = nn.Embedding(n, num_dims, sparse=False)
        self.Y.weight.data = torch.Tensor(embed_init)
        
        if use_cuda:
            self.triplets = Variable(torch.from_numpy(triplets).type(torch.cuda.LongTensor))
            self.weights = Variable(torch.cuda.FloatTensor(weights))
        else:
            self.triplets = Variable(torch.from_numpy(triplets).type(torch.LongTensor))
            self.weights = Variable(torch.FloatTensor(weights))
    
    def forward(self, t):
        y_ij = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 1])
        y_ik = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 2])
        d_ij = 1 + torch.sum(y_ij**2, -1)
        d_ik = 1 + torch.sum(y_ik**2, -1)

        # loss = self.weights.dot(log_t_ratio(d_ij, d_ik, t))
        # loss = self.weights.dot(d_ij / (d_ij + d_ik))
        loss = self.weights.dot(log_t(d_ij / d_ik, t))
        num_viol = torch.sum((d_ij > d_ik).type(torch.FloatTensor))

        return loss, num_viol
    
    def get_embeddings(self):
        return self.Y.weight.data.cpu().numpy()

def log_t(x, t=2):
    if t == 1:
        return torch.log(x + 1)
    else:
        unscaled = (x + 1)**(1 - t)
        return (unscaled - 1) / (1 - t)

# alternate loss function -- punish deviation from proper ratio in either direction
def abs_log(a, b):
    return torch.abs(torch.log(a * b))
