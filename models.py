import torch
from torch import nn
from torch.autograd import Variable


class TriMap(nn.Module):

    def __init__(self, triplets, weights, out_shape, embed_init, t=2):
        super(TriMapper, self).__init__()
        n, num_dims = out_shape
        self.Y = nn.Embedding(n, num_dims, sparse=False)
        self.Y.weight.data = torch.Tensor(embed_init)
        
        self.triplets = Variable(torch.cuda.LongTensor(triplets))
        self.weights = Variable(torch.cuda.FloatTensor(weights))
        
        self.t = t
    
    def forward(self):
        y_ij = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 1])
        y_ik = self.Y(self.triplets[:, 0]) - self.Y(self.triplets[:, 2])
        d_ij = 1 + torch.sum(y_ij**2, -1)
        d_ik = 1 + torch.sum(y_ik**2, -1)
        num_viol = torch.sum((d_ij > d_ik).type(torch.FloatTensor))
#         loss = self.weights.dot(torch.log(1 + d_ij / d_ik))
#         loss = self.weights.dot(d_ij / (d_ij + d_ik))
        ratio = d_ij / d_ik
        loss = self.weights.dot(log_t(ratio, self.t))
        return loss, num_viol
    
    def get_embeddings(self):
        return self.Y.weight.data.cpu().numpy()

def log_t(l, t=2):
    return 1 - 1 / (1 + l)**(t - 1)
