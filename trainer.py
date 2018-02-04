from __future__ import print_function

import pickle

import numpy as np
from sklearn.decomposition import TruncatedSVD
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable

from models import *
from triplets import generate_triplets


class Trainer(object):

    def __init__(self, config, X):
        pass
    
    def embed(self, num_iters=2000, embed_init=None,
              optimizer='gd', lr=None, t=2,
              return_seq=False, verbose=False):

        num_examples = self.X.shape[0]
        num_triplets = self.triplets.shape[0]
        
        if embed_init is None:
            embed_init = 0.0001 * np.random.normal(size=[self.X.shape[0], 2])
        
        model = TriMapper(self.triplets, self.weights, out_shape=[num_examples, 2],
                          embed_init=embed_init, t=t)
        model.cuda()

        tol = 1e-7
        C = np.inf
        best_C = np.inf
        best_Y = None

        if lr == None:
            eta = num_examples * 1000.0 / num_triplets
        else:
            eta = lr

        if optimizer == 'gd':
            trainer = optim.SGD(model.parameters(), lr=eta)
        elif optimizer == 'gd-momentum':
            trainer = optim.SGD(model.parameters(), lr=eta, momentum=.9)
        elif optimizer == 'adam':
            trainer = optim.Adam(model.parameters(), lr=eta)
        elif optimizer == 'adadelta':
            trainer = optim.Adadelta(model.parameters(), lr=eta)
        elif optimizer == 'rmsprop':
            trainer = optim.RMSprop(model.parameters(), lr=eta)

        if return_seq:
            Y_seq = []

        for i in range(num_iters):
            old_C = C

            loss, num_viol = model()
            trainer.zero_grad()
            loss.backward()
            trainer.step()
            
            C = loss.data.cpu().numpy()
            viol = float(num_viol) / num_triplets
            
            if optimizer in ['gd', 'gd-momentum']:
                if old_C < C - tol:
                    eta *= 0.9
                else:
                    eta *= 1.01
                trainer.param_groups[0]['lr'] = eta

            if return_seq:
                Y = model.get_embeddings()
                Y_seq.append(Y)
            
            if C < best_C:
                best_C = C
                best_Y = model.get_embeddings()

            if verbose and (i+1) % 100 == 0:
                print('Iteration: %4d, Loss: %3.3f, Violated triplets: %0.4f' % (i+1, loss, viol))

        return Y_seq if return_seq else best_Y
    
    def save_triplets(self, path):
        with open(path, 'wb') as f:
            pickle.dump((self.triplets, self.weights), f)
    
    def load_triplets(self, path):
        with open(path, 'rb') as f:
            self.triplets, self.weights = pickle.load(f)
