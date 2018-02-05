import os
import argparse
import pickle

from wrapper import Wrapper
from visualize import *


parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str, default='mnist2500')

parser.add_argument('--num_iters', type=int, default=1000)
parser.add_argument('--optimizer', type=str, default='gd-momentum')
parser.add_argument('--out_dim', type=int, default=2)
parser.add_argument('--svd_dim', type=int, default=50)
parser.add_argument('--lr', type=float, default=1000.0)

parser.add_argument('--anneal_scheme', type=int, default=0,
                    help='annealing scheme: 0 is no annealing, '+
                    '1 is linear annealing from tmin to tmax after half of iterations')
parser.add_argument('--t', type=float, default=2.0)
parser.add_argument('--t_max', type=float, default=3.0)

parser.add_argument('--save_fig', action='store_true')
parser.add_argument('--animate', action='store_true')

parser.add_argument('--verbose', action='store_true')
parser.add_argument('--print_every', type=int, default=100)


def main(config):

    # initialize trimap
    with open('data/%s.pkl' % config.dataset, 'rb') as f:
        X, labels = pickle.load(f)

    trimap = Wrapper(config)

    triplets_path = 'triplets/%s.pkl' % config.dataset
    if os.path.isfile(triplets_path):
        trimap.load_triplets(triplets_path)
    else:
        if not os.path.exists('triplets'):
            os.makedirs('triplets')
        trimap.generate_triplets(X, triplets_path)

    if config.save_fig:
        # create and save an embedding
        fig_name = '%s-%s' % (config.dataset, config.optimizer)
        fig_temp = 'figures/%s.' + ('gif' if config.animate else 'png')

        if not os.path.exists('figures'):
            os.makedirs('figures')

        i = 0
        while os.path.exists(fig_temp % (fig_name+str(i))):
            i += 1
        fig_path = fig_temp % (fig_name+str(i))
        
        if config.animate:
            Y_seq = trimap.embed(return_seq=True)
            savegif(Y_seq, labels, fig_name, fig_path)
        else:
            Y = trimap.embed()
            savepng(Y, labels, fig_name, fig_path)
    else:
        Y = trimap.embed()
        scatter(Y, labels)

if __name__ == '__main__':
    config, unparsed = parser.parse_known_args()
    main(config)
