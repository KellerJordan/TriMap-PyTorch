from __future__ import print_function

import numpy as np
from sklearn.neighbors import NearestNeighbors as knn
from sklearn.decomposition import TruncatedSVD


def generate_triplets(X, kin=50, kout=10, kr=5, svd_dim=50,
                      weight_adj=False, random_triplets=True, verbose=False):
    """
    generate_triplets()
    Created on Sat May 27 12:46:25 2017

    @author: ehsanamid
    """

    X = X.astype(np.float32)
    X -= np.min(X)
    X /= np.max(X)
    X -= np.mean(X, axis=0)
    # set svd_dim = None for no projection
    if svd_dim:
        X = TruncatedSVD(n_components=svd_dim, random_state=0).fit_transform(X)
    
    num_extra = max(kin+50, 50) # look up more neighbors
    n = X.shape[0]
    nbrs = knn(n_neighbors= num_extra + 1, algorithm='auto').fit(X)
    distances, indices = nbrs.kneighbors(X)
    sig = np.maximum(np.mean(distances[:, 10:20], axis=1), 1e-20) # scale parameter
    P = np.exp(-distances**2/np.reshape(sig[indices.flatten()],[n, num_extra + 1])/sig[:, np.newaxis])
    sort_indices = np.argsort(-P, axis = 1) # actual neighbors

    triplets = np.zeros([n * kin * kout, 3], dtype=np.int32)
    weights = np.zeros(n * kin * kout)

    cnt = 0
    for i in range(n):
        for j in range(kin):
            sim = indices[i,sort_indices[i, j+1]]
            p_sim = P[i,sort_indices[i, j+1]]
            rem = indices[i,sort_indices[i, :j+2]].tolist()
            l = 0
            while (l < kout):
                out = np.random.choice(n)
                if out not in rem:
                    triplets[cnt] = [i, sim, out]
                    p_out = max(np.exp(-np.sum((X[i] - X[out])**2) / (sig[i] * sig[out])), 1e-20)
                    weights[cnt] = p_sim / p_out
                    rem.append(out)
                    l += 1
                    cnt += 1
        if verbose and (i+1) % 500 == 0:
            print('Generated triplets %d / %d' % (i+1, n))
    if random_triplets:
        triplets_rand = np.zeros([n * kr, 3], dtype=np.int32)
        weights_rand = np.zeros(n * kr)
        for i in range(n):
            cnt = 0
            while cnt < kr:
                sim = np.random.choice(n)
                out = np.random.choice(n)
                if sim == i or out == i or out == sim:
                    continue
                p_sim = max(np.exp(-np.sum((X[i]-X[sim])**2)/(sig[i] * sig[sim])), 1e-20)
                p_out = max(np.exp(-np.sum((X[i]-X[out])**2)/(sig[i] * sig[out])), 1e-20)
                if p_sim < p_out:
                    sim, out = out, sim
                    p_sim, p_out = p_out, p_sim
                triplets_rand[i * kr + cnt] = [i, sim, out]
                weights_rand[i * kr + cnt] = p_sim / p_out
                cnt += 1
            if verbose and (i+1) % 500 == 0:
                print('Generated random triplets %d / %d' % (i+1, n))
        triplets = np.vstack((triplets, triplets_rand))
        weights = np.hstack((weights, weights_rand))
    triplets = triplets[~np.isnan(weights)]
    weights = weights[~np.isnan(weights)]
    weights /= np.max(weights)
    weights += 0.0001
    if weight_adj:
        weights = np.log(1 + 50 * weights)
        weights /= np.max(weights)
    
    return triplets, weights
