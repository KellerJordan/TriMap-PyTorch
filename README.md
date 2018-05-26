# TriMap-PyTorch
Implementation of TriMap dimensionality reduction in PyTorch

Making .gif animations requires `imagemagick`.

    cd data
    python3 mnist_np.py
    cd ..
    python3 main.py --dataset=mnist70k --num_iters=1500 --save_fig --animate --verbose

Animations of training (see [tSNE-Animation](https://github.com/KellerJordan/tSNE-Animation) for the equivalent animations with t-SNE):

![batch gradient descent with momentum](https://github.com/KellerJordan/figures/blob/master/sgd-momentum60k.gif)

![linear annealing](https://github.com/KellerJordan/figures/blob/master/mnist70k-gd-momentum-anneal2.gif)

### Authors

Keller Jordan, Ehsan Amid
