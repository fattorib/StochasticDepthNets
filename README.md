# StochasticDepthNets
PyTorch implementation of ResNet110 as described in Deep Networks with Stochastic Depth (Huang et al.)

## Training the ResNets on Imagenette (add link)

Experiments attempted
- For ResNet110: lr = 0.1 at the start caused slow extermely slow convergence. Following advice of ResNet papers, we experiment with a "warm-up" learning rate of 0.01. (Doing short hyperparam search on [None, 2, 8, 15] warmup steps at 0.01)
- More data augmentation: Adding randomaffine transforms 
- AdamW optimizer(?)

Transfer learning benchmarks:
ResNet18 - 5 epochs: 94% acc
ResNet101 - 10 epochs: 
