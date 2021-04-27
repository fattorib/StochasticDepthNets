# Deep Networks with Stochastic Depth - CIFAR10

PyTorch implementation of ResNet110 as described in *Deep Networks with Stochastic Depth* ([Huang et al.](https://arxiv.org/abs/1603.09382))

# Setup

# Training 
ResNet110 was trained for 500 epochs, with a batch size of 128 on a 45000/5000 train/validation split. We train the model using SGD with momentum and anneal the learning rate by a factor of 10 at 
250 and 375 epochs. Standard data augmentation procedures are used (horizontal flips & rotations). See training time breakdown for two different GPUs:

| GPU | Time (hrs) |
|-----------------|----------|
| Tesla V100 (Google Colab) | 5 hours |
| GTX 1070 (Local)       | 9 hours |

Something to note is that using ```torch.cuda.amp``` for mixed precision training on the V100 provided minimal speedup. (WHY)

To train the model from scratch, run ```train_model_script.py```.

# Results
The ResNet110 model in the paper achieves a top 1 error rate of 5.25%. My implementation reaches a top 1 error rate of 5.47%. Trained model weights are saved under ```weights/ResNet110.pth```
