# Differentially Private Follow-the-Regularized-Leader (DP-FTRL)

Code for ["Practical and Private (Deep) Learning without Sampling or Shuffling"](https://arxiv.org/abs/2103.00039)
by Peter Kairouz, Brendan McMahan, Shuang Song, Om Thakkar, Abhradeep Thakurta, and Zheng Xu.
The paper proposed Differentially Private Follow-the-Regularized-Leader (DP-FTRL),
a differentially private algorithm that does not rely on shuffling or subsampling
as in Differentially Private Stochastic Gradient Descent (DP-SGD), but achieves
comparable (or even better) utility. 

This repository contains the implementation and experiments for *centralized learning*. 
Please see [another repository](https://github.com/google-research/federated/blob/master/dp_ftrl/README.md)
for the implementation and experiments in the *Federated learning setting*.

This is not an officially supported Google product.


## Overview of the code

The code is written in PyTorch. 
*  `main.py` contains the training and evaluation steps for three datasets: `MNIST`, `CIFAR10`, and `EMNIST (byMerge)`.
*  `optimizers.py` contains the DP-FTRL optimizer, and `ftrl_noise.py` contains the tree-aggregation protocol, which
   is the core of the optimizer. 
*  `privacy.py` contains the privacy accounting function for DP-FTRL.


## Example usage of the code

First, install the packages needed. The code is implemented using PyTorch, with
the [Opacus library](https://github.com/pytorch/opacus) 
used for gradient clipping (but not noise addition).
```bash
# This is an example for creating a virtual environment. 
sudo apt install python3-dev python3-virtualenv python3-tk imagemagick
virtualenv -p python3.7 --system-site-packages env3
. env3/bin/activate

# Install the packages.
pip install -r requirements.txt
```

Then, we set up a path where the data will be downloaded.
```bash
export ML_DATA="path to where you want the datasets saved"  # set a path to store data
```

Now we can run the code to do DP-FTRL training. 
For example, the following command trains a small CNN for `CIFAR-10` 
with DP-FTRL noise `10.0`, batch size `500` for `100` epochs. 
```bash
run=1
CUDA_VISIBLE_DEVICES=0 PYTHONHASHSEED=$(($run - 1)) python main.py \
    --data=cifar10 --run=$run --dp_ftrl=true \
    --epochs=100 --batch_size=500 --noise_multiplier=10 \
    --learning_rate=20 --momentum=0.9
```
The results, including accuracies and privacy guarantees, will be written as a
tensorboard file in the current folder (can be configured with flag `dir`).
You can view it with tensorboard
```bash
tensorboard --port 6006 --logdir .
```


## Citing this work

```
@article{kairouz2021practical,
  title={Practical and Private (Deep) Learning without Sampling or Shuffling},
  author={Kairouz, Peter and McMahan, H Brendan and Song, Shuang and Thakkar, Om and Thakurta, Abhradeep and Xu, Zheng},
  journal={arXiv preprint arXiv:2103.00039},
  year={2021}
}
```
