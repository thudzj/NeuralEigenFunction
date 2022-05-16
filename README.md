# NeuralEF: Deconstructing Kernels by Deep Neural Networks

Code for the paper [NeuralEF: Deconstructing Kernels by Deep Neural Networks](https://arxiv.org/pdf/2205.00165.pdf).

## Environment settings and libraries we used in our experiments

This project is tested under the following environment settings:
- Python 3
- PyTorch: 1.9.0


## Usage

### Cope with RBF and polynomial kernels
```
python neuralef-classic-kernels.py
```


### Cope with NN-GP kernels on Circles/Two moon
```
python neuralef-toy-nngpkernels.py
```

### Cope with CNN-GP kernels on MNIST
```
python neuralef-mnist-cnngpkernels.py
```


### Cope with NTKs on CIFAR-10

#### Estimate the eigenfunctions of the NTK corresponding to a binary classifier
```
python neuralef-cifar-ntks.py --nef-amp --classes 0 1 --ood-classes 8 9 --draw \
                              --resume path/to/pretrained_models
```

#### Leverage NeuralEF to accelerate linearized Laplace approximation
```
python neuralef-cifar-ntks.py --nef-amp --ntk-std-scale 20
```

### Leverage NeuralEF to approximate the implicit kernel induced by SGD trajectory
```
python neuralef-cifar-sgd-trajectory.py --data-dir path/to/data \
                                        --nef-amp --nef-class-cond --swa-lr 0.1 \
                                        --pre-trained-dir path/to/pretrained_models
```

## Giving Credit
If you use this code in your work, we ask that you cite the paper.

## Acknowledgement
The implementation of the baselines is based on [SWAG](https://github.com/wjmaddox/swa_gaussian) and [SpIN](https://github.com/deepmind/spectral_inference_networks).
