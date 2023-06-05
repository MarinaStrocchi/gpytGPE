# gpytGPE

A robust univariate Gaussian Process Emulator (GPE) implementation around GPyTorch (gpyt).

> :warning: **gpytGPE has been completely refactored and is now maintained under the new name of GPErks! (https://github.com/stelong/GPErks)**

This library contains the tools needed for constructing a univariate Gaussian process emulator (GPE) as a surrogate model of a generic map *X -> y*. The map (e.g. a computer code input/output) is simply described by the (*N x D*) *X* matrix of input parameters and the respective (*N x 1*) *y* vector of one output feature, both provided by the user. GPEs are implemented as the sum of a mean function given by a linear regression model (with first-order degreed polynomials) and a centered (zero-mean) Gaussian process regressor with RBF/Matern kernel as covariance function.

The GPE training can be performed either against a validation set (by validation loss) or by itself (by training loss), using an "early-stopping" criterion to stop training at the point when performance on respectively validation dataset/training dataset starts to degrade. The entire training process consists in firstly performing a *K*-fold cross-validation training by validation loss, producing a set of *K* GPEs. Secondly, a final additional GPE is trained on the entire dataset by training loss, using an early-stopping patience level and maximum number of allowed epochs both equal to the average stopping epoch number previously obtained across the cross-validation splits. Each single training is performed by restarting the loss function optimization algorithm from different initial points in the hyperparameter high-dimensional space by log-uniformly sampling the initial guess.

At each training epoch, it is possible to monitor training loss, validation loss and a metric of interest (the last two only if applicable i.e., if traning against a validation set). Available metrics are all the regression metrics provided by the third-party, Torchmetrics (https://torchmetrics.readthedocs.io/en/latest/) library. Losses over epochs plots can be automatically outputed. It is also possible to switch between GPE's noise-free and noisy implementations. Data is automatically scaled and backtransformed within the code (input scaled to unit cube and output standardized). The user can additionally opt for log-transforming the output before it gets standardized before training.

The entire code runs on both CPU and GPU. The cross-validation training loop is implemented to run in parallel with multiprocessing.

---
## Information

**Author**: [stelong](https://github.com/stelong)

---
## Getting Started

```
git clone https://github.com/MarinaStrocchi/gpytGPE.git
```

---

## Installing

```
cd gpytGPE/
```
```
# (this block is optional)
conda create -n py38 python=3.18.13
conda activate py38
python -m pip install --upgrade pip
```
```
pip install .
```

