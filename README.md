Summary
=======
Equivalent implementation of XOR in dynamic Neural Network frameworks.  
Currently known frameworks are: dynet, chainer, pyTorch.

Details (Network and Training)
==============================
IN -> linear(2x2) -> sigmoid -> linear(2x1) -> sigmoid -> OUT  
Optimizer: SGD, with learning rate 1  
Loss Fun : binary cross entropy  
Weigths : initialised from normal distribution (sigma = 1)  
Training : 500 itterations  
Plots : mean training (with standard error) for a 100 training runs  
