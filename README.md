## FashionMNIST_classification

Multi-classification project over the Fasion MNIST dataset. This repo gives a general structure for pytorch projects.

### Best performances reached : 

Number of train epochs : `50`

Test accuracy : `91.1%`


Model description :
```
54666 trainable parameters

ConvNet(
    (convlayers): Sequential(
        (0): Conv2d(1, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU(inplace=True)
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (3): Conv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (4): ReLU(inplace=True)
        (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (6): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (7): ReLU(inplace=True)
        (8): AdaptiveAvgPool2d(output_size=7)
    )
    (linear_layer): Sequential(
        (0): Linear(in_features=3136, out_features=10, bias=True)
        (1): ReLU(inplace=True)
    )
)


Optimizer
========
Adadelta (
Parameter Group 0
    capturable: False
    differentiable: False
    eps: 1e-06
    foreach: None
    lr: 1.0
    maximize: False
    rho: 0.9
    weight_decay: 0
)
```

### TODO
- Regularize this best model 