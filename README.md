# PyTorch implementation of NovoGrad 

## Install 

```
pip install novograd
```

## Notice

When using NovoGrad, learning rate scheduler play an important role.  Do not forget to set it.

## Performance

### MNIST

Under Trained 3 epochs, same Architecture Neural Netwrok. 

|                | Test Acc(%) |  lr    | lr scheduler   | beta1  | beta2 | weight decay |
|:---------------|:------------|:-------|:---------------|:-------|:------|:-------------|
| Momentum SGD   |  96.92      | 0.01   | None           |  0.9   | N/A   |   0.001      |
| Adam           |  96.72      | 0.001  | None           |  0.9   | 0.999 |   0.001      |
| AdamW          |  97.34      | 0.001  | None           |  0.9   | 0.999 |   0.001      |
| NovoGrad       |  97.55      | 0.01   | cosine         |  0.95  | 0.98  |   0.001      |

## Refference
Boris Ginsburg, Patrice Castonguay, Oleksii Hrinchuk, Oleksii Kuchaiev, Vitaly Lavrukhin, Ryan Leary, Jason Li, Huyen Nguyen, Jonathan M. Cohen, Stochastic Gradient Methods with Layer-wise Adaptive Moments for Training of Deep Networks, 	arXiv:1905.11286 [cs.LG], https://arxiv.org/pdf/1905.11286.pdf

