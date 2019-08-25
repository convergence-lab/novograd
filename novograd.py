# Author Masashi Kimura (Convergence Lab.)
import torch
from torch import optim
import math

class NovoGrad(optim.Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.95, 0.98), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(NovoGrad, self).__init__(params, defaults)
        self._lr = lr
        self._beta1 = betas[0]
        self._beta2 = betas[1]
        self._eps = eps
        self._wd = weight_decay
        self._grad_averaging = grad_averaging

        self._momentum_initialized = False

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        if not self._momentum_initialized:
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    grad = p.grad.data
                    if grad.is_sparse:
                        raise RuntimeError('NovoGrad does not support sparse gradients')

                    v = torch.norm(grad)**2
                    m = grad/(torch.sqrt(v) + self._eps) + self._wd * p.data
                    state['v'] = v
                    state['m'] = m
            self._momentum_initialized = True
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                v, m = state['v'], state['m']

                grad = p.grad.data
                g2 = torch.norm(grad)**2
                if self._wd > 0.:
                    grad += self._wd*p
                if self._grad_averaging:
                    grad *= (1. - self._beta1)
                
                v = self._beta2*v + (1. - self._beta2)*g2
                m = self._beta1*m + (grad / (torch.sqrt(v) + self._eps) + self._wd*p.data)
         
                state['v'], state['m']= v, m
                p.data.add_(-group['lr'], m)
        return loss
