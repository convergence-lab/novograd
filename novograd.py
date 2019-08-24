import torch
from torch import optim
import math

class NovoGrad(optim.Optimizer):
    def __init__(self, params, grad_averaging=False, lr=0.1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
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

                    v = torch.norm(grad)
                    state['step'] = 0
                    state['v'] = v
                    state['m'] = grad/torch.sqrt(v + self._eps) + self._wd * p.data
                    state['grad_ema'] = None
            self._momentum_initialized = True
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                state = self.state[p]
                state['step'] += 1
                v, m, grad_ema, step = state['v'], state['m'], state['grad_ema'], state['step']

                grad = p.grad.data     

                g2 = torch.sum(grad**2)
                grad_ema = g2 if grad_ema is None else grad_ema * \
                    self._beta2 + g2*(1. - self._beta2)
                grad *= 1.0 / (torch.sqrt(grad_ema) + self._eps)

                if self._wd > 0.:
                    grad += self._wd*p
                if self._grad_averaging:
                    grad *= (1. - self._beta1)
                
                gn2 = torch.norm(grad)
                v = self._beta2*v + (1. - self._beta2)*gn2
                m = self._beta1*m + (grad / (torch.sqrt(v) + self._eps) + self._wd*p.data)
         
                bias_correction1 = 1 - self._beta1 ** step
                bias_correction2 = 1 - self._beta2 ** step
                step_size = self._lr * (math.sqrt(bias_correction2) + self._eps) / bias_correction1
                
                state['v'], state['m'], state['grad_ema'] = v, m, grad_ema
                p.data.add_(-step_size, m)
        return loss
