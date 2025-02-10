## # "Statistical foundations of machine learning" software
# autograd.py
# Author: G. Bontempi
# Comparison of symbolic and numeric differentiation in pyTorch

import torch
import torch.nn.functional as F
from torch.autograd import grad

y = torch.tensor([4.5])
x = torch.tensor([1.7])
w = torch.tensor([2.8], requires_grad=True)
b = torch.tensor([3.0], requires_grad=True)

yhat = x * w + b  ## 1*2+3


L = (y-yhat)**2 # (1-5)^2

## dL/dw=-2 *(y-yhat) * dyhat/dw=2 *(y-yhat) * x
Dsymbolic=-2*(y-yhat)*x

## dL/db=-2 *(y-yhat) * dyhat/dw=2 *(y-yhat) 
Dsymbolic2=-2*(y-yhat)


grad_L_w1 = grad(L, w, retain_graph=True)
grad_L_b = grad(L, b, retain_graph=True)



print('L=',L.detach().numpy(),
      '\n dL/dw=', grad_L_w1,':', 
      Dsymbolic.detach().numpy())

print('\n dL/db=',grad_L_b,':',
      Dsymbolic2.detach().numpy())