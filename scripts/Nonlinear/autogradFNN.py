## # "Statistical foundations of machine learning" software
# autogradFNN.py
# Author: G. Bontempi
# Comparison of symbolic and numeric differentiation in pyTorch for the FNN example in the slides

import torch
import torch.nn.functional as F
from torch.autograd import grad

def g(a):
    return 1.0/(1.0+torch.exp(-a))

def gp(a):
    return torch.exp(-a) / (1 + torch.exp(-a))**2
    


y = torch.tensor([-4.5])
x = torch.tensor([-1.7])
w11_1 = torch.tensor([0.13], requires_grad=True)
w12_1 = torch.tensor([-2.0], requires_grad=True)

w11_2 = torch.tensor([1.0], requires_grad=True)
w21_2 = torch.tensor([0.5], requires_grad=True)

a1_1= w11_1 * x
z1 = g(a1_1)

a2_1 = w12_1 * x
z2 = g(a2_1)  # 1/(1+exp(-a2_1))

a1_2 = w11_2 * z1 + w21_2 * z2
yhat = g(a1_2)  #

L = (y-yhat)**2 # (

# Compute analytical gradients  by symbolic differentiation
Sgrad_L_w11_1 = -2 *(y-yhat) * gp(a1_2) * w11_2 * gp(a1_1) * x   # dL/dw11_1
Sgrad_L_w12_1 = -2 *(y-yhat)*gp(a1_2) * w21_2 * gp(a2_1) * x   # dL/dw12_1
Sgrad_L_w11_2 = -2 *(y-yhat)*gp(a1_2) * z1                     # dL/dw11_2
Sgrad_L_w21_2 = -2 *(y-yhat)*gp(a1_2) * z2                     # dL/dw21_2



L.backward()

grad_L_w11_1 = w11_1.grad # grad(L, w11_1, retain_graph=True)
grad_L_w12_1= w12_1.grad  #grad(L, w12_1, retain_graph=True)
grad_L_w11_2 =w11_2.grad  # grad(L, w11_2, retain_graph=True)
grad_L_w21_2= w21_2.grad  # grad(L, w21_2, retain_graph=True)


## Comparison of symbiolic and numerical differentiation

print('L=',L.detach().numpy(), '\n dL/dw11_2=', 
      grad_L_w11_2,':',  Sgrad_L_w11_2.detach().numpy())

print('\n dL/dw21_2=', 
      grad_L_w21_2,':',  Sgrad_L_w21_2.detach().numpy())

print('\n dL/dw11_1=', 
      grad_L_w11_1,':',  Sgrad_L_w11_1.detach().numpy())

print('\n dL/dw12_1=', 
      grad_L_w12_1,':',  Sgrad_L_w12_1.detach().numpy())



  