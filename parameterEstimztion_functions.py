import os, sys
import torch
import numpy as np
import scipy as sp
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd.functional import jacobian
from torch.autograd.functional import jvp


import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

import torch.autograd.functional as taf

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

    
def yp_fun(p, X):
    #x' = sigma*(y-x)
    #y' = x*(rho-z)-y
    #z' = x*y - beta*z
    #sigma = 10, rho = 28 beta=8/3, 
        
    xp1 = p[0]*(X[:,1] - X[:,0])
    xp2 = X[:,0]*(p[1] - X[:,2]) - X[:,1]
    xp3 = X[:,0]*X[:,1] - p[2]*X[:,2]
        
    yp = torch.cat((xp1[:,None], xp2[:,None], xp3[:,None]), dim=1)
        
    return yp
    
                
def dynamical_system(p, x0, yp_fun, nt, dt):
        
    X = torch.zeros(x0.shape[0], 3, nt+1)
    X[:, :, 0] = x0
    for i in range(nt):
        Xi = X[:,:,i].clone() 
        k1 = yp_fun(p, Xi)
        k2 = yp_fun(p, Xi + dt/2*k1)
        k3 = yp_fun(p, Xi + dt/2*k2)
        k4 = yp_fun(p, Xi + dt*k3)
        Xp =  Xi + dt/6*(k1+2*k2+2*k3+k4)
        X[:,:,i+1] = Xp
    return X


def phaseDiagram(X):
    # Data for a three-dimensional line
    x = X[0,0,:].detach().cpu()
    y = X[0,1,:].detach().cpu()
    z = X[0,2,:].detach().cpu()

    plt.figure(1)
    plt.subplot(3,1,1)
    plt.plot(x)
    plt.subplot(3,1,2)
    plt.plot(y)
    plt.subplot(3,1,3)
    plt.plot(z)
    
    plt.figure(2)
    ax = plt.axes(projection='3d')
    ax.plot3D(x, y, z)
    
    return ax
    
Nt = 50  
dt = 1e-2
      
p = torch.tensor([10, 28, 8/3])
#p.requires_grad = True
x0 = torch.randn(1,3)
X = dynamical_system(p, x0, yp_fun, Nt, dt)

def dynamical_system_wrapper(p):
    Nt = 50
    dt = 1e-2
    x0 = torch.ones(1,3)
    X = dynamical_system(p, x0, yp_fun, Nt, dt)
    return X

#J = jacobian(dynamical_system_wrapper, p)

# Computing J^TJv for any v
# J^T ( Jv)
v = torch.randn_like(p)
Jv = jvp(dynamical_system_wrapper,p,  v)



print('h')
