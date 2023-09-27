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
import torch.optim as optim
from scipy.sparse.linalg import spsolve

import torchvision
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

import torch.autograd.functional as taf

#device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'

class dynamical_system(nn.Module):
    def __init__(self, dt, nt, p=torch.rand(3)):
        super(dynamical_system, self).__init__()
        
        self.dt = dt
        self.nt = nt
        
        self.p  = nn.Parameter(p)
    
    def yp_fun(self, X):
        #x' = sigma*(y-x)
        #y' = x*(rho-z)-y
        #z' = x*y - beta*z
        #sigma = 10, rho = 28 beta=8/3, 
        
        xp1 = self.p[0]*(X[:,1] - X[:,0])
        xp2 = X[:,0]*(self.p[1] - X[:,2]) - X[:,1]
        xp3 = X[:,0]*X[:,1] - self.p[2]*X[:,2]
        
        yp = torch.cat((xp1[:,None], xp2[:,None], xp3[:,None]), dim=1)
        
        return yp
    
                
    def forward(self, x0):
        
        X = torch.zeros(x0.shape[0], 3, self.nt+1)
        X[:, :, 0] = x0
        for i in range(self.nt):
            Xi = X[:,:,i].clone() 
            k1 = self.yp_fun(Xi)
            k2 = self.yp_fun(Xi + self.dt/2*k1)
            k3 = self.yp_fun(Xi + self.dt/2*k2)
            k4 = self.yp_fun(Xi + self.dt*k3)
            Xp =  Xi + self.dt/6*(k1+2*k2+2*k3+k4)
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
fwd = dynamical_system(dt=dt, nt=Nt, p=p)

x0 = torch.randn(1,3) 
X = fwd(x0)

Xobs = X.detach()

#ax = phaseDiagram(X)

#print('h')

# Now estimate parameters from the data
# slice the function
# choose a random direction
# v = torch.randn(3)
# u = torch.randn(3)
# ns = 65
# f = torch.zeros(ns,ns)
# t = torch.linspace(-10,5, ns)
# for i in range(ns):
#     for j in range(ns):
#         with torch.no_grad():
#             fwd = dynamical_system(dt=dt, nt=Nt, p=p+t[i]*v+t[j]*u)
#             X = fwd(x0)
#             f[i,j] = F.mse_loss(X, Xobs)
#             print('%3d   %3d   %3.2e'%(i, j, f[i,j]))

# print('h')

Ntfit = 3 
fwd = dynamical_system(dt=dt, nt=Ntfit, p=torch.rand(3)*10) 
            
niterations = 50
opt = optim.SGD(fwd.parameters(), lr=1e0)
#batch_size = 100
torch.autograd.set_detect_anomaly(True)
for i in range(niterations):
    
    opt.zero_grad()
    # Compute the loss
    Xcomp = fwd(x0)
    loss = F.mse_loss(Xcomp, Xobs[:,:,:Ntfit+1])
    # Compute the gradient
    loss.backward()
    
    # Update
    opt.step()
    
    print('%3d   %3.2e'%(i, loss))

ax = phaseDiagram(Xobs)
ax = phaseDiagram(Xcomp)
    
print('h')


