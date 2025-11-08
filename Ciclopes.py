import numpy as np
import pandas as pd
import torch
from torch import nn

class Ciclopes(nn.Module):
    
    def __init__(self, Ecg, Nf=1, coeffs=None, coeffs_init=None, device='cuda', zero=1e-6):
        super(Ciclopes, self).__init__()
        self.device = device
        self.Ecg = Ecg
        self.Ng = self.Ecg.shape[1]
        self.Nf = Nf
        self.zero = torch.tensor(zero, dtype=torch.float32, device=self.device)

        if coeffs is not None:
            Creal, Cimag = coeffs
            self.Creal = torch.tensor(Creal, dtype=torch.float32, device=self.device)
            self.Cimag = torch.tensor(Cimag, dtype=torch.float32, device=self.device)
        else:
            if coeffs_init is not None:
                Creal_init, Cimag_init = coeffs_init
                self.Creal = nn.Parameter(torch.tensor(Creal_init, dtype=torch.float32, device=self.device), requires_grad=True)
                self.Cimag = nn.Parameter(torch.tensor(Cimag_init, dtype=torch.float32, device=self.device), requires_grad=True)
            else:
                self.Creal = nn.Parameter(torch.randn(size=(self.Ng, self.Nf),  dtype=torch.float32, device=self.device), requires_grad=True)
                self.Cimag = nn.Parameter(torch.randn(size=(self.Ng, self.Nf),  dtype=torch.float32, device=self.device), requires_grad=True)
        
        self.h1_dim = int(0.75 * self.Ng)
        self.h2_dim = int(0.50 * self.Ng)
        self.h3_dim = int(0.25 * self.Ng)
        
        self.Encoder =  nn.Sequential(
                        nn.Linear(self.Ng, self.h1_dim),
                        nn.LeakyReLU(0.1),
                        nn.Linear(self.h1_dim, self.h2_dim),
                        nn.LeakyReLU(0.1),
                        nn.Linear(self.h2_dim, self.h3_dim),
                        nn.LeakyReLU(0.1),
                        nn.Linear(self.h3_dim, 2)
                        )
                
    def Encode(self, X):
        Z = self.Encoder(X) + self.zero
        Z = Z / Z.norm(dim=-1, keepdim=True).clamp_min(1e-3)
        return Z
    
    
    def FourierBasis(self, theta):
        m = torch.arange(-self.Nf, self.Nf + 1, dtype=torch.float32, device=self.device)
        exponent = 2 * 1j * torch.pi * m[:, None] * theta[None, :]  # shape: (theta, 2*self.Nf+1), dtype=complex64
        Fb = torch.exp(exponent)                                    # shape: (theta, 2*self.Nf+1), dtype=complex64
        return Fb
    
    
    def GEXmodel(self, theta):
        Cplus = torch.complex(self.Creal, self.Cimag) 
        Cminus = torch.complex(self.Creal, -self.Cimag).flip(dims=[1])
        C0 = torch.zeros(size=(self.Ng,1), dtype=torch.complex64, device=self.device)
        Cn = torch.cat([Cplus, C0, Cminus], dim=1)
        Fb = self.FourierBasis(theta)
        return (Cn @ Fb).real
    
    
    def Forward(self, X): 
        Z =  self.Encode(X)
        theta = (torch.pi + torch.atan2(Z[:,1] + self.zero,  Z[:,0] + self.zero)) / (2 * torch.pi)
        Xm = self.GEXmodel(theta)
        return theta, Xm
    
    
    def Loss(self, X, Xm):
        Nc = X.shape[0]
        Ng = X.shape[1]
        #Rg = ((X - Xm.T)**2).sum(dim=0)
        Xb  = X  - X.mean(dim=0, keepdim=True)               # per-gene batch centering
        Xmb = Xm.T - Xm.T.mean(dim=0, keepdim=True)
        Rg  = ((Xb - Xmb)**2).sum(dim=0)        
        term1 = 0.5 * Nc * (torch.log(Rg + self.zero).sum())
        term2 = (Nc + Ng) * torch.log(torch.tensor(1.0 / (np.sqrt(2*np.pi)), dtype=torch.float32, device=self.device))
        term3 = (0.5 * Nc - 1) * Ng * torch.log(torch.tensor(2.0, dtype=torch.float32, device=self.device))
        term4 = Ng * torch.lgamma(torch.tensor(0.5 * Nc, dtype=torch.float32, device=self.device))
        return term1 - (term2 + term3 + term4)     
