# Import Packages

import matplotlib.pyplot as plt
import numpy as np
from qutip import *
from qutip.qip.circuit import *

import torch

# Global parameters
tau_global = torch.tensor(0, dtype=torch.float, requires_grad=False)
gamma_ps_global = torch.tensor(0, dtype=torch.float, requires_grad=False)
param_global = torch.tensor([0,0], dtype=torch.float, requires_grad=True)
phi_global = torch.tensor(0, dtype=torch.float, requires_grad=True)

# Defining hamiltonian
H = -0.5*sigmaz()

def dephase_factor(tau):
    """ 
    Calculate the dephasing factor for a given dephasing time tau.

    Args:
        tau (torch.Tensor): Dephasing time.

    Returns:
        torch.Tensor: Dephasing factor.
    """  
    return 1 - torch.exp(-2*tau)

def quantumcircuit():
    pass


def post_selection(phi):
    """ 
    Perform post-selection on the output of the `circuit` function.

    Args:
        phi (torch.Tensor): Phase angle for the quantum gate.

    Returns:
        torch.Tensor: Post-selected density matrix after applying a Kraus operator.
    """
    global param_global, gamma_ps_global
