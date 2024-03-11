import time

import torch
import pennylane as qml
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

dev = qml.device('default.mixed', wires=1)

H = qml.Hamiltonian(
    coeffs = [1],
    observables = [qml.PauliZ(0)]
)

H_1 = qml.Hamiltonian(
    coeffs = [1],
    observables = [qml.PauliY(0)]
)

# dephasing and post-selection parameters
tau_global = torch.tensor(0, dtype = torch.float, requires_grad = False)
gamma_ps_global = torch.tensor(0, dtype = torch.float, requires_grad = False)

# Rotation gate parameters
param_global = torch.tensor([0,0], dtype = torch.float, requires_grad = True)

# time evolution parameter
phi_global = torch.tensor(0, dtype = torch.float, requires_grad = True)


def Dephase_factor(tau):
    """ 
    Calculate the dephasing factor for a given dephasing time tau.

    Args:
        tau (torch.Tensor): Dephasing time.

    Returns:
        torch.Tensor: Dephasing factor.
    """  
    return 1 - torch.exp(-2 * tau)


@qml.compile
@qml.qnode(dev, interface = 'torch', diff_method = 'backprop')
def circuit(phi):
    """ 
    Construct a quantum circuit with specified gates and operations.

    Args:
        phi (torch.Tensor): Phase angle for the quantum gate.

    Returns:
        torch.Tensor: Density matrix of the quantum system after applying gates.
    """
    global param_global, tau_global
    theta_x = param_global[0]
    phi_z = param_global[1]
    
    gamma_dephase = Dephase_factor(tau_global)  

    qml.RX(torch.pi/2, wires = 0)

    qml.ApproxTimeEvolution(H, phi, 1)
    qml.PhaseDamping(gamma_dephase, wires = 0) 

    qml.RZ(phi_z, wires = 0)  # phi_z
    
    qml.RX(theta_x, wires = 0)  # theta_x
    
    return qml.density_matrix(wires = 0)

@qml.compile
@qml.qnode(dev, interface='torch', diff_method='backprop')
def post_selection(phi):
    """ 
    Perform post-selection on the output of the `circuit` function.

    Args:
        phi (torch.Tensor): Phase angle for the quantum gate.

    Returns:
        torch.Tensor: Post-selected density matrix after applying a Kraus operator.
    """
    global param_global, gamma_ps_global
    get_density_matrix = circuit(phi)
        
    # Kraus operator for 2*2 matrix
    K = torch.tensor([
        [torch.sqrt(1 - gamma_ps_global), 0],
        [0, 1]
    ], dtype=torch.complex128)

    Numerator = K @ get_density_matrix @ K.conj().T
    Denominator = torch.trace(Numerator)
    rho_ps = Numerator / Denominator

    qml.QubitDensityMatrix(rho_ps, wires = 0)
    
    return qml.density_matrix(wires = 0) 


def set_circuit(desired_tau_dephase, desired_gamma_post_selection):
    """
    Set the global dephasing rate and post-selection rate for the circuit.

    Args:
        desired_tau_dephase (float): Desired dephasing rate tau.
        desired_gamma_post_selection (float): Desired post-selection rate gamma.
    """
    global Tau_global, Gamma_ps_global 
    
    Tau_global = torch.tensor(desired_tau_dephase)
    Gamma_ps_global = torch.tensor([desired_gamma_post_selection])
    

def cost_function(paras):
    """ 
    Compute the cost using classical Fisher information for the given parameters.

    Args:
        paras (torch.Tensor): Parameters for quantum gates.

    Returns:
        torch.Tensor: Computed cost.
    """
    global param_global, phi_global
    param_global = paras

    CFI = qml.qinfo.classical_fisher(post_selection)(phi_global)
    return -CFI


def fit(sweep_range, initial_parameters):
    """
    Performs optimization with phi values starting from initialized parameters.

    Args:
        sweep_range (torch.Tensor): phi's spanning range. Formatted as [start, stop, step].
        initial_parameters (torch.Tensor): initial values of theta_x, phi_z.

    Returns:
        data (torch.Tensor): optimization tracking data.
    """

    phi = torch.arange(*sweep_range, dtype=torch.float32)

    data = torch.zeros((len(phi), len(initial_parameters) + 2))
    data[:, 0] = phi

    global param_global, phi_global

    params_tensor = initial_parameters.clone().requires_grad_(True)

    opt = torch.optim.Adam([params_tensor])

    def closure():
        opt.zero_grad()
        loss = cost_function(params_tensor)
        loss.backward()
        return loss
    
    f_logs = [cost_function(params_tensor).item()]

    # variables for early stopping
    steps = 500
    ftol = 1e-9
    patience = 0

    lval = 0.

    for phi_idx in tqdm(range(len(phi))):

        phi_global = phi[phi_idx].clone().requires_grad_(True)

        patience = 0
        for i in range(steps):

            lval = opt.step(closure).item()
            f_logs.append(lval)
            if i:
                if np.abs(f_logs[i] - f_logs[i-1] / f_logs[i]) < ftol:
                    patience += 1
                else: patience = 0

            if patience > 5: break
            
        data[phi_idx, 1] = -lval
        data[phi_idx, 2:] = params_tensor

    return data


def optimization_by_tau(sweep_range, init_par, tau_dephase, gamma_post_selection):
    """ 
    Iterate over different values of tau_dephase and gamma_post_selection for optimization.

    Args:
        sweep_range (list): Range of phi values for optimization.
        init_par (torch.Tensor): Initial parameters for optimization.
        tau_dephase (list): List of dephasing rates tau to iterate over.
        gamma_post_selection (float): Post-selection rate gamma.
        method (str): Optimization method.

    Returns:
        np.ndarray: Numpy array with optimization results for each tau.
    """
    for tau_idx, tau_current in enumerate(tau_dephase):
        set_circuit(tau_current, gamma_post_selection)
        
        temp = fit(sweep_range, init_par).detach().cpu().numpy()
        if tau_idx == 0:
            data = np.zeros((len(tau_dephase), len(temp[:,0]), len(temp[0,:])))
            data[tau_idx][:, :] = temp
        else:
            data[tau_idx][:, :] = temp
            
    return data


sweep_range = torch.tensor([1e-2, 2*torch.pi, 1e-2], dtype=torch.float, requires_grad=False)
init_par = torch.tensor([
    # theta_x
    torch.pi/2, 
    
    # phi_z
    torch.pi/2
    ], dtype=torch.float)

tau_dephase = 0.,
gamma_ps = 0.5

start_time = time.time()
res = optimization_by_tau(sweep_range, init_par, tau_dephase, gamma_ps)
end_time = time.time()

running_time = (end_time - start_time) / 60



plt.plot(res[0][:,0], res[0][:,1], label=f'{tau_dephase[0]}')

plt.show()

np.save(f'./opt_result.npy', res)

