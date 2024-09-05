import scipy.sparse.linalg
import QAOA_numpy_lib
import time

import scipy.optimize
import scipy.interpolate

import multiprocessing as mp
import numpy as np
import os
import sys


######### CRAB functions ############

#Define the basis of L2 integrable function  
def func_basis(pulses, Nc, P, BASIS= 'FOU'):

    '''
    Parameters
    ----------
    pulses : Nc-dimensional array of floats
        Real numbers (typically integers + random perturbations) labelling the element of the functional basis
    Nc : int
        Number of functional coefficients.
    P : int
        Number of Trotter steps.
    BASIS : String, optional
        Functional basis. For the moment only two options available: FOU (fourier modes) and CHB (chebyschev polynomials). The default is 'FOU'.

    Returns
    -------
    fi_t : (Nc x P)-dimensional array
        Functional basis element as a function of discrete time.

    '''
    fi_t = np.zeros((P, Nc))

    if BASIS == 'FOU':
        for m in range(P):
            fi_t[m] = [np.sin(np.pi*pulses[n]*(m+ 0.5)/P) for n in range(Nc)]        
    elif BASIS == 'CHB':   
        for m in range(P):
            fi_t[m] = [np.cos(pulses[n]*np.arccos((m+0.5)/P)) for n in range(Nc)]
    else : print('Functional basis not available yet')
    return fi_t
 
    

#Returns the driving fields given pulses and functional basis
def amplitudes(C0_x, C0_z, Ci_x, Ci_z, dressing_x, dressing_z, fi_t): 
    '''
    Parameters
    ----------
    C0_x : Float
        In the limit where all the functional coefficients are zero, it coincides with the discrete time step
    C0_z : Float
        In the limit where all the functional coefficients are zero, it coincides with the discrete time step
    Ci_z : Nc-dimensional array of floats
        Functional coefficients of theta_z
    Ci_x : Nc-dimensional array of floats
        Functional coefficients of theta_x.
    fi_t : (Nc x P)-dimensional array
        Functional basis element as a function of discrete time.

    Returns
    -------
    theta_z : P-dimensional array of floats
        Theta_z(m) .
    theta_x : P-dimensional array of floats
        Theta_x(m).
    s : P-dimensional array of floats
        theta_z(m) / (theta_z(m) + theta_x(m))

    '''
    P = fi_t.shape[0]
    
    theta_z, theta_x = np.zeros(P), np.zeros(P)
    for m in range(P):
        theta_z[m] =  dressing_z[m]*C0_z + (m+0.5)/P*np.einsum('i->' , Ci_z*fi_t[m])
        theta_x[m]  = dressing_x[m]*C0_x + (1-(m+0.5)/P)*np.einsum('i->' , Ci_x*fi_t[m])
        
    s = theta_z/(theta_z+theta_x)
    return theta_z, theta_x, s




# Cost function - to be optimized with gradient free algorithm (first or second order Trotter)
def fig_of_merit(theta_i, qaoa_cirq, dressing_x, dressing_z, pulses, fi_t, dt, psi_0, order):
    '''
    
    Parameters
    ----------
    theta_i : (2Nc+2)-dimensional array
        Array of functional coefficients in the form: theta_i[0] = C_z,  theta_i[1] = C_x  , theta_i[2::2] = Ci_z, theta_i[3::2] = Ci_x.
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    pulses : Nc-dimensional array of floats
        Real numbers (typically integers + random perturbations) labelling the element of the functional basis
    fi_t : (Nc x P)-dimensional array
        Functional basis element as a function of discrete time.
    dt : float
        Time scale - in a digital circuit dt = 1.
    psi_0 : 2^(N-1)-dimensional array 
        Initial state of the digital evolution.
    order : 1,2
        Order of the Trotter decomposition. Currently only 1st and 2nd order are available

    Returns
    -------
    efinal : float
        Final expectation value of H_target.

    '''
    P = fi_t.shape[0]
    
    C0_z = theta_i[0]
    C0_x = theta_i[1]

    Ci_z, Ci_x =  theta_i[2::2], theta_i[3::2]
    
    theta_z, theta_x, s_t =  amplitudes(C0_x, C0_z, Ci_x, Ci_z, dressing_x, dressing_z, fi_t)
    
    psi_final = psi_0
      
    if order == 1:
        for m in range(P):
            psi_final = qaoa_cirq.apply_trotter_I(theta_z[m], theta_x[m], dt, psi_final)
    
    elif order == 2:
        for m in range(P):
            psi_final = qaoa_cirq.apply_sym_trotter_II(theta_z[m], theta_x[m], dt, psi_final)


    efinal = qaoa_cirq.energy_expect(psi_final,1,0)
    
    return efinal
    
    
#Returns the final state evaluated using trotterized dynamics (first or second order Trotter)
def digital_state(qaoa_cirq, theta_z, theta_x, dt, psi_0, order):
    '''
    
    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    gamma_vec :P-dimensional array of floats
        Theta_z(m) .
    beta_vec : P-dimensional array of floats
        Theta_x(m) .
    dt : float
        Time scale - tipically of order 1.
    psi_0 : 2^(N-1)-dimensional array 
        Initial state of the digital evolution.
    order : 1,2
        Order of the Trotter decomposition. Currently only 1st and 2nd order are available

    Returns
    -------
    psi_qaoa :  2^(N-1)-dimensional array 
        Final state.

    '''
    P = theta_z.shape[0]
    psi_qaoa = psi_0
    
    if order ==1:
        for m in range(P):
            psi_qaoa = qaoa_cirq.apply_trotter_I(theta_z[m], theta_x[m], dt, psi_qaoa)
    elif order == 2:
        for m in range(P):
            psi_qaoa = qaoa_cirq.apply_sym_trotter_II(theta_z[m], theta_x[m], dt, psi_qaoa)

    return psi_qaoa
 

#Cost function and its gradient respect to theta_i - to be optimized with gradient based algorithm (first and second order Trotter)

def energy_with_gradient(theta_i, qaoa_cirq, dressing_x, dressing_z, pulses, fi_t, dt, psi_0, order):
    '''
    
    Parameters
    ----------
    theta_i : (2Nc+2)-dimensional array
        Array of functional coefficients in the form: theta_i[0] = delta_t , theta_i[1::2] = gamma_i, theta_i[2::2] = beta_i.
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    pulses : Nc-dimensional array of floats
        Real numbers (typically integers + random perturbations) labelling the element of the functional basis
    fi_t : (Nc x P)-dimensional array
        Functional basis element as a function of discrete time.
    dt : float
        Time scale - tipically of order 1.
    psi_0 : 2^(N-1)-dimensional array 
        Initial state of the digital evolution.
    order : 1,2
        Order of the Trotter decomposition. Currently only 1st and 2nd order are available

    Returns
    -------
    efinal : float
        Final expectation value of H_target.
    grad_theta_i : (2Nc+2)-dimensional array
        Gradient with respect to theta_i coefficients.

    '''
    C0_z = theta_i[0]
    C0_x = theta_i[1]

    Ci_z, Ci_x =  theta_i[2::2], theta_i[3::2]

    theta_z, theta_x, s_t =  amplitudes(C0_x, C0_z, Ci_x, Ci_z, dressing_x, dressing_z, fi_t)
    
    P = fi_t.shape[0]
    Nc = len(Ci_z)   
    
    ######################################################################
    
    psi_m =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')
    psi_m[0] = psi_0
    
    if order == 1 :
        for m in range(P):
            psi_m[m+1] = qaoa_cirq.apply_trotter_I(theta_z[m], theta_x[m], dt, psi_m[m])
            
    elif order == 2:
        for m in range(P):
            psi_m[m+1] = qaoa_cirq.apply_sym_trotter_II(theta_z[m], theta_x[m], dt, psi_m[m])
        

    psi_final = psi_m[-1]
    
    ########################################################################
    
    if order == 1:
        phi_z = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
        for m in range(P):
            phi_z[m]  = qaoa_cirq.apply_Uz(psi_m[m], theta_z[m]*dt)
    
    elif order == 2:
        phi_z, phi_xz = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex'), np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
        for m in range(P):
            phi_z[m]  = qaoa_cirq.apply_Uz(psi_m[m], theta_z[m]*dt/2)
            phi_xz[m] = qaoa_cirq.apply_Ux(phi_z[m],  theta_x[m]*dt)
    
  
    ################################################################
    
    backwards_prop =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')

    hz_psi_f = qaoa_cirq.apply_Hz(psi_final, 1)
    
    backwards_prop[0] = hz_psi_f
    
    
    if order == 1 :
        for m in range(P):
            backwards_prop[m+1] = qaoa_cirq.apply_trotter_I(np.flip(theta_z)[m], np.flip(theta_x)[m], dt, backwards_prop[m], herm = True)
    
    elif order == 2:
         for m in range(P):
             backwards_prop[m+1] = qaoa_cirq.apply_sym_trotter_II(np.flip(theta_z)[m], np.flip(theta_x)[m], -dt, backwards_prop[m])
       
    
    d_dtheta_x, d_dtheta_z = np.zeros(P, dtype = 'complex'), np.zeros(P, dtype = 'complex')
    chain_rule_z, chain_rule_x = np.zeros((P, Nc), dtype = 'complex'), np.zeros((P, Nc), dtype = 'complex')
    chain_rule_0x, chain_rule_0z  = np.zeros(P, dtype = 'complex'), np.zeros(P, dtype = 'complex'),
    
    
    if order == 1: 
        chi_x = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
        for m in range(P):
            chi_x [m] = qaoa_cirq.apply_Ux(backwards_prop[m], -np.flip(theta_x)[m]*dt) 

            
        chi_x = np.flip(chi_x, axis = 0)    

        for m in range(P):
            d_dtheta_x[m]  = np.dot(np.conjugate(chi_x[m]), -1j*dt*qaoa_cirq.apply_Hx(phi_z[m],1))
            d_dtheta_z[m] = np.dot(np.conjugate(chi_x[m]), -1j*dt*qaoa_cirq.apply_Hz(phi_z[m],1))
    
            chain_rule_0x[m] = dressing_x[m]*2*np.real(d_dtheta_x[m]) 
            chain_rule_0z[m] = dressing_z[m]*2*np.real(d_dtheta_z[m])
    
            
            chain_rule_x[m] = (1-(m+0.5)/P)*fi_t[m]*2*np.real(d_dtheta_x[m])
            chain_rule_z[m] = (m+0.5)/P*fi_t[m]*2*np.real(d_dtheta_z[m])


    elif order == 2:
         chi_z, chi_xz = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex'), np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
      
         for m in range(P):
             chi_xz[m] = qaoa_cirq.apply_Uz(backwards_prop[m], - np.flip(theta_z)[m]*dt/2)
             chi_xz[m] = qaoa_cirq.apply_Ux(chi_xz[m], - np.flip(theta_x)[m]*dt)
          

         chi_xz = np.flip(chi_xz, axis = 0)    
         for m in range(P):
             
             chi_z [m] = qaoa_cirq.apply_Ux(chi_xz[m], theta_x[m]*dt) 
             d_dtheta_x[m]  = np.dot(np.conjugate(chi_xz[m]), -1j*dt*qaoa_cirq.apply_Hx(phi_z[m],1))
             d_dtheta_z[m] = np.dot(np.conjugate(chi_xz[m]), -1j*dt/2*qaoa_cirq.apply_Hz(phi_z[m],1)) + np.dot(np.conjugate(chi_z[m]), -1j*dt/2*qaoa_cirq.apply_Hz(phi_xz[m],1)) 
             
             chain_rule_0x[m] = (1-(m+0.5)/P)*2*np.real(d_dtheta_x[m]) 
             chain_rule_0z[m] = (m+0.5)/P*2*np.real(d_dtheta_z[m])

             
             chain_rule_x[m] = (1-(m+0.5)/P)*fi_t[m]*2*np.real(d_dtheta_x[m])
             chain_rule_z[m] = (m+0.5)/P*fi_t[m]*2*np.real(d_dtheta_z[m])
   
    
    ######################################################################
    grad_x_i  = np.einsum('ij->i' , chain_rule_x.T)        
    grad_z_i = np.einsum('ij->i' , chain_rule_z.T) 
    grad_0x = np.einsum('i->' , chain_rule_0x) 
    grad_0z = np.einsum('i->' , chain_rule_0z) 

    

    grad_theta_i = np.zeros(2*Nc+2)
 
    grad_theta_i[0] = np.real(grad_0z)
    grad_theta_i[1] = np.real(grad_0x)

    grad_theta_i[2::2], grad_theta_i[3::2] = np.real(grad_z_i), np.real(grad_x_i)

    efinal = qaoa_cirq.energy_expect(psi_m[-1],1,0)

    return efinal, grad_theta_i   



#Performs the optimization
def optimize_pulse(fig_of_merit, theta_i, args, ftol, method, Emin, Emax, ORDER, HESS = False, callback = None, verbose = True):
    '''
    

    Parameters
    ----------
    fig_of_merit : Scalar Function
        Loss function.
    theta_i : (2Nc+2)-dimensional array
        Array of functional coefficients in the form: theta_i[0] = delta_t , theta_i[1::2] = gamma_i, theta_i[2::2] = beta_i.
    args : list
        Additional arguments of the cost function.
    ftol : float
        The optimization stops when the variation of the cost function between two consecutive iterations is smaller than this threshold.
    method : string
        Optimization methods. Three of them are availble: Nelder Mead (gradient free optimization), BFGS-numerical (gradient based method: numerical approximation of the derivative) BFGS-analytical (gradient based method: analytical calculation of the gradient) 
        REMARK: if BFGS-analytical is used, the cost function is automatically set to be energy_with_gradient(*args)
    Emin : Float
        Minimum energy of Hz.
    Emax : Float
        Maximum energy of Hz.
    ORDER :  1,2
        Order of the Trotter decomposition. Currently only 1st and 2nd order are available
    HESS : Bool, optional
        If True, the inverse Hessian of the cost function evaluated in the optimal point is evaluated. The default is False.
    verbose : Bool, optional
        If True, extensive details about the optimization will be printed. The default is True.

    Returns
    ----------
    eres: Float
        Final residual energy
    
    theta_opt: (2Nc+2)-dimensional array
        Optimal functional coefficients
 
    '''
    
#   bounds = ((-1,1), (-1,1))*(int(len(theta_i)/2))
    bounds = None
    
    if method == 'Nelder-Mead':
        jac = False
        print('Gradient free optimization starts...')
        res = scipy.optimize.minimize(fig_of_merit, theta_i, args=args, method='Nelder-Mead', bounds = bounds, tol = ftol, callback = callback)

    
    elif method == 'BFGS-numerical':
        jac = False
        print('Gradient based optimization starts...')
        res = scipy.optimize.minimize(fig_of_merit, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac, callback = callback)
    


    elif method == 'BFGS-analytical':
        jac = True
        print('Exact gradient based optimization starts...')
        res = scipy.optimize.minimize(energy_with_gradient, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac, callback = callback)
    
    eres = (res.fun-Emin)/(Emax - Emin)
    theta_opt = res.x
    print(res.message)
    

    if verbose:
        print('Number of iterations is ', res.nit)
        print('Number of calls of the cost function is ', res.nfev)
        
    if HESS:
        hess = res.hess_inv
        return eres, theta_opt, hess
    
    else: return eres, theta_opt




def write_output_file(output_dir, tau, Nc, n_reals, dt, 
                      C0_z_start, C0_x_start, Cz_i_start, Cx_i_start,
                      pulses_opt, 
                      C0_z_opt, C0_x_opt, Cz_i_opt, Cx_i_opt, 
                      theta_z_opt, theta_x_opt, 
                      eres_opt, t_gba, basis, Warm_start, RAND, NOISE, MP):
    '''
    

    Parameters
    ----------
    output_dir : string
        Output directory.
    tau : Float
        Total time of the evolution. If dt=1, coincides with P
    Nc : Int
        Number of functional coefficients.
    n_reals : Int
        If RAND flag is on, is the number of random perturbation of the functional basis.
    dt : float
        Time scale - tipically of order 1.
    C0_z_start : Float
        Initial value of C0_z
    C0_x_start : Float
        Initial value of C0_x
    Cz_i_start : Nc-dimensional array
        Functional coefficients of theta_z used as initial point for the optimization.
    Cx_i_start : Nc-dimensional array
        Functional coefficients of theta_x used as initial point for the optimization.
    pulses_opt :  Nc-dimensional array of floats
        Optimal values of the pulses. It defines the optimized functional basis
    C0_z_opt : Float
        Optimal value of C0_z
    C0_x_opt : Float
        Optimal value of C0_x

    Cz_i_opt : Nc-dimensional array
        Optimal functional coefficients of theta_z.
    Cx_i_opt : Nc-dimensional array
        Optimal functional coefficients of theta_x.
    theta_z_opt :  P-dimensional array
        Optimal theta_z(m)
    theta_x_opt : P-dimensional array
        Optimal theta_x(m)
    eres_opt : Float
        Optimal residual energy.
    t_gba : float
        Elapsed time for the optimization.
    basis : string
        Functional basis.
    Warm_start : Bool
        True if Warm_start procedure was used.
    RAND : Bool
        True if randomization of the functional basis was used.

    Returns
    -------
    Saves the output file.

    '''
    
    if RAND:
        if Warm_start:
            filename_output = output_dir+'/Warm_start_'+NOISE+'_'+basis+'_OUTPUT_file_tau_'+str(tau)+"_Nc"+str(Nc)+"_nreals_"+str(n_reals)+".dat"
        else:  
            filename_output = output_dir+'/'+NOISE+'_'+basis+'_OUTPUT_file_tau_'+str(tau)+"_Nc"+str(Nc)+"_nreals_"+str(n_reals)+".dat"


    else: 
        if Warm_start:
            filename_output = output_dir+'/Warm_start_'+basis+'_OUTPUT_file_tau_'+str(tau)+"_Nc"+str(Nc)+".dat"
        else: 
            filename_output = output_dir+'/'+basis+'_OUTPUT_file_tau_'+str(tau)+"_Nc"+str(Nc)+".dat"
    
    if MP: 
        name_split = filename_output.split('/')
        name_split[-1] = 'MP_'+name_split[-1] 
        filename_output = '/'.join(name_split)


        
    P = int(tau/dt)
    t_list = np.linspace(0,tau-dt,P) 

    left = np.vstack((t_list, theta_x_opt, theta_z_opt)).T
    right = np.vstack((Cx_i_start, Cz_i_start, pulses_opt, Cx_i_opt, Cz_i_opt)).T
    czeros= np.vstack((C0_x_start , C0_z_start, C0_x_opt, C0_z_opt)).T

    file_out = open(filename_output,'w')


    file_out.write("### PARAMETERS OF THE TIME EVOLUTION ### \n")
    
    file_out.write("Total time tau = " + str(tau)+"\nTimescale dt = "+str(dt)+"\nP = tau/dt = "+str(P)+"\n")
    file_out.write("Number of Fourier coefficients Nc = "+str(Nc)+"\n")
    
    file_out.write("\nCRAB coefficient C0x_0 // C0z_0 // C0x_opt // C0z_opt \n")
    np.savetxt(file_out, czeros)

    
    file_out.write("\nCRAB coefficient Cx_i_start // Cz_i_start // pulses_i // Cx_i_opt // Cz_i_opt  \n")
    np.savetxt(file_out, right)
    
    
    file_out.write("\nAmplitudes vs time: t // theta_x_opt // theta_z_opt  \n")
    np.savetxt(file_out, left)
    
    file_out.write("\nResidual energy after optimization is "+str(eres_opt))
    file_out.write('\nOptimization procedure took '+str(t_gba) +" seconds")
    
    
    file_out.close()


# Reads an input file (produced by this library) and returns the relevant parameters
def read_input_file(filename):
    '''
    

    Parameters
    ----------
    filename : string
        Name of the input file.

    Returns
    -------
    P : Int
        Number of Trotter steps.
    Nc : Int
        Number of functional coefficients.
    gamma_opt :  P-dimensional array
        Optimal theta_z(m)
    beta_opt : P-dimensional array
        Optimal theta_x(m)
    pulses_i : Nc-dimensional array of floats
        Optimal values of the pulses. It defines the optimized functional basis
    gamma_i_opt : Nc-dimensional array
        Optimal functional coefficients of theta_z.
    beta_i_opt : Nc-dimensional array
        Optimal functional coefficients of theta_x.
    C0 : float
        delta_t coefficient.
    eres : float
        Optimal residual energy.

    '''
    file_crab = open(filename, 'r')
    file_lines = file_crab.readlines()

    P   = int(file_lines[3].split()[-1])
    Nc  = int(file_lines[4].split()[-1])

    
    
    pulses_i    = np.zeros(Nc,dtype='float')
    Cx0_opt, Cz0_opt = float(file_lines[7].split()[-2]), float(file_lines[7].split()[-1]) 
    
    Cx_i_opt  = np.zeros(Nc,dtype='float')
    Cz_i_opt = np.zeros(Nc,dtype='float')


    for m in range(Nc):
        [pulses_i[m], Cx_i_opt[m], Cz_i_opt[m]] = file_lines[10:10+Nc][m][:-1].split()[2::]

    
    theta_x_opt , theta_z_opt  = np.zeros(P ,dtype='float') , np.zeros(P ,dtype='float')

    for m in range(P):
        [theta_x_opt[m], theta_z_opt[m]] = file_lines[10+Nc+2:10+Nc+2+P][m][:-1].split()[1::]

    for m in range(P):
        while theta_x_opt[m] > np.pi/2: theta_x_opt[m] = theta_x_opt[m] - np.pi/2
        while theta_x_opt[m] < 0: theta_x_opt[m] = theta_x_opt[m] + np.pi/2
    
    eres = float(file_lines[-2].split()[-1])
    return P, Nc, theta_z_opt, theta_x_opt, pulses_i, Cx0_opt, Cz0_opt, Cx_i_opt, Cz_i_opt, eres





def random_pulse_generation(pulses, var, NOISE, Warm_start, Warm_from_file, Nc,  Nc0 = None , Nc_prev = None):
    if NOISE == 'ADD':        
        if Warm_start:
            if Nc == Nc0 and not Warm_from_file: 
                random_pulses = np.array([pulses[n]+var*(np.random.rand() - 0.5 )  for n in range(Nc)] ) 
            else:
                random_pulses = np.copy(pulses)
                random_pulses[Nc_prev::] = np.array( [pulses[n]+var*(np.random.rand() - 0.5 )  for n in range(Nc_prev, Nc)] )   
        
        else: random_pulses = np.array( [pulses[n]+var*(np.random.rand() - 0.5 )  for n in range(Nc)] ) 
  

    elif NOISE == 'MULT':
        if Warm_start:
            if Nc == Nc0 and not Warm_from_file: 
                random_pulses = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc)] ) 
            
            else: 
                random_pulses = np.copy(pulses)
                random_pulses[Nc_prev:] = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc_prev, Nc)] )   
        
        else: random_pulses = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc)] ) 
    
    
    elif NOISE == 'GAMMA':
        
        shape, scale = 2, 2   # This quantities has to be tuned...open point 
#       shape, scale = 1.5, 4 # This quantities has to be tuned...open point 
        
        if Warm_start:
            if Nc == Nc0 and not Warm_from_file: 
                random_pulses = np.random.gamma(shape, scale, Nc)
            
            else: 
                random_pulses = np.copy(pulses)
                random_pulses[Nc_prev:] = np.random.gamma(shape, scale, Nc-Nc_prev)   
        
        else: random_pulses =  np.random.gamma(shape, scale, Nc) 


    else: 
        random_pulses = None
        print('Specify whether additional or multiplicaitive random noise ')
    
    
    return random_pulses


# Clean the code by adding a dictionary or the args and modify locally 
def single_random_realization(random_pulses, qaoa_cirq, theta_start, ftol, Nc, P, dressing_x, dressing_z, basis, dt, psi_0, Emin, Emax, HESS, ORDER, opt_method, callback = None):
    fi_t = func_basis(random_pulses, Nc, P, basis)
    args= (qaoa_cirq, dressing_x, dressing_z, random_pulses, fi_t, dt, psi_0, ORDER)

    if HESS:
        eres_j, theta_j, hess_j= optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, HESS = HESS, callback = callback, verbose = False)
        return eres_j, theta_j, hess_j
    else:
        eres_j, theta_j = optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, HESS = HESS, callback = callback,  verbose = False)
#        print('Residual energy for this realization is ' , eres_j)
        return eres_j, theta_j
        
    
    
def worker(i, pulses_list, output, args):
    pulses_i = pulses_list[i]
    result_eres, result_theta = single_random_realization(pulses_i, *args)
    output.put((i, result_eres, result_theta))
    
    
    

def CRAB_main(qaoa_cirq, output_dir, figures_dir, tau, dt, basis, Ncs, 
              n_reals = None, Full = False, RAND = True, ORDER = 1, 
              Warm_start = False, Warm_from_file = False, input_file = None, 
              opt_method = 'BFGS-analytical', MP = False, HESS = False, NOISE = 'MULT', 
              DRESS = False, theta_z_old = None, theta_x_old = None):
    '''
    
    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    output_dir : string
        Output directory.
    figures_dir : string
        Figures directory.
    tau : float
        Total time of evolution - will be trotterized! tau = dt*P.
    dt : float
        Time scale. Tipically of order 1.
    basis : String
        Functional basis = 'FOU' for Fourier modes or 'CHB' for Chebyschev polynomials.
    Ncs : Array of integers
        Number of functional coefficients (it only matters if Full = False).
    n_reals : integer, optional
        If RAND = True, number of realizations over different sets of Ncs random frequencies. The default is None.
    Full : Bool, optional
        If True, Nc = P/2 regardless the value of Ncs. The default is False.
    RAND : Bool, optional
        If True, a random functional basis if used. The default is True.
    ORDER : 1,2, optional
        Order of the Trotter decomposition. Only 1st and 2nd orders are available. The default is 1.
    Warm_start : Bool, optional
        If True, the initial condition for the j-th element in Ncs is the optimal result for the (j-1)-th element of Ncs. The default is False.
    Warm_from_file : Bool, optional
        The initial condition for the first Nc in the list is extracted from an input file. Afterwards, the regular Warm_start procedure starts. The default is False.
    input_file : string, optional
        If Warm_from_file is True, it is the path of the input file. The default is None.
    opt_method : string, optional
        Optimization methods. Three of them are availble: Nelder Mead (gradient free optimization), BFGS-numerical (gradient based on numerical approximation of the derivative) BFGS-analytical (gradient based on analytical calculation of the gradient) 
        REMARK: if BFGS-analytical is used, the cost function is automatically set to be energy_with_gradient(*args)
    HESS : Bool, optional
        If True, the inverse Hessian of the cost functiob evaluated in the optimal points is saved. The default is False.
    NOISE : string, optional
        Nature of the noise to be introduced in the functional basis. Available option are 'MULT' or 'ADD' for respectively multiplicative or additive noise. The default is 'MULT'.

    Returns
    -------
    Performs the CRAB optimization of the selected MaxCut instance using the selected options.

    '''
 
    
    crab_dir = output_dir+"/CRAB/" # setting the output folder for CRAB parameters
    if not os.path.exists(crab_dir):os.makedirs(crab_dir)

    if Warm_from_file and not Warm_start : 
        print("In order to use Warm_from_file, Warm_start option should be active. Process aborted.")
        sys.exit()
        
    if Warm_start and Full:
        print("Full and Warm_start options cannot co-exist as they are contradictory. Process aborted.")
        sys.exit()

    
    # Time discretization 

    P = int(tau/dt)

    # Circuit parameters 
    Emin, Emax = qaoa_cirq.Emin, qaoa_cirq.Emax
    psi_0 = qaoa_cirq.psi_start


    #Optimization parameters
    ftol = 1e-6 # Tolerance of the optimization 
    
    if Full: Ncs = [P]
#    if Full: Ncs = [int(P/2)]
    

    dressing_x , dressing_z = np.zeros(P) , np.zeros(P)
    for m in range(P):
        dressing_x[m] , dressing_z[m] = (1-(m+0.5)/P), (m+0.5)/P 

    # Randomization parameters - set to False to have a ordered chopped functional basis
    if RAND:
        n_reals = n_reals # Number of realizations for each set of random frequencies 
        
        if basis == 'CHB' : var = 1  # Size of the interval of the random sampling 
        else : var = 1



    # Initialization of the optimal paramters (real time)
    theta_z_opt, theta_x_opt, s_opt = np.zeros(( len(Ncs) ,  P)), np.zeros(( len(Ncs) ,  P)) , np.zeros(( len(Ncs),  P))
    Nc0 = None

    if Warm_start:
        Nc0 = Ncs[0]
        if Warm_from_file:
            P_ff, Nc_ff, theta_z_ff, theta_x_ff, pulses_opt, Cx0_opt, Cz0_opt, Cx_i_opt, Cz_i_opt, eres_void = read_input_file(input_file)
        else: 
            Cx_i_opt, Cz_i_opt = [0]*(Ncs[0]), [0]*(Ncs[0])
            pulses_opt = np.arange(1,Ncs[0]+1)
            Cx0_opt , Cz0_opt = 1 , 1

    
    for count, Nc in enumerate(Ncs):
        print("Nc = ", Nc)
        
        if DRESS: 
            dressing_x[m] , dressing_z[m] = theta_x_old[count, m], theta_z_old[count, m]

        
        #Initial guess for the functional coefficients
        Cx_i_start, Cz_i_start = [0]*(Nc), [0]*(Nc)

        #Initial guess for the optimal delta_t
        Cx0_start , Cz0_start = 1 , 1

        Nc_prev = None
        if Warm_start:    
            Nc_prev = len(Cz_i_opt)
            Cx_i_start[0:Nc_prev], Cz_i_start[0:Nc_prev] = Cx_i_opt, Cz_i_opt
            Cx_i_start[Nc_prev::], Cz_i_start[Nc_prev::] = [0]*(Nc-Nc_prev), [0]*(Nc-Nc_prev)
            Cx0_start , Cz0_start =  Cx0_opt, Cz0_opt
        
        theta_start = np.zeros(2*Nc+2)
        
        theta_start[0] ,theta_start[1] = Cz0_start , Cx0_start
        theta_start[2::2], theta_start[3::2] = Cz_i_start, Cx_i_start

    
        start_time = time.time()
    
        pulses = np.arange(1,Nc+1, dtype = float)
        if not Full and Warm_start: 
            pulses[0:Nc_prev] = pulses_opt
 
            
        # RAND = TRUE
        if RAND:
            reals_eres, reals_theta, reals_pulses = np.zeros(n_reals), np.zeros((n_reals, 2*Nc+2)), np.zeros((n_reals, Nc))
            if HESS:
                reals_hess = np.zeros((n_reals, 2*Nc+2, 2*Nc+2 ))
               
            
               
            # Start multiprocessing
            if MP: 
                args = (qaoa_cirq, theta_start, ftol, Nc, P, dressing_x, dressing_z, basis, dt, psi_0, Emin, Emax, HESS, ORDER, opt_method)
                print('MP starts!')
                # First we generate all the random pulses involved in the calculation
                for j in range(n_reals):
                    reals_pulses[j] = random_pulse_generation(pulses, var, NOISE, Warm_start, Warm_from_file, Nc,  Nc0 , Nc_prev)

                # Create a Queue to store the results
                output = mp.Queue(n_reals)
                
                # Create a list to keep track of processes
                processes = []
                print('Create the processes')
                for j in range(n_reals):
                    p = mp.Process(target=worker, args=(j, reals_pulses, output, args))
                    processes.append(p)
                    p.start()
                print('Collect the result')
                # Collect the results
                results = [output.get() for _ in range(n_reals)]
            
                print('Wait for everybody')
                # Ensure all processes have finished
                for p in processes:
                    p.join()

                min_result = min(results, key=lambda x: x[1])
                j_opt, eres_opt, theta_opt = min_result
                print('MP finished!')

            # Finish multiprocessing: start subsequential    
            else:
                for j in range(n_reals):
                    print('Realization ' + str(j+1)+ '/'+str(n_reals))
     
                    random_pulses = random_pulse_generation(pulses, var, NOISE, Warm_start, Warm_from_file, Nc,  Nc0 , Nc_prev)      
                    
                    if HESS:
                        reals_eres[j], reals_theta[j], reals_hess[j]= single_random_realization(random_pulses, qaoa_cirq, theta_start, ftol, Nc, P, dressing_x, dressing_z, basis, dt, psi_0, Emin, Emax, HESS, ORDER, opt_method)
                        
                    else:  reals_eres[j], reals_theta[j] = single_random_realization(random_pulses, qaoa_cirq, theta_start, ftol, Nc, P, dressing_x, dressing_z, basis, dt, psi_0, Emin, Emax, HESS, ORDER, opt_method)
                    
                    reals_pulses[j] = random_pulses

                j_opt = np.argmin(reals_eres)
                eres_opt = reals_eres[j_opt]
                theta_opt = reals_theta[j_opt]

            # Multiprocessing and subsequential evaluation recombine
            print('The show must go on')
            pulses_opt = reals_pulses[j_opt]
            fi_t = func_basis(pulses_opt, Nc, P, basis)
            
            
            if HESS:
                hess_opt =  reals_hess[j_opt]
                hess_dir =  crab_dir+"/Hessians/"
                if not os.path.exists(hess_dir):os.makedirs(hess_dir)
                hess_output = hess_dir+'Hess_'+basis+'_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+"_nreals_"+str(n_reals)+".dat"
                np.savetxt(hess_output, hess_opt)
                
                

        # RAND = FALSE
        
        else: 
            pulses_opt = pulses
            fi_t = func_basis(pulses_opt, Nc, P, basis)

            args= (qaoa_cirq, dressing_x, dressing_z, pulses_opt, fi_t, dt, psi_0, ORDER)
            
            if HESS: eres_opt, theta_opt, hess_opt = optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, HESS = HESS, verbose = False)
                
            else:
                eres_opt, theta_opt = optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, HESS = HESS, verbose = False)
 
            if HESS:
                hess_dir =  crab_dir+"/Hessians/"
                if not os.path.exists(hess_dir):os.makedirs(hess_dir)
                hess_output = hess_dir+'Hess_'+basis+'_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+".dat"
                np.savetxt(hess_output, hess_opt)
    
        Cz0_opt , Cx0_opt = theta_opt[0] , theta_opt[1]
        Cz_i_opt, Cx_i_opt =  theta_opt[2::2], theta_opt[3::2]
        theta_z_opt[count], theta_x_opt[count], s_opt[count] = amplitudes(Cx0_opt, Cz0_opt, Cx_i_opt, Cz_i_opt, dressing_x, dressing_z, fi_t)
        t_gba = time.time() - start_time


        for m in range(P):
            while theta_x_opt[count][m] > np.pi/2: theta_x_opt[count][m] = theta_x_opt[count][m] - np.pi/2
            while theta_x_opt[count][m] < 0: theta_x_opt[count][m] = theta_x_opt[count][m] + np.pi/2

    
        #Export: one output file for each Nc  
        
        write_output_file(crab_dir, tau, Nc, n_reals, dt,
                      Cz0_start, Cx0_start, Cz_i_start, Cx_i_start,
                      pulses_opt, 
                      Cz0_opt, Cx0_opt, Cz_i_opt, Cx_i_opt, theta_z_opt[count], theta_x_opt[count], 
                      eres_opt, t_gba, basis, Warm_start, RAND, NOISE, MP)
        
            
    # END OF THE LOOP OVER Ncs
    return theta_x_opt, theta_z_opt
    
    
######### INTERP ############

def interpolation(y_prev, P_new):
    '''
    

    Parameters
    ----------
    y_prev : Array
        Array to be interpolated.
    P_new : Integer
        Number of point of the interpolation.
 
    Returns
    -------
    y_new : P_new-dimensional array
        Interpolated array.

    '''
    assert y_prev.ndim == 1
    P_prev = y_prev.shape[0]
    x_prev = np.linspace(0, 1, P_prev)
    x_new = np.linspace(0, 1, P_new)
    
    scipy.interpolate.interp1d(x_prev, y_prev, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value="extrapolate", assume_sorted=True)
    y_new = np.interp(x_new, x_prev, y_prev, left=None, right=None, period=None)
    return y_new  


def interpolation_z(theta_z, Pnew):
    '''
    

    Parameters
    ----------
    theta_z : Array
        Theta_z(m) coming from a previous optimization.
    P_new : Integer
        Number of point of the interpolation.

    Returns
    -------
    theta_z_new : P_new-dimensional array
        Interpolated theta_z with boundary condition theta_z[0] = 0. USEFUL FOR LogINTERP INTERPOLATION

    '''
    theta_z_new = np.zeros(Pnew)
    P_prev = len(theta_z)

    theta_z = np.flip(np.append(np.flip(theta_z), 0))   

    for m in range(1,P_prev):
        theta_z_new[2*m-1] = theta_z[m]
        theta_z_new[2*m] = 0.5*(theta_z[m] + theta_z[m+1])
        
    theta_z_new[Pnew-1] = theta_z[P_prev]
    return theta_z_new


def interpolation_x(theta_x, Pnew):
    '''

    Parameters
    ----------
    theta_x : Array
        Theta_x(m) coming from a previous optimization.
    P_new : Integer
        Number of point of the interpolation.

    Returns
    -------
    theta_x_new : P_new-dimensional array
        Interpolated theta_x with boundary condition theta_x[-1] = 0. USEFUL FOR LogINTERP INTERPOLATION

    '''
    Pold = len(theta_x)
    theta_x_new = np.zeros(Pnew)

    theta_x = np.append(theta_x, 0)

    for m in range(Pold):
        theta_x_new[2*m] = theta_x[m]
        theta_x_new[2*m-1] = 0.5*(theta_x[m-1] + theta_x[m])

    theta_x_new[Pnew-1] = 0
    return theta_x_new


def load_iter_optimized_schedule(params_dir, P, steps="Lin"):
    '''
    

    Parameters
    ----------
    params_dir : String
        QAOA parameters directory.
    P : Int
        Depth of the quantum circuit.
    steps : String, optional
        Interpolation procedure. Can be 'Lin' or 'Log'. The default is "Lin".

    Returns
    -------
    gamma_vec : P-dimensional array of floats
        Theta_z(m) .
    beta_vec : P-dimensional array of floats
        Theta_x(m).
    s_vec : P-dimensional array of floats
        Theta_z(m) / (Theta_z(m) + Theta_x(m))

    '''
    if steps == "Lin":
        beta_vec, gamma_vec, s_vec = np.loadtxt(f"{params_dir}/P" +str(P)+"_"+steps+"_INTERP_optimal_parameters.dat", unpack=True)
    elif steps == "Log":
        beta_vec, gamma_vec, s_vec = np.loadtxt(f"{params_dir}/P" +str(P)+"_"+steps+"_INTERP_optimal_parameters.dat", unpack=True)
   
    return gamma_vec, beta_vec, s_vec



def analize_final_state_np(qaoa_cirq, gamma_vec, beta_vec):
    '''
    

    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    gamma_vec : P-dimensional array
        Theta_z(m).
    beta_vec :  P-dimensional array
        Theta_c(m).

    Returns
    -------
    E_qaoa : Float
        Final expectation value of Hz.

    '''
    gamma_vec = np.array(gamma_vec)
    beta_vec = np.array(beta_vec)
    psi_qaoa = qaoa_cirq.state(gamma_vec, beta_vec)

    E_qaoa = qaoa_cirq.energy_expect(psi_qaoa, 1., 0)
    
    return E_qaoa



def INTERP_main(qaoa_cirq, output_dir, P0, Pmax, interp_method, 
                NEW = True,  opt_method = 'BFGS'):
    '''
    

    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    output_dir : string
        Output directory.
    P0 : Int
        Initial depth of the interpolation procedure.
    Pmax : Int
        Maximum depth of the interpolation procedure.
    interp_method : string
        Interpolation method. Can be 'Log' or 'Lin'
    NEW : Bool, optional
        If True, it optimizes the circuit of depth P0. If False, it imports an existing file of parameters. The default is True.
    opt_method : string, optional
        Optimization method. The default is 'BFGS' (based on analytical calculation of the gradient).

    Returns
    -------
    Performs the QAOA optimization over a array of depths using the desired interpolation warm-start procedure.

    '''
    
    params_dir = output_dir+"/QAOA_params/" # setting the output folder for INTERP parameters
    if not os.path.exists(params_dir):os.makedirs(params_dir)
    
    #INPUT INFORMATIONS FOR INTERPOLATION+OPTIMIZATION PROCEDURE
    psi_start = qaoa_cirq.psi_start
    Emin, Emax = qaoa_cirq.Emin, qaoa_cirq.Emax

    nruns = 1
    niter = 1 

    if NEW:
        
        op_seq_0 = P0 * [qaoa_cirq.apply_Uz, qaoa_cirq.apply_Ux]
        gamma_start, beta_start = 0,0
    
        theta_P0_guess = np.zeros(2*P0)
        theta_P0_guess[::2], theta_P0_guess[1::2] = gamma_start, beta_start
    
        E_P0, theta_P0, psi_qaoa, E_list, theta_list = QAOA_numpy_lib.optimimize_angles(psi_start, op_seq_0, qaoa_cirq, nruns, niter, method = opt_method, theta_guess=theta_P0_guess)
        eres_0 = (E_P0 - Emin)/(Emax - Emin)
    
        gamma_0, beta_0 =  theta_P0[::2], theta_P0[1::2]
        s_0 = gamma_0/(beta_0 + gamma_0)
    
        params_filename = f"{params_dir}/P" +str(P0)+"_"+ interp_method +"_INTERP_optimal_parameters.dat"
        np.savetxt(params_filename, np.column_stack(( beta_0, gamma_0, s_0)), header = "beta // gamma // s")
        
    else: 
        gamma_0, beta_0, s_0 = load_iter_optimized_schedule(params_dir, P0, steps='Lin')
        E0 = analize_final_state_np(qaoa_cirq, gamma_0, beta_0)
        eres_0 = (E0 - Emin) / (Emax - Emin)

        

    # INTERPOLATION PROCEDURE STARTS #

    if interp_method=='Lin':
        Ps = np.linspace(P0+1,Pmax,Pmax-P0, dtype = 'int')

    if interp_method == 'Log':
        Ps = np.logspace(P0, np.log2(Pmax),num = int(np.log2(Pmax))-1,base=2.0,dtype='int')

    eres = np.zeros(Ps.size)
    gamma_opt, beta_opt, s_opt = gamma_0, beta_0, s_0
    start_total_time = time.time() 

    for count, P in enumerate(Ps):
        print(P)
        op_seq = P*[qaoa_cirq.apply_Uz, qaoa_cirq.apply_Ux]
        
        if interp_method =='Lin': # Linear Interpolation
            gamma_guess = interpolation(gamma_opt, P)
            beta_guess = interpolation(beta_opt, P)
        
        if interp_method == 'Log': # Log Interpolation + boundary conditions
            gamma_guess = interpolation_z(gamma_opt, P)
            beta_guess = interpolation_x(beta_opt, P)

      
        theta_guess = np.zeros(2*P)
        theta_guess[::2], theta_guess[1::2] = gamma_guess, beta_guess
        
        E_opt, theta_opt, psi_qaoa, E_list, theta_list = QAOA_numpy_lib.optimimize_angles(psi_start, op_seq, qaoa_cirq, nruns, niter, method="BFGS", theta_guess=theta_guess)
        
        gamma_opt, beta_opt =  theta_opt[::2], theta_opt[1::2]
        s_opt = gamma_opt / (gamma_opt + beta_opt)

        eres[count] = (E_opt - Emin)/(Emax - Emin)
        
        params_filename = f"{params_dir}/P" +str(P)+"_"+ interp_method +"_INTERP_optimal_parameters.dat"
        np.savetxt(params_filename, np.column_stack(( beta_opt, gamma_opt, s_opt)), header = "beta // gamma // s")
        

    # EXPORT DATA
    maxcuts_filename = f"{params_dir}/"+interp_method+"_INTERP_residual_energies_vs_P.dat"    
    Ps_exp = np.concatenate(([P0], Ps))
    eres_exp = np.concatenate(([eres_0], eres))
    np.savetxt(maxcuts_filename, np.column_stack((Ps_exp, eres_exp)) , header = "P // residual energy")

    print('QAOA-INTERP optimization took ' , (time.time() - start_total_time), ' seconds')




###### LINEAR DQA #########


def get_QAOA_linear_schedule(P, dt):
    '''
    

    Parameters
    ----------
    P : Int
        Depth of the circuit.
    dt : float
        Time-step.

    Returns
    -------
    gamma : P-dimensional array
        Theta_z(m).
    beta : P-dimensional array
        Theta_x(m).
    s : P-dimensional array of floats
        Theta_z(m) / (Theta_z(m) + Theta_x(m))

    '''
    m = np.arange(1., P + 1.)
    s = m / (P + 1.)
    gamma = s * dt 
    beta = (1 - s) * dt
    return gamma, beta, s



def Optimal_dt(qaoa_cirq, P, dts):
    '''
        Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    P : Int
        Depth of the circuit.
    dts : Array
        Array of time steps .

    Returns
    -------
    eres_opt : float
        Optimal residual energy.
    dt_opt : float
        Optimal time step for a fixed P.
    eres_vs_dt : Array
        Residual energy as a function of dts.

    '''
    L = len(dts)
    eres_vs_dt = np.zeros(L)
    
    
    for j,dt in enumerate(dts):
        gamma, beta,s = get_QAOA_linear_schedule(P, dt)
        eres_vs_dt[j] = analize_final_state_np(qaoa_cirq, gamma, beta)
    
    best = np.argmin(eres_vs_dt)
    dt_opt = dts[best]
    eres_opt = eres_vs_dt[best]
    
    return eres_opt, dt_opt, eres_vs_dt


def DQA_main(qaoa_cirq, dts, Ps, output_dir, Ps_selected = None):
    '''
    

    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    dts : Array
        Array of time steps .
    Ps : Array
        Array of circuit depths.
    output_dir : string
        Output directory.
    Ps_selected : Array of integers, optional
        Selected values of P for which export residual energy vs dt. To be chosen among Ps The default is None.

    Returns
    -------
    Perform DQA optimization over the selected MaxCut instance.

    '''
    params_dir = output_dir+"/DQA/" # setting the output folder for DQA 
    if not os.path.exists(params_dir):os.makedirs(params_dir)

    l = len(Ps)
    l_dt = len(dts)
    
    eres_vs_P, dts_vs_P = np.zeros(l), np.zeros(l)
    eres_vs_dt_P = np.zeros((l,l_dt))  
    for j,P in enumerate(Ps):
        eres_vs_P[j], dts_vs_P[j], eres_vs_dt_P[j] =  Optimal_dt(qaoa_cirq, P, dts)
        
    # Export 
    
    exp = np.vstack((Ps, eres_vs_P, dts_vs_P)).T
    filename = params_dir + '/DQA_residual_energies_vs_P.dat'
    np.savetxt(filename, exp, header = 'P // Residual energies // Optimal dt')
    
    if Ps_selected:
        eres_vs_dt_P_fixed = np.zeros(( len(Ps_selected), l_dt))
        for j,P_s in enumerate(Ps_selected):
            indx = Ps.index(P_s)
            eres_vs_dt_P_fixed[j] = eres_vs_dt_P[indx]
            
            filename_vs_dt = params_dir + '/eres_vs_dt_P_{}.dat'.format(P_s)
            exp_dt= np.vstack((dts, eres_vs_dt_P_fixed[j])).T
            np.savetxt(filename_vs_dt, exp_dt, header = 'dt // Residual energies')

        
###################### ADDITIONAL FUNCTIONS #####################################
    

#Computes the population of a set of n_eigs states
def population(states,eigs,n_eigs):
    '''
    

    Parameters
    ----------
    states : array
        State of the system as a function of time.
    eigs : array
        Instantaneous eigenstates of the Hamiltonian as a function of time.
    n_eigs : int
        Number of eigenstates considered in the calculation.

    Returns
    -------
    pop array
        Population of the first n_eigs intantaneous eigenstates as a function of m.

    '''
    
    n_eigs = n_eigs
    times = len(states)
    pop = np.zeros((times, n_eigs))
    for t in range(times):
        for m in range(n_eigs):
            pop[t,m] = np.abs(np.dot(np.conjugate(eigs[t].T[m]),states[t]))**2
    return pop.T
 

#Computes the populations of the first n_eigs eigenvectors and the spectrum of the time dependent Hamiltonian

def population_and_spectrum_vs_t(qaoa_cirq, params_dir, beta_pop, gamma_pop, n_eigs, tau, dt):

    '''
    Parameters
    ----------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    params_dir : string
        Parameters directory.
    beta_pop : P-dimensional array
        Theta_x(m).
    gamma_pop : P-dimensional array
        Theta_z(m).
    n_eigs : int
        Number of eigenstates considered in the calculation.
    tau : float
        Total time evolution. If dt=1 coincides with P 
    dt : float
        Time scale. In a digital circuit, dt = 1.

    Returns
    -------
    Exports spectrum vs time and population of the first n_eigs instantaneous eigenstates vs time.

    '''
    P = int(tau/dt)
    t_list = np.linspace(0,tau-dt,P) 
    N = qaoa_cirq.N

    #Spectrum 
    energies = np.zeros((t_list.size, n_eigs)) 
    eigenvectors_vs_t = np.zeros((t_list.size, 2**(N-1), n_eigs), dtype = 'complex')
    eigvecs0 = None
    for m,t in enumerate(t_list):
        eigenvalues, eigenvectors = qaoa_cirq.Hxz_eigsh_np(gamma_pop[m], beta_pop[m], k=n_eigs, eigvecs0=eigvecs0)
        energies[m] = eigenvalues
        eigenvectors_vs_t[m] = eigenvectors
        eigvecs0 = np.sum(eigenvectors, axis=1) # set initial guess for next iteration
    
    psi_0 = qaoa_cirq.psi_start
    state_step = [digital_state(gamma_pop[:m], beta_pop[:m], dt, psi_0) for m in range(P)]
    pops = population(state_step,eigenvectors_vs_t,n_eigs)
   
    #Export files 
   
    filename_output_pops = params_dir+"/Populations.dat"
    np.savetxt(filename_output_pops, np.vstack((t_list, pops)).T)


    filename_output_spec = params_dir+"/Spectrum.dat"
    np.savetxt(filename_output_spec, np.vstack((t_list, energies.T)).T)
    print("Done!")
    


#Numerical gradient with respect to an array x of the generic function f(x, *args) USEFUL FOR BENCHMARKS
def num_gradient(f, x, args, h = 1e-6 ):   
    '''
    

    Parameters
    ----------
    f : function
        Scalar function.
    x : array
        Array with respect to compute the gradient.
    args : list
        Additional arguments of f.
    h : float, optional
        Finite differences approximation parameter. The default is 1e-6.

    Returns
    -------
    df_dxi : Array
        Numerical approximation of the gradient of f with respect to x.

    '''
    df_dxi = np.zeros(len(x))
    f_x = f(x, *args)  
    
    for i in range(len(x)):
        x_bin = list(x)
        x_bin[i] += h
        f_i = f(x_bin , *args)
        df_dxi[i] = (f_i - f_x)/ h
    
    return df_dxi
  


#This generate the MaxCut problem given the weight matrix and a string denoting the instance
def generate_instance(w_file_path, char, flip_sym = True):
    '''
    

    Parameters
    ----------
    w_file_path : string
        Path of the weight matrix of the MaxCut instance.
    char : string
        String to associate to the instance: for output purposes

    Returns
    -------
    qaoa_cirq : Object of QAOA_numpy_lib
        Object containing all the informations about the instance and the related Hamiltonians.
    sol0 : array
        Classical solution of the MaxCut instance.
    instance_dir : string
        Instance directory.
    figures_dir : string
        Figures directory.

    '''

    # Read the weight matrix
    graph_data = np.loadtxt(w_file_path)


    ### Locate the output folders ###
    
    curr_dir = os.getcwd()
    instances_dir = curr_dir+'/Hard_instances/'
    
    instance_dir = instances_dir+char+"3r_N14/"  # Specific output folder of the instance taken into consideration
    figures_dir = instance_dir+"/figs/"

    if not os.path.exists(instance_dir):os.makedirs(instance_dir)
    if not os.path.exists(figures_dir):os.makedirs(figures_dir)

    #### ###
       
    N = int(np.max(graph_data[:, 0:2]))+ 1
    weighted_edges = []

    for i_edge in range(graph_data.shape[0]):
        i, j, w = int(graph_data[i_edge, 0]), int(graph_data[i_edge, 1]), graph_data[i_edge, 2]
        weighted_edges.append((w, (i,j)))

 
    ## Creation of the MaxCut Hamiltonian ##

    qaoa_cirq = QAOA_numpy_lib.QAOA(N, weighted_edges, flip_sym = flip_sym)

    sol_list = qaoa_cirq.solution_list
    sol0 = sol_list[0]
    
    return qaoa_cirq, sol0, instance_dir, figures_dir

