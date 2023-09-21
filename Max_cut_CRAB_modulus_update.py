import scipy.sparse.linalg
import QAOA_numpy_lib
import time

import scipy.optimize
import scipy.interpolate

import numpy as np
import os
import sys


######### CRAB ############

#Define the basis of L2 integrable function    
def func_basis(pulses, Nc, P, BASIS= 'FOU'):
    fi_t = np.zeros((P, Nc))

    if BASIS == 'FOU':
        for m in range(P):
            fi_t[m] = [np.sin(np.pi*pulses[n]*m/P) for n in range(Nc)]        
    elif BASIS == 'CHB':   
        for m in range(P):
            fi_t[m] = [np.cos(pulses[n]*np.arccos(m/P)) for n in range(Nc)]
    else : print('Functional basis not available yet')
    return fi_t
 
    

#Returns the driving fields given pulses and functional basis
def amplitudes(delta_t, gamma_i, beta_i, fi_t): 
    P = fi_t.shape[0]
    
    gamma, beta = np.zeros(P), np.zeros(P)
    for m in range(P):
        gamma[m] = m/P*(delta_t+ np.einsum('i->' , gamma_i*fi_t[m]))
        beta[m]  = (1-m/P)*(delta_t+ np.einsum('i->' , beta_i*fi_t[m]))
    s_opt = gamma/(gamma+beta)
    return gamma, beta, s_opt




# Cost function - to be optimized with gradient free algorithm (first or second order Trotter)

def fig_of_merit(theta_i, qaoa_cirq, pulses, fi_t, dt, psi_0):
    P = fi_t.shape[0]
    delta_t = theta_i[0]
    gamma_i, beta_i =  theta_i[1::2], theta_i[2::2]
    gamma_t, beta_t, s_t = amplitudes(delta_t, gamma_i,beta_i,fi_t)
    psi_final = psi_0
    
    for m in range(P):
        psi_final = qaoa_cirq.apply_trotter_I(gamma_t[m], beta_t[m], dt, psi_final)
    efinal = qaoa_cirq.energy_expect(psi_final,1,0)
    
    return efinal
    

def fig_of_merit_II(theta_i, qaoa_cirq, pulses, fi_t, dt, psi_0):
    P = fi_t.shape[0]
    delta_t = theta_i[0]
    gamma_i, beta_i =  theta_i[1::2], theta_i[2::2]
    gamma_t, beta_t, s_t = amplitudes(delta_t, gamma_i,beta_i,fi_t)
    psi_final = psi_0
    
    for m in range(P):
        psi_final = qaoa_cirq.apply_sym_trotter_II(gamma_t[m], beta_t[m], dt, psi_final)
    efinal = qaoa_cirq.energy_expect(psi_final,1,0)
    
    return efinal
    


#Returns the final state evaluated using trotterized dynamics (first and second order Trotter)

def digital_state(qaoa_cirq, gamma_vec, beta_vec, dt, psi_0):
        P = gamma_vec.shape[0]
        psi_qaoa = psi_0
        for m in range(P):
            psi_qaoa = qaoa_cirq.apply_trotter_I(gamma_vec[m], beta_vec[m], dt, psi_qaoa)
        return psi_qaoa
 
def digital_state_II(qaoa_cirq, gamma_vec, beta_vec, dt, psi_0):
        P = gamma_vec.shape[0]
        psi_qaoa = psi_0
        for m in range(P):
            psi_qaoa = qaoa_cirq.apply_sym_trotter_II(gamma_vec[m], beta_vec[m], dt, psi_qaoa)
        return psi_qaoa


#Cost function and its gradient respect to theta_i - to be optimized with gradient based algorithm (first and second order Trotter)


def energy_with_gradient(theta_i, qaoa_cirq, pulses, fi_t, dt, psi_0):
    
    delta_t = theta_i[0]
    gamma_i, beta_i =  theta_i[1::2], theta_i[2::2]
    gamma_t, beta_t, s_t = amplitudes(delta_t, gamma_i,beta_i,fi_t)
    
    P = fi_t.shape[0]
    Nc = len(gamma_i)   
    
    ######################################################################
    
    psi_m =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')
    psi_m[0] = psi_0
    for m in range(P):
        psi_m[m+1] = qaoa_cirq.apply_trotter_I(gamma_t[m], beta_t[m], dt, psi_m[m])

    psi_final = psi_m[-1]
    
        
    phi_z = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
    for m in range(P):
        phi_z[m]  = qaoa_cirq.apply_Uz(psi_m[m], gamma_t[m]*dt)
    
    ####################################################################
    backwards_prop =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')

    hz_psi_f = qaoa_cirq.apply_Hz(psi_final, 1)
    
    backwards_prop[0] = hz_psi_f
    for m in range(P):
        backwards_prop[m+1] = qaoa_cirq.apply_trotter_I(np.flip(gamma_t)[m], np.flip(beta_t)[m], dt, backwards_prop[m], herm = True)
        

    chi_x = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
    for m in range(P):
        chi_x [m] = qaoa_cirq.apply_Ux(backwards_prop[m], -np.flip(beta_t)[m]*dt) 

        
    chi_x = np.flip(chi_x, axis = 0)    
    
    d_dbeta, d_dgamma = np.zeros(P, dtype = 'complex'), np.zeros(P, dtype = 'complex')
    chain_rule_g, chain_rule_b = np.zeros((P, Nc), dtype = 'complex'), np.zeros((P, Nc), dtype = 'complex')
    chain_rule_delta_t = np.zeros(P, dtype = 'complex')
    
    for m in range(P):
        d_dbeta[m]  = np.dot(np.conjugate(chi_x[m]), -1j*dt*qaoa_cirq.apply_Hx(phi_z[m],1))
        d_dgamma[m] = np.dot(np.conjugate(chi_x[m]), -1j*dt*qaoa_cirq.apply_Hz(phi_z[m],1))

        chain_rule_delta_t[m] = (1-m/P)*2*np.real(d_dbeta[m]) + m/P*2*np.real(d_dgamma[m])

        
        chain_rule_b[m] = (1-m/P)*fi_t[m]*2*np.real(d_dbeta[m])
        chain_rule_g[m] = m/P*fi_t[m]*2*np.real(d_dgamma[m])

    ######################################################################
    grad_beta_i  = np.einsum('ij->i' , chain_rule_b.T)        
    grad_gamma_i = np.einsum('ij->i' , chain_rule_g.T) 
    grad_delta_t = np.einsum('i->' , chain_rule_delta_t) 

    

    grad_theta_i = np.zeros(2*Nc+1)
 
    grad_theta_i[0] = np.real(grad_delta_t)
    grad_theta_i[1::2], grad_theta_i[2::2] = np.real(grad_gamma_i), np.real(grad_beta_i)

    efinal = qaoa_cirq.energy_expect(psi_m[-1],1,0)

    return efinal, grad_theta_i   




def energy_with_gradient_II(theta_i, qaoa_cirq, pulses, fi_t, dt, psi_0):
    
    delta_t = theta_i[0]
    gamma_i, beta_i =  theta_i[1::2], theta_i[2::2]
    gamma_t, beta_t, s_t = amplitudes(delta_t, gamma_i,beta_i,fi_t)
    
    P = fi_t.shape[0]
    Nc = len(gamma_i)    
    
    ######################################################################
    
    psi_m =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')
    psi_m[0] = psi_0
    for m in range(P):
        psi_m[m+1] = qaoa_cirq.apply_sym_trotter_II(gamma_t[m], beta_t[m], dt, psi_m[m])

    psi_final = psi_m[-1]
    
        
    phi_z, phi_xz = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex'), np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
    for m in range(P):
        phi_z[m]  = qaoa_cirq.apply_Uz(psi_m[m], gamma_t[m]*dt/2)
        phi_xz[m] = qaoa_cirq.apply_Ux(phi_z[m],  beta_t[m]*dt)
    
    ####################################################################
    
    backwards_prop =  np.zeros((P+1,qaoa_cirq.dimH), dtype = 'complex')

    hz_psi_f = qaoa_cirq.apply_Hz(psi_final, 1)
    
    backwards_prop[0] = hz_psi_f
    for m in range(P):
        backwards_prop[m+1] = qaoa_cirq.apply_sym_trotter_II(np.flip(gamma_t)[m], np.flip(beta_t)[m], -dt, backwards_prop[m])
        

    chi_z, chi_xz = np.zeros((P,qaoa_cirq.dimH), dtype = 'complex'), np.zeros((P,qaoa_cirq.dimH), dtype = 'complex')
    d_dbeta, d_dgamma = np.zeros(P, dtype = 'complex'), np.zeros(P, dtype = 'complex')
 
    chain_rule_g, chain_rule_b = np.zeros((P, Nc), dtype = 'complex'), np.zeros((P, Nc), dtype = 'complex')
    chain_rule_delta_t = np.zeros(P, dtype = 'complex')
       

    
    for m in range(P):
        chi_xz[m] = qaoa_cirq.apply_Uz(backwards_prop[m], - np.flip(gamma_t)[m]*dt/2)
        chi_xz[m] = qaoa_cirq.apply_Ux(chi_xz[m], - np.flip(beta_t)[m]*dt)
     

    chi_xz = np.flip(chi_xz, axis = 0)    
    for m in range(P):
        
        chi_z [m] = qaoa_cirq.apply_Ux(chi_xz[m], beta_t[m]*dt) 
        d_dbeta[m]  = np.dot(np.conjugate(chi_xz[m]), -1j*dt*qaoa_cirq.apply_Hx(phi_z[m],1))
        d_dgamma[m] = np.dot(np.conjugate(chi_xz[m]), -1j*dt/2*qaoa_cirq.apply_Hz(phi_z[m],1)) + np.dot(np.conjugate(chi_z[m]), -1j*dt/2*qaoa_cirq.apply_Hz(phi_xz[m],1)) 
        
        chain_rule_delta_t[m] = (1-m/P)*2*np.real(d_dbeta[m]) + m/P*2*np.real(d_dgamma[m])

        
        chain_rule_b[m] = (1-m/P)*fi_t[m]*2*np.real(d_dbeta[m])
        chain_rule_g[m] = m/P*fi_t[m]*2*np.real(d_dgamma[m])

    ######################################################################
    grad_beta_i  = np.einsum('ij->i' , chain_rule_b.T)        
    grad_gamma_i = np.einsum('ij->i' , chain_rule_g.T) 
    grad_delta_t = np.einsum('i->' , chain_rule_delta_t) 

    

    grad_theta_i = np.zeros(2*Nc+1)
 
    grad_theta_i[0] = np.real(grad_delta_t)
    grad_theta_i[1::2], grad_theta_i[2::2] = np.real(grad_gamma_i), np.real(grad_beta_i)

    efinal = qaoa_cirq.energy_expect(psi_m[-1],1,0)

    return efinal, grad_theta_i   



#Performs the optimization
def optimize_pulse(fig_of_merit, theta_i, args, ftol, method, Emin, Emax, ORDER, verbose = True):
    
#   bounds = ((-1,1), (-1,1))*(int(len(theta_i)/2))
    bounds = None
    
    if method == 'Nelder-Mead':
        jac = False
        print('Gradient free optimization starts...')
 
        if ORDER == 1:
            res = scipy.optimize.minimize(fig_of_merit, theta_i, args=args, method='Nelder-Mead', bounds = bounds, tol = ftol)

        elif ORDER == 2:
            res = scipy.optimize.minimize(fig_of_merit_II, theta_i, args=args, method='Nelder-Mead', bounds = bounds, tol = ftol)

        else: print("Specify a valid order for the Trotter time evolution")

    
    elif method == 'BFGS-numerical':
        jac = False
        print('Gradient based optimization starts...')
        if ORDER == 1:
            res = scipy.optimize.minimize(fig_of_merit, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac)
 
        elif ORDER == 2:
            res = scipy.optimize.minimize(fig_of_merit_II, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac)

        else: print("Specify a valid order for the Trotter time evolution")
    


    elif method == 'BFGS-analytical':
        jac = True
        print('Exact gradient based optimization starts...')
        if ORDER ==1:
            res = scipy.optimize.minimize(energy_with_gradient, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac)
 
        elif ORDER ==2:
            res = scipy.optimize.minimize(energy_with_gradient_II, theta_i, args=args, method='BFGS', bounds = bounds, tol = ftol, jac = jac)
    
        else: print("Specify a valid order for the Trotter time evolution")
    
    eres = (res.fun-Emin)/(Emax - Emin)
    theta_opt = res.x
    print(res.message)

    if verbose:
        print('Number of iterations is ', res.nit)
        print('Number of calls of the cost function is ', res.nfev)
    return eres, theta_opt






############################################################################################


#This generate the MaxCut problem given the weight matrix and a string denoting the instance
def generate_instance(w_file_path, char):

    # Read the weight matrix
    graph_data = np.loadtxt(w_file_path)


    # Locate the output folders    
    curr_dir = os.getcwd()
    proj_dir = os.path.dirname(curr_dir)+'/'
    instances_dir = proj_dir+'Hard_instances/'
    
    
    
    
    N = int(np.max(graph_data[:, 0:2]))+ 1
    weighted_edges = []
    string = char

    for i_edge in range(graph_data.shape[0]):
        i, j, w = int(graph_data[i_edge, 0]), int(graph_data[i_edge, 1]), graph_data[i_edge, 2]
        weighted_edges.append((w, (i,j)))


    # Specific output folder for the instance taken into consideration
    instance_dir = instances_dir+string+"3r_N14"
    
    if not os.path.exists(instance_dir):os.makedirs(instance_dir)
        
    params_dir = instance_dir+"/CRAB/Warm_start"
    figures_dir = instance_dir+"/figs"

    if not os.path.exists(params_dir):os.makedirs(params_dir)
    if not os.path.exists(figures_dir):os.makedirs(figures_dir)

 
    ## Creation of the MaxCut Hamiltonian ##

    qaoa_cirq = QAOA_numpy_lib.QAOA(N, weighted_edges, flip_sym = True)
    output_dir = params_dir

    sol_list = qaoa_cirq.solution_list
    sol0 = sol_list[0]
    
    return qaoa_cirq, sol0, output_dir, figures_dir




def CRAB_main(qaoa_cirq, output_dir, figures_dir, tau, dt, basis, Ncs = [5], Full = False, RAND = True, ORDER = 1, Warm_start = False, Warm_from_file = False, input_file = None, opt_method = 'BFGS-analytical'):
 

    if Warm_from_file and not Warm_start : 
        print("In order to use Warm_from_file, Warm_start option should be active. Process aborted.")
        sys.exit()
        
    if Warm_start and Full:
        print("Full and Warm_start options cannot co-exist as they are contradictory. Process aborted.")
        sys.exit()

 
    
 
    # Time discretization 

    P = int(tau/dt)
    t_list = np.linspace(0,tau-dt,P) 

    # Circuit parameters 
    Emin, Emax = qaoa_cirq.Emin, qaoa_cirq.Emax
    psi_0 = qaoa_cirq.psi_start


    #Optimization parameters
    ftol = 1e-6 # Tolerance of the optimization 
    
#    if Full: Ncs = [P]
    if Full: Ncs = [int(P/2)]
    

    # Randomization parameters - set to False to have a ordered chopped functional basis
    if RAND:
        n_reals = 5 # Number of realizations for each set of random frequencies 
        var = 1 # Size of the interval of the random sampling 


    #Optimization starts


    #Coefficient of the series expansion
    gamma_opt, beta_opt, s_opt = np.zeros(( len(Ncs) ,  t_list.size)), np.zeros(( len(Ncs) ,  t_list.size)) , np.zeros(( len(Ncs),  t_list.size))
    
    if not Full and Warm_start:
        if Warm_from_file:
            P_ff, Nc_ff, gamma_ff, beta_ff, pulses_opt, gamma_i_opt, beta_i_opt, delta_t_opt = read_input_file(input_file)
        else: 
            gamma_i_opt, beta_i_opt = [0]*(Ncs[0]), [0]*(Ncs[0])
            pulses_opt = np.arange(1,Ncs[0]+1)
            delta_t_opt = 1

    count = 0
    for Nc in Ncs:
        print("Nc = ", Nc)
        
        #Initial guess for the functional coefficients
        gamma_i_start, beta_i_start = [0]*(Nc), [0]*(Nc)

        #Initial guess for the optimal delta_t
        delta_i_start = 1

        
        if not Full and Warm_start:    
            Nc_prev = len(gamma_i_opt)
            gamma_i_start[0:Nc_prev], beta_i_start[0:Nc_prev] = gamma_i_opt, beta_i_opt
            gamma_i_start[Nc_prev::], beta_i_start[Nc_prev::] = [0]*(Nc-Nc_prev), [0]*(Nc-Nc_prev)
            delta_i_start = delta_t_opt
        
        theta_start = np.zeros(2*Nc+1)
        theta_start[0] = delta_i_start
        theta_start[1::2], theta_start[2::2] = gamma_i_start, beta_i_start

    
        #Optimization of the amplitudes (analytical calculation of the gradient)
        start_time = time.time()
    
        pulses = np.arange(1,Nc+1, dtype = float)
        if not Full and Warm_start: 
            pulses[0:Nc_prev] = pulses_opt
        
        if RAND:
            reals_eres, reals_theta, reals_pulses = np.zeros(n_reals), np.zeros((n_reals, 2*Nc+1)), np.zeros((n_reals, Nc))
            for j in range(n_reals):
                print('Realization ' + str(j+1)+ '/'+str(n_reals))
                
                
                if not Full and Warm_start:
                    if Nc == Ncs[0] and not Warm_from_file: pulses = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc)] ) 
                    else: pulses[Nc_prev::] = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc_prev, Nc)] )   
                
                else: pulses = np.array( [pulses[n]*(1+var*(np.random.rand() - 0.5 ) ) for n in range(Nc)] ) 
                
                fi_t = func_basis(pulses, Nc, P, basis)
                args= (qaoa_cirq, pulses, fi_t, dt, psi_0)
                reals_eres[j], reals_theta[j] = optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, verbose = False)
                reals_pulses[j] = pulses

            j_opt = np.argmin(reals_eres)

            eres_opt = reals_eres[j_opt]
            theta_opt = reals_theta[j_opt]
        
            pulses = reals_pulses[j_opt]
            fi_t = func_basis(pulses, Nc, P, basis)
            
            if Warm_start:
                filename_output = output_dir+'/Warm_start_'+basis+'_OUTPUT_file_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+"_nreals_"+str(n_reals)+".dat"
            else:  
                filename_output = output_dir+'/'+basis+'_OUTPUT_file_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+"_nreals_"+str(n_reals)+".dat"

        else: 
            fi_t = func_basis(pulses, Nc, P, basis)
            args= (qaoa_cirq, pulses, fi_t, dt, psi_0)
            eres_opt, theta_opt = optimize_pulse(fig_of_merit, theta_start, args, ftol, method = opt_method, Emin = Emin, Emax = Emax, ORDER = ORDER, verbose = False)
            
            if Warm_start:
                filename_output = output_dir+'/Warm_start_'+basis+'_OUTPUT_file_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+".dat"
            else: 
                filename_output = output_dir+'/'+basis+'_OUTPUT_file_tau_'+str(tau)+"_dt_"+str(dt)+"Nc"+str(Nc)+".dat"


        delta_t_opt = theta_opt[0]
        gamma_i_opt, beta_i_opt =  theta_opt[1::2], theta_opt[2::2]
        gamma_opt[count], beta_opt[count], s_opt[count] = amplitudes(delta_t_opt, gamma_i_opt, beta_i_opt, fi_t)
        t_gba = time.time() - start_time

    
        #Export: one output file for each Nc
        pulses_opt = pulses
        left = np.vstack((t_list, beta_opt[count-1], gamma_opt[count-1], beta_opt[count], gamma_opt[count])).T
        right = np.vstack((beta_i_start, gamma_i_start, pulses_opt, beta_i_opt, gamma_i_opt)).T


        file_out = open(filename_output,'w')


        file_out.write("### PARAMETERS OF THE TIME EVOLUTION ### \n")

        file_out.write("Total time tau = " + str(tau)+"\nTimestep dt = "+str(dt)+"\nP = tau/dt = "+str(P)+"\n")
        file_out.write("Number of Fourier coefficients Nc = "+str(Nc)+"\n\n")

        file_out.write("gamma(t) = t/T*(delta_t + \sum_i^Nc gamma_i*f_i(wi, t) \nbeta(t) = (1-t/T)*(delta_t + \sum_i^Nc beta_i*f_i(wi, t)\n" )

        file_out.write("\nCRAB coefficient beta_i_0 // gamma_i_0 // pulses_i // beta_i_opt // gamma_i_opt  \n")
        np.savetxt(file_out, right)


        file_out.write("\nAmplitudes vs time: t // beta_0 // gamma_0 // beta_opt // gamma_opt  \n")
        np.savetxt(file_out, left)

        file_out.write("\nResidual energy after optimization is "+str( eres_opt)+' with delta_t = '+str(delta_t_opt))
        file_out.write('\nOptimization procedure took '+str(t_gba) +" seconds")


        file_out.close()
    
        count = count +1

    # END OF THE LOOP OVER Ncs
    
 

###################### ADDITIONAL FUNCTIONS #####################################
    



#Computes the population of a set of n_eigs states
def population(states,eigs,n_eigs):
    
    n_eigs = n_eigs
    times = len(states)
    pop = np.zeros((times, n_eigs))
    for t in range(times):
        for m in range(n_eigs):
            pop[t,m] = np.abs(np.dot(np.conjugate(eigs[t].T[m]),states[t]))**2
    return pop.T
 

#Computes the populations of the first n_eigs eigenvectors and the spectrum of the time dependent Hamiltonian

def CRAB_populations(qaoa_cirq, output_dir, beta_pop, gamma_pop, n_eigs, tau, dt):
    P = int(tau/dt)
    t_list = np.linspace(0,tau-dt,P) 
    N = qaoa_cirq.N

    #Spectrum 
    energies = np.zeros((t_list.size, n_eigs)) 
    eigenvectors_vs_t = np.zeros((t_list.size, 2**(N-1), n_eigs))
    eigvecs0 = None
    for m,t in enumerate(t_list):
        eigenvalues, eigenvectors = qaoa_cirq.Hxz_eigsh_np(gamma_pop[m], beta_pop[m], k=n_eigs, eigvecs0=eigvecs0)
        energies[m] = eigenvalues
        eigenvectors_vs_t[m] = eigenvectors
        eigvecs0 = np.sum(eigenvectors, axis=1) # set initial guess for next iteration
    
    psi_0 = qaoa_cirq.psi_0
    state_step = [digital_state(gamma_pop[:m], beta_pop[:m], dt, psi_0) for m in range(P)]
    pops = population(state_step,eigenvectors_vs_t,n_eigs)
   
    #Export files 
   
    filename_output_pops = output_dir+"/Populations.dat"
    np.savetxt(filename_output_pops, np.vstack((t_list, pops)).T)


    filename_output_spec = output_dir+"/Spectrum.dat"
    np.savetxt(filename_output_spec, np.vstack((t_list, energies.T)).T)
    print("Done!")
    


#Numerical gradient with respect to an array x of the generic function f(x, *args) USEFUL FOR BENCHMARKS
def num_gradient(f, x, args, h = 1e-6 ):    
    df_dxi = np.zeros(len(x))
    f_x = f(x, *args)  
    
    for i in range(len(x)):
        x_bin = list(x)
        x_bin[i] += h
        f_i = f(x_bin , *args)
        df_dxi[i] = (f_i - f_x)/ h
    
    return df_dxi
  


#Interpolates the elements of an array and sample P_new points from the resulting distribution USEFUL FOR ITERATIVE SCHEDULES
def interpolation(y_prev, P_new, left=None, right=None):
    
    assert y_prev.ndim == 1
    P_prev = y_prev.shape[0]
    x_prev = np.linspace(0, 1, P_prev)
    x_new = np.linspace(0, 1, P_new)
    
    scipy.interpolate.interp1d(x_prev, y_prev, kind='linear', axis=-1, copy=True, bounds_error=None, fill_value="extrapolate", assume_sorted=True)
    y_new = np.interp(x_new, x_prev, y_prev, left=left, right=right, period=None)
    return y_new  



def read_input_file(filename):
    file_crab = open(filename, 'r')
    file_lines = file_crab.readlines()

    P   = int(file_lines[3].split()[-1])
    Nc  = int(file_lines[4].split()[-1])

    
    
    pulses_i    = np.zeros(Nc,dtype='float')
    beta_i_opt  = np.zeros(Nc,dtype='float')
    gamma_i_opt = np.zeros(Nc,dtype='float')


    for m in range(Nc):
        [pulses_i[m], beta_i_opt[m], gamma_i_opt[m]] = file_lines[10:10+Nc][m][:-1].split()[2::]

    
    beta_opt  = np.zeros(P ,dtype='float')
    gamma_opt = np.zeros(P ,dtype='float')

    for m in range(P):
        [beta_opt[m], gamma_opt[m]] = file_lines[10+Nc+2:10+Nc+2+P][m][:-1].split()[3::]

    for m in range(P):
        while beta_opt[m] > np.pi/2: beta_opt[m] = beta_opt[m] - np.pi/2
        while beta_opt[m] < 0: beta_opt[m] = beta_opt[m] + np.pi/2
    
    C0 = float(file_lines[-2].split()[-1])

    return P, Nc, gamma_opt, beta_opt, pulses_i, gamma_i_opt, beta_i_opt, C0
