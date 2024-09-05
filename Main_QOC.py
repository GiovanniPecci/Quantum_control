import Max_cut_QOC_modulus
import os

curr_dir = os.getcwd()

#Path of weight_matrix.dat
w_file_path = curr_dir+"/weight_matrices/W_weight_matrix.dat"
qaoa_cirq, sol0, output_dir, figures_dir = Max_cut_QOC_modulus.generate_instance(w_file_path, 'W')


## QOC METHODS FOR SMOOTH OPTIMAL PARAMETERS: Set the desired methods to True  ## 

CRAB = True
QAOA_INTERP = False
DQA_LIN = False



################################################ CRAB Options ################################################ 

if CRAB:
    ORDER = 1 # Order of the Trotter approximation
    opt_method = 'BFGS-analytical' # Optimization method : 'BFGS-analytical' , 'Nelder-Mead', 'BFGS-numerical'. See Max_cut_QOC_modulus for details 
    RAND = True # Randomization of the functional basis : if False the desired orthogonal functional basis is used
    
    Full = False # Regardless the value of Ncs, uses P/2 functional coefficients
    Warm_start = False # The initial condition for the j-th element in Ncs is the optimal result for the (j-1)-th element of Ncs
    WFF = False # The initial condition for the first Nc in the list is extracted from an input file. Afterwards, the regular Warm_start procedure starts
    STATS = False # Reiterates the calculation different times : useful to produce statistics of the solutions 
    HESS = False # Store and saves the approximate inverse Hessian of the optimal solution
    DRESS = False # Implements dressed CRAB 
    MP = True # Implements parallel computation : use only for large P 
    
    #### PARAMETERS ####
    taus = [2] # total time of evolution - will be trotterized!
    
    basis= 'FOU' # functional basis = 'FOU' for Fourier modes or 'CHB' for Chebyschev polynomials
    dt = 1 # Time unit
#    Ncs = [2] #Number of functional coefficients (it only matters if Full = False)
    
    
    ### CRAB CONDITIONAL PARAMETERS ###
    
    if WFF:
        file = output_dir+'/CRAB/Warm_start_FOU_OUTPUT_file_tau_40_Nc5_nreals_4.dat' # Path of the input file
    else : file = None
    
    if STATS:
        N_samples = 100 # Number of ripetitions of a single CRAB optimization. If RAND is TRUE, the total number of optimizations is n_reals*N_samples
    
    if RAND:
        n_reals = 10 # Number of realizations over different sets of Ncs random frequencies
        NOISE  = 'GAMMA' # Set ADD or MULT to have respectively additive or multipicative noise  
    else : 
        n_reals = None
        NOISE = None
    
    if DRESS: 
        n_super_iter = 5


################################################ INTERP Options ################################################ 

if QAOA_INTERP:
    P_0 = 2 # Inital value of P
    P_max = 10 # Final value of P 
    
    NEW = True # If True, optimize the first step P=P0; if False, import optimal parameters from a file for P=P0. Set to False only after optimizing P=P0. 
    interp_method = 'Lin' # Interpolation methods: choose between "Lin" or "Log" 
    

################################################ DQA Options ################################################ 

if DQA_LIN:
    export_data_eres_vs_dt_for_fixed_P = True # If True, export a file containing residual energies vs dt for each fixed P in the array P_selected. If False, nothing happens.
    
    dt_min, dt_max, step_dt = 0.1, 0.2, 0.01    # Respectively maximum and minimum dt and the corresponding increment. It fixes the array dts. 
    P_min, P_max, step_P = 2, 6, 1 # Respectively maximum and minimum P and the corresponding increment. It fixes the array Ps.
    
    
    
    Ps =  list(range(P_min, P_max+step_P, step_P)) 
    
    if export_data_eres_vs_dt_for_fixed_P:
        P_selected = Ps[::2] # Values of P for which export the file of residual energies vs dt. Default: every other element in Ps. 
    else : P_selected = None
      

################################################ MAIN ################################################ 

if __name__ == "__main__":

    if CRAB:
        print('CRAB starts..')
        for tau in taus:
            
            Ncs = [tau//2]
            
            args = (qaoa_cirq, output_dir, figures_dir, 
                    tau, dt, basis, Ncs, n_reals, 
                    Full, RAND, ORDER, Warm_start, WFF, file, 
                    opt_method, MP, HESS, NOISE, DRESS)   
            
            if STATS:
                args = (figures_dir, 
                        tau, dt, basis, Ncs, n_reals, 
                        Full, RAND, ORDER, Warm_start, WFF, file, 
                        opt_method, MP, HESS, NOISE, DRESS)   
                            
                output_dir = output_dir + '/Stats_CRAB_tau_'+str(tau)
                for j in range(N_samples):
                    print('Sample ' + str(j+1)+ '/'+str(N_samples))
                    sample_dir = output_dir+'/sample_'+str(j+1)+'/'
                    if not os.path.exists(sample_dir):os.makedirs(sample_dir)
                    Max_cut_QOC_modulus.CRAB_main(qaoa_cirq, sample_dir, *args)
            
            if DRESS:
                
                args = (figures_dir, 
                        tau, dt, basis, Ncs, n_reals, 
                        Full, RAND, ORDER, Warm_start, WFF, file, 
                        opt_method, MP, HESS, NOISE, False)   
                
                print('Super iteration ' + str(1)+ '/'+str(n_super_iter))
    
                dress_fold = output_dir + '/Super_iter_0/'
                if __name__ == "__main__":
                    theta_x_old, theta_z_old = Max_cut_QOC_modulus.CRAB_main(qaoa_cirq, dress_fold, *args)
                    print(theta_x_old, theta_z_old)
               
                
                    for j in range(1,n_super_iter):
                        print('Super iteration ' + str(j+1)+ '/'+str(n_super_iter))
                                
                        dress_fold = output_dir + '/Super_iter_{}/'.format(j)
        
                        args = (figures_dir, 
                                tau, dt, basis, Ncs, n_reals, 
                                Full, RAND, ORDER, Warm_start, WFF, file, 
                                opt_method, MP, HESS, NOISE, DRESS, theta_z_old, theta_x_old)  

                        theta_x_old, theta_z_old = Max_cut_QOC_modulus.CRAB_main(qaoa_cirq, dress_fold, *args)
                        print(theta_x_old, theta_z_old)
                        
                                
            else : 
                Max_cut_QOC_modulus.CRAB_main(*args)
                
        

    if QAOA_INTERP:
        print('INTERP starts..')
        args = (qaoa_cirq, output_dir, P_0, P_max, interp_method)    
        Max_cut_QOC_modulus.INTERP_main(*args)
    
    
    
    if DQA_LIN: 
        print('dQA starts..')
        dts = [x *step_dt for x in range(int(dt_min/step_dt), int(dt_max/step_dt)+1, 1)]
        Max_cut_QOC_modulus.DQA_main(qaoa_cirq, dts, Ps, output_dir, Ps_selected = P_selected)
    
    
    print('Done! \n')



