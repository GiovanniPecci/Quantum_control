import Max_cut_CRAB_modulus
import os

curr_dir = os.getcwd()
proj_dir = os.path.dirname(curr_dir)+'/'

#Path of weight_matrix.dat
w_file_path = proj_dir+"weight_matrices/W_weight_matrix.dat"
qaoa_cirq, sol0, output_dir, figures_dir = Max_cut_CRAB_modulus.generate_instance(w_file_path, 'w')

#### Options #### 

ORDER = 1 # Order of the Trotter approximation
opt_method = 'BFGS-analytical' # Optimization method : 'BFGS-analytical' , 'Nelder-Mead', 'BFGS-numerical'
RAND = True # Randomization of the functional basis : if False the desired orthogonal functional basis is used
Full = True # Regardless the value of Ncs, uses P/2 functional coefficients
Warm_start = False # The initial condition for the j-th element in Ncs is the optimal result for the (j-1)-th element of Ncs
WFF = False # The initial condition for every Nc is extracted from an input file
STATS = False # Reiterates the calculation different times : useful to produce statistics of the solutions 
HESS = True # Store and saves the approximate inverse Hessian of the optimal solution

#### PARAMETERS ####

taus = [64] # total time of evolution - will be trotterized!
basis= 'CHB' # functional basis = 'FOU' for Fourier modes or 'CHB' for Chebyschev polynomials
dt = 1 # Time unit
Ncs = [2,3] #Number of functional coefficients 


### CONDITIONAL PARAMETERS ###

if WFF:
    file = output_dir+'/CHEB/_CHB_OUTPUT_file_tau_64_dt_1Nc20_nreals_10.dat' # Path of the input file
else : file = None

if STATS:
    N_samples = 4 # Number of ripetitions of a single CRAB optimization. If RAND is TRUE, the total number of optimizations is n_reals*N_samples

if RAND:
    n_reals = 10
else : n_reals = None



### MAIN ###

for tau in taus:
#    Ncs = [2]+list(range(10,int(tau/2)-9,10)) + [int(tau/2)]

    args = (qaoa_cirq, output_dir, figures_dir, 
            tau, dt, basis, Ncs, n_reals, 
            Full, RAND, ORDER, Warm_start, WFF, file, 
            opt_method, HESS)

    if STATS:
        output_dir = output_dir + '/Stats_tau_'+str(tau)
        for j in range(N_samples):
            print('Sample ' + str(j+1)+ '/'+str(N_samples))
            sample_dir = output_dir+'/sample_'+str(j+1)+'/'
            if not os.path.exists(sample_dir):os.makedirs(sample_dir)
            Max_cut_CRAB_modulus.CRAB_main(*args)

    
    else : Max_cut_CRAB_modulus.CRAB_main(*args)

print('Done!')



