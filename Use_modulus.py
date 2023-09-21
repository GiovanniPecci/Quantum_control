import Max_cut_CRAB_modulus_update
import os

curr_dir = os.getcwd()

proj_dir = os.path.dirname(curr_dir)+'/'

#Path of weight_matrix.dat
w_file_path = proj_dir+"weight_matrices/W_weight_matrix.dat"


qaoa_cirq, sol0, output_dir, figures_dir = Max_cut_CRAB_modulus_update.generate_instance(w_file_path, 'w')

WFF = True

if WFF:
    file = output_dir+'/Warm_start_FOU_OUTPUT_file_tau_64_dt_1Nc30_nreals_5.dat'
else : file = None

taus = [64] 
Ncs = [32]


basis= 'FOU' # basis = 'FOU' or 'CHB'
dt = 1

for tau in taus:
#    Ncs = [2]+list(range(10,int(tau/2)-9,10)) + [int(tau/2)]
#    print(Ncs)

    Max_cut_CRAB_modulus_update.CRAB_main(qaoa_cirq, output_dir, figures_dir, tau, dt, basis, Ncs = Ncs, Warm_start = True, Warm_from_file= WFF, input_file = file, Full = False)

print('Done!')




