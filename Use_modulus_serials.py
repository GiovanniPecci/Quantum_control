import Max_cut_CRAB_modulus_update
import os

curr_dir = os.getcwd()

proj_dir = os.path.dirname(curr_dir)+'/'

wm_dir = proj_dir+"weight_matrices/"


for wm in os.listdir(wm_dir):
    print(wm)
    w_file_path = wm_dir + wm
    string = wm.split('_')[0]

    qaoa_cirq, sol0, output_dir, figures_dir = Max_cut_CRAB_modulus_update.generate_instance(w_file_path, string)

    taus = [2] 

    basis= 'FOU' # basis = 'FOU' or 'CHB'
    dt = 1

    for tau in taus:
        Ncs = [int(tau/2)]
        print(Ncs)
        Max_cut_CRAB_modulus_update.CRAB_main(qaoa_cirq, output_dir, figures_dir, tau, dt, basis, Ncs = Ncs, Warm_start = True, Full = False, RAND = True)

    print('Done!')




