import numpy as np
import scipy
import sys
"""
Models for the QAOA algorithm. Models on wich to run the QAOA algorithm. Made by Glen Mbeng. 
"""


#------------------------------------------
# functions for general gate applications
#------------------------------------------
def get_basis(N, flip_sym=False):
    # must be compatible with shaffle_rules
    state_indices = np.arange(2**N, dtype=int)
    bit_vals = ((state_indices.reshape(-1,1) & (2**np.arange(N))) != 0).astype(int)
    bit_vals = bit_vals[:,::-1]  # correct the order of the bits
    #bit_vals = bit_vals[::-1,:] # DO NOT invert the ordering of the rows --> Qiskit convention!
    if flip_sym:
        bit_vals = bit_vals[bit_vals[:,0]==0]

    return bit_vals 

def get_sigmax_rules(basis, flip_sym=False):
    N = basis.shape[1]
    state_indices = np.arange(2 ** N, dtype=int) 
    sigmax_rules =  np.zeros((2 ** N, N), dtype=int)
    for j in range(N):
        # get rule to flip spin j
        j_rev = N-1-j # revert the order of the bits, consistent with get_zbasis
        sigmax_rules[:,j] =  np.bitwise_xor(2 ** (N-1-j) , state_indices)
    if flip_sym:
        full_flip_rule = np.bitwise_and(np.bitwise_not(state_indices), (2**N)-1)
        bit_vals = get_basis(N, flip_sym=False)
        sym_state_indices = np.arange(2 ** (N-1), dtype=int) 
        sym_flags = (bit_vals[:,0] == 0)
        # indices_map1 = np.stack((state_indices[sym_flags], sym_state_indices), axis=1) 
        # indices_map2 = np.stack((full_flip_rule[state_indices[sym_flags]], sym_state_indices), axis=1) 

        indices_map = np.zeros(2 ** N, dtype=int) - 1
        indices_map[state_indices[sym_flags]] = sym_state_indices
        indices_map[full_flip_rule[state_indices[sym_flags]]] = sym_state_indices


        sigmax_rules = sigmax_rules[sym_flags]
        for j in range(N):
            sigmax_rules[:,j] = indices_map[sigmax_rules[:,j]]

    return sigmax_rules

def apply_sigmax(psi_in, j, sigmax_rules, basis):
    psi_out = psi_in[sigmax_rules[:,j]]
    return psi_out

def apply_sigmax_product(psi_in, e, sigmax_rules, basis):
    psi_out = psi_in
    for j in e:
        psi_out = apply_sigmax(psi_out, j, sigmax_rules, basis)
    return psi_out

def apply_golobal_flip(psi_in, sigmax_rules, basis):
    N = basis.shape[1]
    psi_out = psi_in
    for j in range(N):
        psi_out = psi_out[sigmax_rules[:,j]]
    psi_out = np.copy(psi_out)
    return psi_out

def apply_sigmay(psi_in, j, sigmax_rules, basis):
    # bit = 0 -> spin = down
    # bit = 1 -> spin = up
    psi_out = 1j * ((-1)** basis[:,j]) * apply_sigmax(psi_in, j, sigmax_rules, basis)
    return psi_out

def apply_Rx(psi_in, cos_beta, sin_beta, j, sigmax_rules, basis):
    # e^{i beta simga^x_j}
    psi_out = (cos_beta * psi_out + 1j * sin_beta * apply_sigmax(psi_in, j, sigmax_rules, bit_vals))
    return psi_out

class QAOA:
    def __init__(self, N, weighted_edges, h = 1.0, flip_sym=True):
        r"""
        weight_matrix should be an array!!

        Hz = \sum_{i,j} w_{ij} \simga^z_i * \simga_^z_j 
        Hx = -h * \sum_{i} \simga^x_i 
        """

        # hilbert space
        self.N = N        
        self.flip_sym = flip_sym
        self.basis = get_basis(self.N, flip_sym=self.flip_sym)
        self.dimH = len(self.basis)
        self.sigmax_rules = get_sigmax_rules(self.basis, flip_sym=self.flip_sym)

        # set Hz
        self.weighted_edges = weighted_edges
        self.diag_Hz = QAOA._copute_diag_Hz(weighted_edges, self.basis, self.N, self.flip_sym)

        self.Emin = np.min(self.diag_Hz)
        self.Emax = np.max(self.diag_Hz)
        i_sol_list = np.arange(self.dimH)[self.diag_Hz == self.Emin]
        self.solution_list = []
        for i_sol in i_sol_list:
            self.solution_list.append(np.copy(2 * self.basis[i_sol] - 1))


        
        # set Hx
        self.h = h

        # set initial state
        self.psi_start = np.ones(self.dimH) / np.sqrt(self.dimH)

    def _copute_diag_Hz(weighted_edges, basis, N, flip_sym):
        dimH = len(basis)
        sz_config = (2 * basis - 1)

        diag_Hz = np.zeros(dimH)
        for w, e in weighted_edges:
            if flip_sym and ( ( len(e) % 2 ) == 1 ):
                raise ValueError(f"graph is incompatible with the filp symmetry")
            hz = w
            for j in e:
                hz = hz * sz_config[:, j]
            diag_Hz = diag_Hz + hz

        return diag_Hz


    def copute_diag_Op(self, weighted_edges):
        diag_Op = QAOA._copute_diag_Hz(weighted_edges, self.basis, self.N, self.flip_sym)
        return diag_Op

    def apply_Sx_iexp(psi_in, beta, h, basis, sigmax_rules, derivative = False, hermitian_conj = False):
        # use the ideintity: 
        #    e^{i \theta \sigma^x_j} = \cos(\theta) + i\sin(\theta)* \sigma^x_j
        N = basis.shape[1]
        
        psi_out = psi_in
        for j in range(N):
            theta = -1 * (- h[j]) * beta
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            if not hermitian_conj:
                psi_out = cos_theta * psi_out + (1j * sin_theta) * apply_sigmax(psi_out, j, sigmax_rules, basis)
            elif hermitian_conj:
                psi_out = cos_theta * psi_out - (1j * sin_theta) * apply_sigmax(psi_out, j, sigmax_rules, basis)
        
        if derivative:
            psi_in2 = psi_out 
            psi_out =np.zeros(psi_in.shape)
            for j in range(N):
                if not hermitian_conj:
                    psi_out = psi_out +  (1j * (-1) * -h[j]) * apply_sigmax(psi_in2, j, sigmax_rules, basis)
                elif hermitian_conj:
                    psi_out = psi_out +  (-1j * (-1) * -h[j]) * apply_sigmax(psi_in2, j, sigmax_rules, basis)

        return psi_out

    def apply_Ux(self, psi_in, beta, derivative = False, hermitian_conj = False):
        # use the ideintity: 
        #    e^{i \theta \sigma^x_j} = \cos(\theta) + i\sin(\theta)* \sigma^x_j
        
        theta = -1 * (- self.h) * beta
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        psi_out = psi_in
        for j in range(self.N):
            if not hermitian_conj:
                psi_out = cos_theta * psi_out + (1j * sin_theta) * apply_sigmax(psi_out, j, self.sigmax_rules, self.basis)
            elif hermitian_conj:
                psi_out = cos_theta * psi_out - (1j * sin_theta) * apply_sigmax(psi_out, j, self.sigmax_rules, self.basis)
        
        if derivative:            
            if not hermitian_conj:
                psi_out = (1j * (-1)) * self.apply_Hx(psi_out, 1.)
            elif hermitian_conj:
                psi_out = (-1j * (-1)) * self.apply_Hx(psi_out, 1.)

        return psi_out

    def apply_golobal_flip(self, psi_in):
        psi_out = apply_golobal_flip(psi_in, self.sigmax_rules, self.basis)
        return psi_out


    def apply_diagonal_minus_iexp(psi_in, gamma, diag_Op, derivative = False, hermitian_conj = False):
        assert not np.any(np.imag(diag_Op))
        theta = -1 * gamma
        if not hermitian_conj:
            diag_Uz = np.exp(1j * theta * diag_Op)
        elif hermitian_conj:
            diag_Uz = np.exp(-1j * theta * diag_Op)

        psi_out = psi_in * diag_Uz
        if derivative:
            if not hermitian_conj:
                psi_out = (1j * (-1)) * diag_Op * psi_out
            elif hermitian_conj:
                psi_out = (-1j * (-1)) * diag_Op * psi_out

        return psi_out

    def apply_Uz(self, psi_in, gamma, derivative = False, hermitian_conj = False):
        theta = -1 * gamma
        if not hermitian_conj:
            diag_Uz = np.exp(1j * theta * self.diag_Hz)
        elif hermitian_conj:
            diag_Uz = np.exp(-1j * theta * self.diag_Hz)

        psi_out = psi_in * diag_Uz
        if derivative:
            if not hermitian_conj:
                psi_out = (1j * (-1)) * self.apply_Hz(psi_out, 1.)
            elif hermitian_conj:
                psi_out = (-1j * (-1)) * self.apply_Hz(psi_out, 1.)

        return psi_out

    def apply_Uxz(self, psi_in, gamma, beta):
        psi_out = self.apply_Uz(psi_in, gamma)
        psi_out = self.apply_Ux(psi_out, beta)
        return psi_out

    
    def apply_Hz(self, psi_in, gamma):
        psi_out = gamma * self.diag_Hz * psi_in
        return psi_out

    def apply_Hx(self, psi_in, beta):
        psi_out =np.zeros(psi_in.shape)
        for j in range(self.N):
            psi_out = psi_out +  apply_sigmax(psi_in, j, self.sigmax_rules, self.basis)
        psi_out = (beta * (-self.h)) * psi_out
        return psi_out

    def apply_Hxz(self, psi_in, gamma, beta):
        psi_out = self.apply_Hz(psi_in, gamma) + self.apply_Hx(psi_in, beta)
        return psi_out
    
    
    def apply_commutator(self, psi_in, gamma, beta):

        psi_out_left = self.apply_Hz(psi_in, gamma)
        psi_out_left = self.apply_Hx(psi_out_left, beta)

        
        psi_out_right = self.apply_Hx(psi_in, beta)
        psi_out_right = self.apply_Hz(psi_out_right, gamma)


        psi_out = psi_out_left - psi_out_right
        return psi_out
    
        
    def apply_BCH(self, psi_in, gamma, beta, order):
        
        if order > 4: 
            print('Precision not available')
            sys.exit()
        else:
            psi_out_first = self.apply_Hz(psi_in, gamma) + self.apply_Hx(psi_in, beta)
            psi_out = psi_out_first
        
            if order >= 2:
                psi_out_second = self.apply_commutator(psi_in, gamma, beta)
                psi_out = psi_out - 1j*0.5*psi_out_second 
        
            if order >= 3:     
                psi_out_third_11 = self.apply_Hx(psi_out_second, beta)
                psi_out_third_12 = self.apply_Hx(psi_in, beta)
                psi_out_third_12 = self.apply_commutator(psi_out_third_12, gamma, beta)  
                psi_out_third_21 = self.apply_Hz(psi_out_second, gamma)
                psi_out_third_22 = self.apply_Hz(psi_in, gamma)
                psi_out_third_22 = self.apply_commutator(psi_out_third_22, gamma, beta)  
            
                psi_out_third = psi_out_third_11 - psi_out_third_12 - psi_out_third_21 + psi_out_third_22
                psi_out = psi_out + 1./12*psi_out_third
        
            if order == 4:
                psi_out_fourth_11   = gamma*self.apply_Hz(( psi_out_third_11 - psi_out_third_12), gamma)
                psi_out_fourth_12_L = beta*self.apply_Hx(psi_out_third_22,beta)
        
                psi_out_fourth_12_R = gamma*self.apply_Hz(psi_in,gamma)
                psi_out_fourth_12_R = beta*self.apply_Hx(psi_out_fourth_12_R,beta)
                psi_out_fourth_12_R = self.apply_commutator(psi_out_fourth_12_R,gamma,beta)
        
                psi_out_fourth_12 = psi_out_fourth_12_L - psi_out_fourth_12_R
                psi_out_fourth = psi_out_fourth_11 - psi_out_fourth_12
                psi_out = psi_out - 1./24*psi_out_fourth          
                
            return psi_out
    
        #psi_out = psi_out_first +1j*0.5*psi_out_second + 1./12*psi_out_third - 1./24*psi_out_fourth
 
        
    def apply_BCH_dot(self, psi_in, gamma, beta, gamma_dot, beta_dot, order):
        
        if order > 4: 
            print('Precision not available')
            sys.exit()
        else:
            psi_out_first = gamma_dot*self.apply_Hz(psi_in, 1) + beta_dot*self.apply_Hx(psi_in, 1)
            psi_out = psi_out_first
        
            if order >= 2:
                psi_out_second = (beta_dot*gamma + beta*gamma_dot)*self.apply_commutator(psi_in, 1, 1)
                psi_out = psi_out - 1j*0.5*psi_out_second 
        
            if order >= 3:     
                psi_out_third_11 = self.apply_Hx(psi_out_second, 1)
                psi_out_third_12 = self.apply_Hx(psi_in, 1)
                psi_out_third_12 = self.apply_commutator(psi_out_third_12, 1, 1)  
                psi_out_third_21 = self.apply_Hz(psi_out_second, 1)
                psi_out_third_22 = self.apply_Hz(psi_in, 1)
                psi_out_third_22 = self.apply_commutator(psi_out_third_22, 1, 1)  
            
                psi_out_third = (2*beta*beta_dot*gamma + beta**2*gamma_dot)*(psi_out_third_11 - psi_out_third_12) - (beta_dot*gamma**2 + 2*beta*gamma*gamma_dot)*(psi_out_third_21 - psi_out_third_22)
                psi_out = psi_out + 1./12*psi_out_third
        
            if order == 4:
                psi_out_fourth_11 = gamma*self.apply_Hz(( psi_out_third_11 - psi_out_third_12), gamma)
                psi_out_fourth_12_L = beta*self.apply_Hx(psi_out_third_22,beta)
        
                psi_out_fourth_12_R = gamma*self.apply_Hz(psi_in,gamma)
                psi_out_fourth_12_R = beta*self.apply_Hx(psi_out_fourth_12_R,beta)
                psi_out_fourth_12_R = self.apply_commutator(psi_out_fourth_12_R,gamma,beta)
        
                psi_out_fourth_12 = psi_out_fourth_12_L - psi_out_fourth_12_R
                psi_out_fourth = psi_out_fourth_11 - psi_out_fourth_12
                psi_out = psi_out - 1./24*psi_out_fourth          
                
            return psi_out
    
        #psi_out = psi_out_first +1j*0.5*psi_out_second + 1./12*psi_out_third - 1./24*psi_out_fourth
 
    

    def apply_trotter_I(self, gamma, beta, dt, psi_in, herm = False):
        if herm == False:
            psi = self.apply_Uz(psi_in, gamma*dt)
            psi = self.apply_Ux(psi, beta*dt)
        else:
            psi = self.apply_Ux(psi_in, -beta*dt)
            psi = self.apply_Uz(psi, -gamma*dt)

        return psi


    def apply_sym_trotter_II(self, gamma, beta, dt, psi_in):
        psi = self.apply_Uz(psi_in, gamma*dt/2)
        psi = self.apply_Ux(psi, beta*dt)
        psi = self.apply_Uz(psi, gamma*dt/2)
            
        return psi

    
    def apply_sym_trotter_IV(self, gamma, beta, dt, psi_in):

        a = 1/(2-2**(1./3))
        b = - 2**(1./3)/(2-2**(1./3))
        psi = self.apply_Uz(psi_in, a*gamma*dt/2)
        psi = self.apply_Ux(psi, a*beta*dt)
        psi = self.apply_Uz(psi, (a+b)/2*gamma*dt)
        psi = self.apply_Ux(psi, b*beta*dt)
        psi = self.apply_Uz(psi, (a+b)/2*gamma*dt)
        psi = self.apply_Ux(psi, a*beta*dt)
        psi = self.apply_Uz(psi, a*gamma*dt/2)
                
        return psi


        
        
    
    def state(self, gamma_vec, beta_vec):
        P = gamma_vec.shape[0]
        assert gamma_vec.shape == (P,)
        assert beta_vec.shape == (P,)
        
        psi_qaoa = self.psi_start
        for m in range(P):
            gamma = gamma_vec[m]
            beta = beta_vec[m]

            psi_qaoa = self.apply_Uxz(psi_qaoa, gamma, beta)

        return psi_qaoa

    def Hxz_eigsh_np(self, gamma, beta, k=5, eigvecs0=None):
        n = len(self.basis)
        def matvec(psi_in):
            psi_out = self.apply_Hxz(psi_in, gamma, beta)
            return psi_out

        H = scipy.sparse.linalg.LinearOperator(shape=(n,n), matvec=matvec)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(H, k=k, which="SA")
        sorted_args = np.argsort(eigvals)
        eigvals = eigvals[sorted_args]
        eigvecs = (eigvecs.T[sorted_args]).T
        return eigvals, eigvecs

    def Hxz_lin_op(self, gamma, beta):
        n = len(self.basis)
        def matvec(psi_in):
            psi_out = self.apply_Hxz(psi_in, gamma, beta)
            return psi_out

        H = scipy.sparse.linalg.LinearOperator(shape=(n,n), matvec=matvec)
        return H
   
    
    
    def Uxz_eigs_np(self, gamma, beta, k=5, which='SM', eigvecs0=None):
        n = len(self.basis)
        def matvec(psi_in):
            psi_out = self.apply_Uxz(psi_in, gamma, beta)
            return psi_out

        print("diagonalize")
        sigma = None
        Uxz = scipy.sparse.linalg.LinearOperator(shape=(n,n), matvec=matvec)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(Uxz, k=k, which=which, sigma=sigma)
        # sort

        # orthogonalize egvectors
        # a == U @ S @ Vh
        # a @ Vh^\dag S^{-1}== U 
        print("orthogonalize")
        U_svd, S_svd, Vh_svd = scipy.linalg.svd(eigvecs, full_matrices=False)
        eigvecs_orth = eigvecs @ Vh_svd.T.conj() @ np.diag(1/S_svd)
        eigvals_orth = eigvals.reshape(1, -1) @ Vh_svd.T.conj() @ np.diag(1/S_svd)
        eigvals_orth = eigvals_orth.reshape(-1)

        print("sort")
#        sorted_args = np.argsort(np.abs(eigvals_orth - sigma))
        sorted_args = np.argsort(np.abs(eigvals_orth))
        eigvals_orth = eigvals_orth[sorted_args]
        eigvecs_orth = (eigvecs_orth.T[sorted_args]).T

        return eigvals_orth, eigvecs_orth

    def Heff_eigsh_np(self, gamma, beta, k, order, eigvecs0=None):
        n = len(self.basis)
        def matvec(psi_in):
            psi_out = self.apply_BCH(psi_in, gamma, beta, order)
            return psi_out

        Heff = scipy.sparse.linalg.LinearOperator(shape=(n,n), matvec=matvec)
        eigvals, eigvecs = scipy.sparse.linalg.eigsh(Heff, k=k, which="SA")
        sorted_args = np.argsort(eigvals)
        eigvals = eigvals[sorted_args]
        eigvecs = (eigvecs.T[sorted_args]).T
        return eigvals, eigvecs
    
    

    def energy_expect(self, psi, gamma, beta):
        psi_r = self.apply_Hz(psi, gamma) + self.apply_Hx(psi, beta)
        psi_l = psi
        norm = np.linalg.norm(psi)
        E = np.sum(psi_l.conj() * psi_r) / (norm ** 2)
        return np.real(E)

    def E_fun(self, x):
        x = x.reshape(-1)
        P = x.shape[0]//2
        gamma_vec = x[:P]
        beta_vec = x[P:]        
        r_rho_qaoa = self.state(gamma_vec, beta_vec)
        E_qaoa = self.energy_expect(r_rho_qaoa, 1., 0)
        return E_qaoa

    def E_with_grad_fun(self, x, check_gradient = False):
        x = x.reshape(-1)
        P = x.shape[0]//2


        gamma_vec = x[:P]
        beta_vec = x[P:]       
        gamma_targ = 1.
        beta_targ = 0

        # set the operator sequence
        op_seq = []
        theta_vec = np.zeros(x.shape)
        for m in range(len(x)):
            i_step = m // 2
            if m % 2 == 0:
                theta_vec[m] = gamma_vec[i_step]
                op_seq.append(self.apply_Uz)
            elif m % 2 == 1:
                theta_vec[m] = beta_vec[i_step]
                op_seq.append(self.apply_Ux)

        # build the left vectors
        psi_left = np.zeros((len(theta_vec) + 1, self.dimH), dtype=complex)
        psi_left[0] = self.psi_start
        for m in range(len(theta_vec)):
            psi_left[m + 1] = op_seq[m](psi_left[m], theta_vec[m], hermitian_conj=False, derivative=False)
        m = len(theta_vec)


        psi_right = np.zeros((len(theta_vec) + 1, self.dimH), dtype=complex)
        psi_right[0] = self.apply_Hz(psi_left[-1], gamma_targ) + self.apply_Hx(psi_left[-1], beta_targ)
        for m in range(len(theta_vec)):
            psi_right[m + 1] = op_seq[-m-1](psi_right[m], theta_vec[-m-1], hermitian_conj=True, derivative=False)

        # compute energy
        E = np.real(np.dot(np.conj(psi_right[0]), psi_left[-1]))

        # compute gradient with resect to angles
        g_theta = np.zeros(theta_vec.shape)
        for m in range(len(theta_vec)):
            g_theta[m] = 2 * np.real(np.dot(
                np.conj(psi_right[-1-1-m]),
                op_seq[m](psi_left[m], theta_vec[m], hermitian_conj=False, derivative=True)
                ))


        # gradient with respect to input x
        g_gamma = g_theta[::2]
        g_beta = g_theta[1::2]
        g_x = np.concatenate((g_gamma, g_beta))

        if check_gradient:
            discrate_g_x = np.zeros(x.shape)
            for j in range(len(x)):
                dx = np.zeros(x.shape)
                dx[j] = 1e-5
                discrate_g_x[j] = (self.E_fun(x+dx) - self.E_fun(x)) / dx[j]

            print(f"g_x..... = {g_x}")
            print(f"disc_g_x = {discrate_g_x}")
        return E, g_x


    def E_with_grad_from_opseq_fun(self, apply_H, theta_vec, op_seq, psi_start = None, check_gradient = False):
        if psi_start is None:
            psi_start = self.psi_start

        theta_vec = theta_vec.reshape(-1)
        assert len(op_seq) == len(theta_vec)

        # build the left vectors
        psi_left = np.zeros((len(theta_vec) + 1, self.dimH), dtype=complex)
        psi_left[0] = psi_start
        for m in range(len(theta_vec)):
            psi_left[m + 1] = op_seq[m](psi_left[m], theta_vec[m], hermitian_conj=False, derivative=False)
        m = len(theta_vec)


        psi_right = np.zeros((len(theta_vec) + 1, self.dimH), dtype=complex)
        psi_right[0] = apply_H(psi_left[-1], 1.)

        for m in range(len(theta_vec)):
            psi_right[m + 1] = op_seq[-m-1](psi_right[m], theta_vec[-m-1], hermitian_conj=True, derivative=False)

        # compute energy
        psi_qaoa = psi_left[-1]
        E = np.real(np.dot(np.conj(psi_right[0]), psi_left[-1]))

        # compute gradient with resect to angles
        g_theta = np.zeros(theta_vec.shape)
        for m in range(len(theta_vec)):
            g_theta[m] = 2 * np.real(np.dot(
                np.conj(psi_right[-1-1-m]),
                op_seq[m](psi_left[m], theta_vec[m], hermitian_conj=False, derivative=True)
                ))

        if check_gradient:
            discrate_g_theta = np.zeros(theta_vec.shape)
            for j in range(len(theta_vec)):
                dtheta = np.zeros(theta_vec.shape)
                dtheta[j] = 1e-5
                discrate_g_theta[j] = (self.E_fun(theta_vec+dtheta) - self.E_fun(theta_vec)) / dtheta[j]

            print(f"g_x..... = {g_theta}")
            print(f"disc_g_x = {discrate_g_theta}")
        return E, g_theta, psi_qaoa



##########################
# Methods to creator
class PhaseSeparationGate():
    def __init__(self, qaoa_cirq, weighted_edges):
        # new arguments are positioned at the end
        diag_Op = QAOA._copute_diag_Hz(weighted_edges, qaoa_cirq.basis, qaoa_cirq.N, qaoa_cirq.flip_sym)
        self.diag_Op = diag_Op
        self.weighted_edges = weighted_edges

    def __call__(self, psi_in, gamma, derivative = False, hermitian_conj = False):
        psi_out = QAOA.apply_diagonal_minus_iexp(psi_in, gamma, self.diag_Op, derivative=derivative, hermitian_conj=hermitian_conj)
        return psi_out    

class MixerGate():
    def __init__(self, qaoa_cirq, h):
        # new arguments are positioned at the end
        self.h = h
        self.basis = qaoa_cirq.basis
        self.sigmax_rules = qaoa_cirq.sigmax_rules

    def __call__(self, psi_in, beta, derivative = False, hermitian_conj = False):
        psi_out = QAOA.apply_Sx_iexp(psi_in, beta, self.h, self.basis, self.sigmax_rules, derivative=derivative, hermitian_conj=hermitian_conj)
        return psi_out    

class DiagonalHamiltonian():
    def __init__(self, qaoa_cirq, diag_Op):
        # new arguments are positioned at the end
        self.diag_Op = diag_Op

    def __call__(self, psi_in, gamma):
        psi_out = gamma * self.diag_Op * psi_in
        return psi_out    


######### functions to optimize #############
def loss_func(x, qaoa_cirq, apply_Htg, op_seq, psi_start, jac=True):
    theta = x
    E, g_theta, psi_qaoa = qaoa_cirq.E_with_grad_from_opseq_fun(apply_Htg, theta, op_seq, psi_start=psi_start)
    if jac:
        return E, g_theta
    else:
        return E


def optimimize_angles(psi_start, op_seq, qaoa_cirq, nruns, niter, method="BFGS", apply_Htg=None, theta_guess=None):
    
    if apply_Htg is None:
        apply_Htg = qaoa_cirq.apply_Hz
    
    gtol = 1e-6
    ftol = 1e-6

    res_list = []
    theta_ranges = len(op_seq) * [[0, 10 * np.pi]]
    if theta_guess is None:
        theta_guess = 2 * np.pi * np.random.rand(len(op_seq))
    for i_run in range(nruns):
        if i_run == 0:
            x0 = theta_guess
        else:
            x0 = theta_guess + 2 * np.pi * np.random.rand(len(op_seq))
        
        if method == "BFGS":
            jac = True        
            minimizer_kwargs={"args":(qaoa_cirq, apply_Htg, op_seq, psi_start, jac), "jac":jac, "method":'BFGS', "options":{'gtol': gtol, 'disp': False}}
            res = scipy.optimize.basinhopping(loss_func, x0, niter=niter, minimizer_kwargs=minimizer_kwargs)
        elif method == "Nelder-Mead":
            jac = False
            minimizer_kwargs={"args":(qaoa_cirq, apply_Htg, op_seq, psi_start, jac), "method":'Nelder-Mead', "options":{'ftol': ftol, 'disp': False}}
            res = scipy.optimize.basinhopping(loss_func, x0, niter=niter, minimizer_kwargs=minimizer_kwargs)
        elif method == "brute":
            jac = False
            x, fval, grid, Jout  = scipy.optimize.brute(loss_func, theta_ranges, args=(qaoa_cirq, apply_Htg, op_seq, psi_start, jac), Ns=niter,full_output=True)
            res = scipy.optimize.OptimizeResult({"x":x, "fun":fval})
        res_list.append(res)
        
    E_list = np.array([res.fun for res in res_list])
    theta_list = np.array([res.x for res in res_list])
    i_opt = np.argmin(E_list)
    E_opt = E_list[i_opt]
    res_opt = res_list[i_opt]
    theta_opt = res_opt.x
    
    E, g_theta, psi_qaoa = qaoa_cirq.E_with_grad_from_opseq_fun(apply_Htg, theta_opt, op_seq, psi_start=psi_start)
    return E_opt, theta_opt, psi_qaoa, E_list, theta_list


def analize_state(psi, diag_Hz):
    distrib = np.abs(psi) ** 2
    Emin = np.min(diag_Hz)
    Emax = np.max(diag_Hz)
    Ps = np.sum(distrib[diag_Hz == np.min(diag_Hz)])

    E = np.sum(distrib*diag_Hz)
    eres = (E - Emin) / (Emax - Emin)
    return E, eres, distrib, Ps







if __name__ == "__main__":
    import qubo_lib
    N = 3
    P = 6
    flip_sym = True
    basis = get_basis(N, flip_sym=flip_sym)
    sigmax_rules = get_sigmax_rules(basis, flip_sym=flip_sym)


    print("basis:\n", basis)

    # apply sigma x
    vec = (1. + 0j) * np.arange(len(basis))
    print("vec:\n", vec)
    j0 = 1
    print(f"apply_sigmax(vec, {j0}, sigmax_rules, basis):\n", apply_sigmax(vec, j0, sigmax_rules, basis))
    print(f"apply_sigmax(vec, {j0}, sigmax_rules, basis):\n", apply_sigmay(vec, j0, sigmax_rules, basis))
    # set 1d Ferro ising (PBC)
    w_matrix = np.zeros((N,N))
    for j in range(N): 
        w_matrix[j, (j+1)%N] = -1
    
    # get qubo graph
    wgraph = qubo_lib.QUBO_graph(w_matrix)
    weighted_edges = wgraph.weighted_edges
    print("qubo graph:")
    wgraph.print()

    w_file_path = f"w3r_N14_Zhou/weight_matrix.dat"
    sol_file_path = f"w3r_N14_Zhou/classical_solutions.dat"
    import load_data_lib
    N, weight_matrix = load_data_lib.load_maxcut_weight_matrix(w_file_path)
    N, num_solutions, soultions = load_data_lib.load_maxcut_classical_solutions(sol_file_path)
    sol1 = soultions[:, 0]
    E_maxcut_sol = np.dot(sol1, np.dot(weight_matrix, sol1))
    Hz_graph = qubo_lib.QUBO_graph(weight_matrix)
    print("E_maxcut_sol = ", E_maxcut_sol)

    # intialize QAOA_cirquit
    qaoa_circuit = QAOA(Hz_graph)

    # get a qaoa state
    # get a qaoa state
    x = 0.1 * (1 + np.arange(2 * P)) # np.pi * np.array([0.2, 0.3])
    x = 2 * np.pi * np.random.rand(2 * P)
    gamma_vec = x[:P]
    beta_vec = x[P:]
    psi_qaoa = qaoa_circuit.state(gamma_vec, beta_vec)
    print("gamma_vec :\n", gamma_vec)
    print("beta_vec :\n", beta_vec) 
    print("qaoa_state :\n", psi_qaoa)
    print()


    gamma = 1
    beta = 0
    print("gamma: ", gamma)
    print("beta: ", beta)
    print("energy = ", qaoa_circuit.energy_expect(psi_qaoa, gamma, beta))
    qaoa_circuit.E_with_grad_fun(x, check_gradient=True)

