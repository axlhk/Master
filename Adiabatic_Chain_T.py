import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from numba import njit
import time 


class Kitaev:
    "dt     :       timestep in tau"
    "tmax   :       Total physical time T"
    "N      :       Number of sites"
    "Stride :       How often the loops print progress"
    "Hamiltonian:   Kitaev or BdG"
    "Integration:   (Euler, Euler_Cromer) or RK4"
    "Evolution:     Adiabatic, Adiabatic_precomputed, Schrodinger, Schrodinger_precomputed"
    "omega, mu, delta: functions(j, N, t, tmax)" 
    "return_full:   True or False, to return all a_t's, or just the gs."
    "trunc_dim:     Keep trunc_dim amount of lowest energy states in Adiabatic evolution"
    "warm_start:    True or False, to use the previous solution to speed up the diagonalizations"
    "HamRepresentation = dense or sparse. How the Hamiltonian is represented. Typically use sparse"
    "n      :       Sites on each arm (including center site)"

    def __init__(self, dt, n, stride, tmax,
                 Hamiltonian, Integration, Evolution,
                 omega_fun, delta_fun, mu_fun,
                 exp_degen, return_full,
                 trunc_dim=None,
                 warm_start=True,
                 HamRepresentation="dense",
                 buffer_sites = 1,
                 geometry = "T"):

        # N depends on geometry
        if geometry == "T":
            # old convention: n = sites per arm (incl center), N = 3n - 2
            self.N = 3*n - 2
        elif geometry == "chain":
            # new convention: n = chain length, N = n
            self.N = n
        else:
            raise ValueError(f"Unknown geometry: {geometry}. Use 'T' or 'chain'.")

        self.geometry = geometry
        self.dt   = dt
        self.n    = n
        self.stride = stride
        self.tmax = tmax
        self.buffer_sites = buffer_sites

        self.exp_degen = exp_degen
        self.return_full = return_full
        self.warm_start = warm_start

        self.omega_fun = omega_fun
        self.delta_fun = delta_fun
        self.mu_fun    = mu_fun

        # For timing
        self.time_build_H = 0.0
        self.time_diag_H  = 0.0
        self.time_evolution_loop = 0.0
        self.time_precompute = 0.0

        # Precomputed-data (for Adiabatic_precomputed)
        self.tau_grid = None
        self.E_store  = None
        self.V_store  = None

        # Choose Hamiltonian
        self.ham_rep = HamRepresentation  # "dense" or "sparse"
        self.ham_type = Hamiltonian   
    
        if Hamiltonian == "Kitaev":
            self.dim = 2**self.N
            self.ntot = precompute_ntot(self.N)
            self.ham_is_sparse = (HamRepresentation == "sparse")
            
            if self.ham_rep == "dense":
                self.ham_builder = H_Kitaev_T
                self.ham_is_sparse = False
            elif self.ham_rep == "sparse":
                self.ham_builder = H_Kitaev_sparse_cached_T
                self.ham_is_sparse = True
            else:
                raise ValueError(
                    f"Unknown HamRepresentation: {self.ham_rep}. "
                    "Use 'dense' or 'sparse'."
                )
                
        elif Hamiltonian == "BdG":
            raise ValueError(f"Unknown Hamiltonian type: {Hamiltonian}. Use Kitaev ")

        #Trunc_dim for adiabatic evolutions 
        if trunc_dim is None:
            self.trunc_dim = self.dim
        else:
            self.trunc_dim = min(trunc_dim, self.dim)

        # Choose integrator for the evolution type
        if Integration == "RK4":
            self.stepper = evolution_step_RK4
        else:
            raise ValueError(f"Unknown integrator: {Integration}. Use either Euler or RK4. Euler_Cromer not yet implemented.")

        #Evolution type
        self.evolution_type = Evolution
        if Evolution == "Adiabatic":
            self.run_method = self.run_adiabatic
        elif Evolution == "Adiabatic_precomputed":
            self.run_method = self._run_adiabatic_precomputed_main
        elif Evolution == "Schrodinger":
            self.run_method = self._run_schrodinger
        elif Evolution == "Schrodinger_precomputed":
            self.run_method = self.run_schrodinger_with_precomputed
        else:
            raise ValueError(
                f"Unknown evolution_type: {Evolution}. "
                "Use Adiabatic, Adiabatic_precomputed or Schrodinger."
            )
        
        # geometry placeholders
        self.bonds_j = None
        self.bonds_k = None
        self.bond_arm = None
        self.bond_depth = None
        
    def run(self, T):    # For easy calling
        return self.run_method(T)

    def build_geometry(self):
        """
        Dispatch to the appropriate geometry builder based on self.geometry.
        """
        if self.geometry == "T":
            self.build_T_geometry()
        elif self.geometry == "chain":
            self.build_chain_geometry()
        else:
            raise ValueError(f"Unknown geometry: {self.geometry}")


    def build_chain_geometry(self):
        """
        Build site and bond structure for a 1D chain of length n (self.n).
        Geometry (site numbering):
            sites: 0, 1, ..., N-1 (N = n)

        Bonds:
            (0,1), (1,2), ..., (N-2, N-1)
        We still define 'arm' and 'depth' so that we can reuse H_Kitaev_T
        and the (arm, depth) dependent protocols.
          - arm[b]   = 0  for all bonds (single arm)
          - depth[b] = bond index = 0..N-2
        """
        N = self.N  # = self.n when geometry == "chain"

        bonds_j_list = []
        bonds_k_list = []
        bond_arm_list = []
        bond_depth_list = []

        for depth in range(N - 1):
            j = depth
            k = depth + 1
            bonds_j_list.append(j)
            bonds_k_list.append(k)
            bond_arm_list.append(0)      # single arm
            bond_depth_list.append(depth)

        self.bonds_j = np.array(bonds_j_list, dtype=np.int32)
        self.bonds_k = np.array(bonds_k_list, dtype=np.int32)
        self.bond_arm = np.array(bond_arm_list, dtype=np.int32)
        self.bond_depth = np.array(bond_depth_list, dtype=np.int32)
    
    def build_T_geometry(self):
        """
        Build the site and bond structure for a T-junction with three arms of length n
        (including the central site).

        Geometry (site numbering):
            - central (junction) site: 0
            - arm A: 0 1 2 ... (n-1)
            - arm B: 0 n (n+1) ... (2n-2)
            - arm C: 0 (2n-1) (2n) ... (3n-3)
            
        Return bonds as arrays of (j,k) pairs, and label each bond by:
            - arm_id[b]   in {0,1,2}   (0 = arm A, 1 = arm B, 2 = arm C)
            - depth[b]    in {0,..., n-2}, where depth = 0 is the bond touching the center.
        """
        
        n = self.n

        bonds_j_list = []
        bonds_k_list = []
        bond_arm_list = []
        bond_depth_list = []

        # Arm A: sites 0,1,...,n-1
        arm_id = 0
        for depth in range(n-1):  # bonds (0,1), (1,2), ..., (n-2,n-1)
            j = depth
            k = depth + 1
            bonds_j_list.append(j)
            bonds_k_list.append(k)
            bond_arm_list.append(arm_id)
            bond_depth_list.append(depth)

        # Arm B: sites 0,n,...,2n-2
        arm_id = 1
        startB = n
        # bond touching center:
        bonds_j_list.append(0)
        bonds_k_list.append(startB)
        bond_arm_list.append(arm_id)
        bond_depth_list.append(0)
        # rest:
        for depth in range(1, n-1):
            j = startB + depth - 1
            k = startB + depth
            bonds_j_list.append(j)
            bonds_k_list.append(k)
            bond_arm_list.append(arm_id)
            bond_depth_list.append(depth)

        # Arm C: sites 0,2n-1,...,3n-3
        arm_id = 2
        startC = 2*n - 1
        # bond touching center:
        bonds_j_list.append(0)
        bonds_k_list.append(startC)
        bond_arm_list.append(arm_id)
        bond_depth_list.append(0)
        # rest:
        for depth in range(1, n-1):
            j = startC + depth - 1
            k = startC + depth
            bonds_j_list.append(j)
            bonds_k_list.append(k)
            bond_arm_list.append(arm_id)
            bond_depth_list.append(depth)

        self.bonds_j = np.array(bonds_j_list, dtype=np.int32)
        self.bonds_k = np.array(bonds_k_list, dtype=np.int32)
        self.bond_arm = np.array(bond_arm_list, dtype=np.int32)
        self.bond_depth = np.array(bond_depth_list, dtype=np.int32)

    def Diag_H(self, H):
        """
        Diagonalise Hamiltonian H (dense or sparse).
        """
        t0 = time.perf_counter()
        if not isinstance(H, np.ndarray):
            # assume sparse matrix
            H = H.toarray()
        E, V = np.linalg.eigh(H)
        t1 = time.perf_counter()
        self.time_diag_H += (t1 - t0)
        return E, V
    
    def build_H(self, t):
        t0 = time.perf_counter()

        if self.ham_type == "Kitaev":
            if self.ham_rep == "dense":
                H = H_Kitaev_T(
                    self.N, t, self.tmax,
                    self.omega_fun, self.delta_fun, self.mu_fun,
                    self.ntot,
                    self.bonds_j, self.bonds_k,
                    self.bond_arm, self.bond_depth,
                    self.n, self.buffer_sites
                )
            else:
                # in sparse case, we normally build H via precompute,
                # so build_H is mainly used in the dense path
                Nb = self.bonds_j.shape[0]
                # simple on-the-fly couplings; for performance you'd rather precompute
                omega_b = np.empty(Nb, dtype=np.float64)
                delta_b = np.empty(Nb, dtype=np.complex128)
                for b in range(Nb):
                    arm = self.bond_arm[b]
                    depth = self.bond_depth[b]
                    omega_b[b] = self.omega_fun(b, arm, depth, t, self.tmax, self.n, self.buffer_sites)
                    delta_b[b] = self.delta_fun(b, arm, depth, t, self.tmax, self.n, self.buffer_sites)

                # site-dependent mu
                mu_j = np.empty(self.N, dtype=np.float64)
                for j in range(self.N):
                    mu_j[j] = self.mu_fun(j, t, self.tmax, self.n, self.buffer_sites)

                H = H_Kitaev_sparse_cached_T(self.N, omega_b, delta_b, mu_j, self.ntot, self.bonds_j, self.bonds_k)
        else:
            raise ValueError(f"Unknown Hamiltonian type: {self.ham_type}")

        t1 = time.perf_counter()
        self.time_build_H += (t1 - t0)
        return H

    def initial_eigenpairs(self, T):
        H0 = self.build_H(T[0])
        E0, V0 = self.Diag_H(H0)

        H1 = self.build_H(T[1])
        E1, V1 = self.Diag_H(H1)

        return E0, V0, E1, V1

    def prepare_precomputed(self, Nt_tau):
        """
        Run precompute once, and save on this instance.
        Has to be called before Adiabatic_precomputed is used.
        """
        print("Precomputing eigenpairs along tau (sparse + eigsh)...")
        t0 = time.perf_counter()
        tau_grid, E_store, V_store = self.precompute_eigensystem_tau(Nt_tau)
        self.tau_grid = tau_grid
        self.E_store  = E_store
        self.V_store  = V_store
        t1 = time.perf_counter()
        print(f"Done precomputing.\n time = {t1 - t0}")

    def precompute_eigensystem_tau(self, Nt_tau):
        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        dim_trunc = self.trunc_dim
        dim_full = self.dim

        Nb = self.bonds_j.shape[0]

        E_store = np.zeros((Nt_tau, dim_trunc))
        V_store = np.zeros((Nt_tau, dim_full, dim_trunc), dtype=complex)

        # Precompute couplings for all tau on the T-graph
        omega_tab, delta_tab, mu_tab = precompute_couplings_Kitaev_T(
            self.N, tau_grid, 1.0,
            Nb, self.bond_arm, self.bond_depth,
            self.n, self.buffer_sites,
            self.omega_fun, self.delta_fun, self.mu_fun
        )

        t0_all = time.perf_counter()

        v0 = None
        for idx, tau in enumerate(tau_grid):
            t_dummy = tau

            if self.ham_type == "Kitaev":
                mu_j = mu_tab[idx]
                H_sparse = H_Kitaev_sparse_cached_T(
                    self.N,
                    omega_tab[idx],   # length Nb
                    delta_tab[idx],   # length Nb
                    mu_j,
                    self.ntot,
                    self.bonds_j,
                    self.bonds_k
                )
            else:
                raise ValueError("precompute_eigensystem_tau only implemented for Kitaev here")

            #Diagonalize lowest trunc_dim eigenvals
            # (eigsh does'nt spit them out sorted).  
            try:
                if self.warm_start and v0 is not None:
                    E_part, V_part = eigsh(H_sparse, k=dim_trunc, which="SA",
                                           v0=v0, tol=1e-7)
                else:
                    E_part, V_part = eigsh(H_sparse, k=dim_trunc, which="SA",
                                           tol=1e-7)
            except ArpackNoConvergence as err:
                print(f"[Precompute] ARPACK did not converge at idx={idx}, tau={tau}. "
                      f"Using converged {len(err.eigenvalues)} eigenpairs.")
                E_part = err.eigenvalues
                V_part = err.eigenvectors
                if E_part.shape[0] < dim_trunc:
                    dim_here = E_part.shape[0]
                    E_tmp = np.full(dim_trunc, np.nan, dtype=float)
                    V_tmp = np.zeros((dim_full, dim_trunc), dtype=complex)
                    E_tmp[:dim_here] = E_part
                    V_tmp[:, :dim_here] = V_part
                    E_part, V_part = E_tmp, V_tmp

            #Sort by energy
            idx_sorted = np.argsort(E_part)
            E_sorted = E_part[idx_sorted]
            V_sorted = V_part[:, idx_sorted]

            E_store[idx] = E_sorted
            V_store[idx] = V_sorted

            # Warm start
            if self.warm_start:
                v0 = V_sorted[:, 0]

            if (idx % 10 == 0):
                print(f"[Precompute] Step {idx} / {Nt_tau-1}. tau = {tau:.3f}")

        t1_all = time.perf_counter()
        self.time_precompute += (t1_all - t0_all)
        
        return tau_grid, E_store, V_store
    
    def run_adiabatic_precomputed(self, T, E_store, V_store):
        """
        Adiabatic evolution where eigenpairs (E,V) are precomputed on the sime time grid which T is based on

        Requires:
        - len(T) == E_store.shape[0] == V_store.shape[0]
        - E_store[n], V_store[n] tilsvarer eigenparene ved T[n].
        """
        dt  = self.dt
        dim_full = self.dim
        dim_trunc = E_store.shape[1]
        Nt  = len(T)

        if Nt != E_store.shape[0]:
            raise ValueError("T og E_store/V_store må ha samme lengde")

        a_t = np.zeros((Nt, dim_trunc), dtype=complex)
        a_t[0, 0] = 1.0

        b_t = np.zeros((Nt, dim_trunc), dtype=complex)
        b_t[0] = a_t[0]

        E_all = np.zeros((Nt, dim_trunc))

        #Find the physical gs at t = 0 in the full basis
        H0_full = self.build_H(0.0)             
        E0_full, V0_full = self.Diag_H(H0_full)
        psi0 = V0_full[:, 0]                    
        # preallocer P0_orig_phys
        P0_orig_phys = np.zeros(Nt, dtype=float)
        P0_orig_phys[0] = 1.0                   # <psi0|psi0>^2 = 1

        # Initial eigenpairs (truncated)
        E0_raw = E_store[0]
        V0_raw = V_store[0]     # (dim_full, dim_trunc)

        E1_raw = E_store[1]
        V1_raw = V_store[1]

        V1_matched, perm01 = match_eigenvectors(V0_raw, V1_raw)
        E1 = E1_raw[perm01]
        V1_fixed = fix_phases(V0_raw, V1_matched)

        V0 = V0_raw
        V1 = V1_fixed
        E0 = E0_raw

        E_all[0] = E0
        E_all[1] = E1

        phase_int = np.zeros((dim_trunc, dim_trunc), dtype=complex)

        A = np.empty((dim_trunc, dim_trunc), dtype=complex)
        lambda_vec = np.empty(dim_trunc, dtype=complex)
        B_mat = np.empty((dim_trunc, dim_trunc), dtype=complex)  

        # n = 0: forward difference
        A0 = psi_dotpsi_forward(V0, V1, dt)
        lambda0 = lambda_from_E_and_A(E0, A0)
        a_t[1], phase_int = self.stepper(a_t[0], phase_int, A0, lambda0, dt)

        #Calculate physical psi(t1) and project onto psi0
        I_k_1 = -phase_int[0, :]
        phase_factors_1 = np.exp(-1j * I_k_1)
        a_phys_1 = phase_factors_1 * a_t[1]     # phys. coeff in instantaneous basis at t1
        psi_trunc_1 = V1 @ a_phys_1             # |psi(t1)> in full basis (truncated)
        P0_orig_phys[1] = np.abs(np.vdot(psi0, psi_trunc_1))**2

        V_prev, V_curr = V0, V1
        E_curr = E1

        t_loop_start = time.perf_counter()

        # Time evolution
        for n in range(1, Nt - 1):
            #Get raw (unsorted) eigenpairs at next step
            E_next_raw = E_store[n+1]
            V_next_raw = V_store[n+1]

            #Match and fix phases relative to current basis
            V_next_matched, perm = match_eigenvectors(V_curr, V_next_raw)
            E_next = E_next_raw[perm]
            V_next = fix_phases(V_curr, V_next_matched)

            central_A_inplace(V_prev, V_curr, V_next, dt, A)
            lambda_from_E_and_A_inplace(E_curr, A, lambda_vec)

            #Adiabatic step in truncated basis
            a_t[n+1], phase_int = self.stepper(a_t[n], phase_int, A, lambda_vec, dt)

            #Reconstruct physical amplitudes in original basis
            I_k = -phase_int[0, :]
            phase_factors = np.exp(-1j * I_k)
            a_phys = phase_factors * a_t[n+1]

            B_mat[:] = V0.conj().T @ V_curr
            b_t[n+1] = B_mat @ a_phys

            #Physical projection onto psi0
            psi_trunc = V_curr @ a_phys
            P0_orig_phys[n+1] = np.abs(np.vdot(psi0, psi_trunc))**2

            #Rotate for next iteration
            V_prev, V_curr = V_curr, V_next
            E_curr = E_next

            E_all[n+1] = E_next

            if self.stride and (n % self.stride == 0):
                print(f"[Adiabatic-precomputed] Step {n} / {Nt-1}. T = {self.tmax}")


        t_loop_end = time.perf_counter()
        self.time_evolution_loop += (t_loop_end - t_loop_start)

        # Reduction to exp_degen lowest in trunc_dim
        k = min(self.exp_degen, dim_trunc)
        a_t_reduced = np.zeros((Nt, k), dtype=complex)
        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]
            idx_sorted = np.argsort(E)
            idx_gs = idx_sorted[:k]
            a_t_reduced[n, :] = d[idx_gs]

        # --- returner også P0_orig_phys ---
        if self.return_full:
            return T, a_t_reduced, b_t, a_t, E_all, P0_orig_phys
        else:
            return T, a_t_reduced, b_t, P0_orig_phys

    def _run_adiabatic_precomputed_main(self, T_dummy):
        """
        Runs when Evolution='Adiabatic_precomputed'.
        Ignore T_dummy. Using precomputed tau_grid/E_store/V_store.
        """
        if self.tau_grid is None or self.E_store is None or self.V_store is None:
            raise RuntimeError(
                "Precomputed eigensystem mangler. "
                "Kall prepare_precomputed(Nt_tau) før du bruker Adiabatic_precomputed."
            )

        # Physical time grid for this tmax
        T = self.tau_grid * self.tmax
        dt_phys = T[1] - T[0]

        # used by stepper
        dt_old = self.dt
        self.dt = dt_phys

        result = self.run_adiabatic_precomputed(T, self.E_store, self.V_store)

        self.dt = dt_old
        return result

    def run_adiabatic(self, T):
        """
        Run time evolution on time grid T (full dim).
        """
        dt  = self.dt
        dim = self.dim
        Nt  = len(T)

        a_t = np.zeros((Nt, dim), dtype=complex)
        a_t[0, 0] = 1.0

        b_t = np.zeros((Nt, dim), dtype=complex)
        b_t[0] = a_t[0]
        E_all = np.zeros((Nt, dim))

        # initial eigenpairs
        H0 = self.build_H(T[0])
        E0_raw, V0_raw = self.Diag_H(H0)

        H1 = self.build_H(T[1])
        E1_raw, V1_raw = self.Diag_H(H1)

        V1_matched, perm01 = match_eigenvectors(V0_raw, V1_raw)
        E1 = E1_raw[perm01]
        V1_fixed = fix_phases(V0_raw, V1_matched)
        V0 = V0_raw
        V1 = V1_fixed
        E0 = E0_raw

        E_all[0] = E0
        E_all[1] = E1

        phase_int = np.zeros((dim, dim), dtype=complex)

        # n = 0 step: forward difference for A0
        A0 = psi_dotpsi_forward(V0, V1, dt)
        lambda0 = lambda_from_E_and_A(E0, A0)

        a_t[1], phase_int = self.stepper(a_t[0], phase_int, A0, lambda0, dt)

        V_prev, V_curr = V0, V1
        E_curr = E1

        t_loop_start = time.perf_counter()

        for n in range(1, Nt - 1):
            t_next = T[n + 1]
            H_next = self.build_H(t_next)
            E_next_raw, V_next_raw = self.Diag_H(H_next)

            V_next_matched, perm = match_eigenvectors(V_curr, V_next_raw)
            E_next = E_next_raw[perm]
            V_next_fixed = fix_phases(V_curr, V_next_matched)
            V_next = V_next_fixed

            E_all[n + 1] = E_next

            A_n = central_A(V_prev, V_curr, V_next, dt)
            lambda_n = lambda_from_E_and_A(E_curr, A_n)

            a_t[n + 1], phase_int = self.stepper(a_t[n], phase_int, A_n, lambda_n, dt)

            V_prev, V_curr = V_curr, V_next
            E_curr = E_next

            # reconstruct physical amplitudes in original basis
            I_k = -phase_int[0, :]
            phase_factors = np.exp(-1j * I_k)
            a_phys = phase_factors * a_t[n+1]

            B = V0.conj().T @ V_curr
            b_t[n+1] = B @ a_phys

            if self.stride and (n % self.stride == 0):
                print(f"[Adiabatic] Step {n} / {Nt-1}. T = {self.tmax}")
        t_loop_end = time.perf_counter()

        self.time_evolution_loop += (t_loop_end - t_loop_start)

        k = self.exp_degen
        a_t_reduced = np.zeros((Nt, k), dtype=complex)
        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]
            idx_sorted = np.argsort(E)
            idx_gs = idx_sorted[:k]
            a_t_reduced[n, :] = d[idx_gs]

        return T, a_t_reduced, b_t
    
    def prepare_precomputed_dense(self, Nt_tau):
        """
        Dense precompute of E(tau), V(tau) for tau in [0,1].
        This always uses the chosen Hamiltonian type (Kitaev or BdG),
        independent of HamRepresentation ("dense"/"sparse").
        """
        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        dim = self.dim

        E_store = np.zeros((Nt_tau, dim))
        V_store = np.zeros((Nt_tau, dim, dim), dtype=complex)

        t0_all = time.perf_counter()
        for idx, tau in enumerate(tau_grid):

            if self.ham_type == "Kitaev":
                # use the Kitaev Hamiltonian (full Fock space)
                H = H_Kitaev_T(self.N, tau, 1.0,
                            self.omega_fun, self.delta_fun, self.mu_fun,
                            self.ntot,
                            self.bonds_j, self.bonds_k,
                            self.bond_arm, self.bond_depth,
                            self.n, self.buffer_sites)
            else:
                raise ValueError(f"Unknown Hamiltonian type: {self.ham_type}")

            E, V = self.Diag_H(H)

            E_store[idx] = E
            V_store[idx] = V

            if self.stride and (idx % 10 == 0):
                print(f"[Precompute dense] Step {idx} / {Nt_tau-1}. tau = {tau:.3f}")

        t1_all = time.perf_counter()
        self.time_precompute += (t1_all - t0_all)
        self.tau_grid = tau_grid
        self.E_store_dense = E_store
        self.V_store_dense = V_store

    def precompute_couplings(self, Nt_tau):
        """
        Helper for plotting tau, omega, delta, mu.
        Returns:
        tau_grid : (Nt_tau,)
        omega_tab : (Nt_tau, Nb)    # bonds
        delta_tab : (Nt_tau, Nb)    # bonds
        mu_tab    : (Nt_tau, N)     # sites
        """
        if self.bonds_j is None:
            raise RuntimeError("Call build_geometry() before precompute_couplings().")

        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        Nb = self.bonds_j.shape[0]

        omega_tab, delta_tab, mu_tab = precompute_couplings_Kitaev_T(
            self.N, tau_grid, 1.0,
            Nb, self.bond_arm, self.bond_depth,
            self.n, self.buffer_sites,
            self.omega_fun, self.delta_fun,
            self.mu_fun,   # NEW
        )
        return tau_grid, omega_tab, delta_tab, mu_tab



"Functions used in run_adiabatic() in class"
def psi_dotpsi_forward(V0, V1, dt):
    overlap_01 = V0.conj().T @ V1
    A0 = (overlap_01 - np.eye(V0.shape[1], dtype=complex)) / dt
    return A0

def lambda_from_E_and_A(E, A):
    return E - 1j * np.diag(A)

def match_eigenvectors(V_curr, V_next_raw):
    dim = V_curr.shape[1]
    O = V_curr.conj().T @ V_next_raw

    used_next = set()
    perm = [-1] * dim
    for k in range(dim):
        overlaps = np.abs(O[k, :])
        for l in used_next:
            overlaps[l] = -1.0
        l_best = np.argmax(overlaps)
        perm[k] = l_best
        used_next.add(l_best)

    V_next_perm = V_next_raw[:, perm]
    return V_next_perm, perm

def fix_phases(V_curr, V_next):
    dim = V_curr.shape[1]
    for k in range(dim):
        overlap = np.vdot(V_curr[:, k], V_next[:, k])
        phase = np.angle(overlap)
        V_next[:, k] *= np.exp(-1j * phase)
    return V_next

def central_A(V_prev, V_curr, V_next, dt):
    overlap_n_next = V_curr.conj().T @ V_next
    overlap_n_prev = V_curr.conj().T @ V_prev
    A_n = (overlap_n_next - overlap_n_prev) / (2 * dt)
    return A_n

def central_A_inplace(V_prev, V_curr, V_next, dt, out):
    overlap_n_next = V_curr.conj().T @ V_next   # (dim_trunc, dim_trunc)
    overlap_n_prev = V_curr.conj().T @ V_prev
    out[:] = (overlap_n_next - overlap_n_prev) / (2 * dt)

def lambda_from_E_and_A_inplace(E, A, out):
    out[:] = E - 1j * np.diag(A)

"Kitaev Hamiltonian and helper functions (full + sparse)"
@njit
def bit(n, i):
    return (n >> i) & 1

@njit
def flip_bit(n, i):
    return n ^ (1 << i)

@njit
def fermion_sign(n, i):
    mask = (1 << i) - 1
    x = n & mask
    cnt = 0
    while x:
        cnt += x & 1
        x >>= 1
    return -1 if (cnt % 2 == 1) else 1

@njit
def a_dag_on_basis_state(n, i):
    if bit(n, i) == 1:
        return 0.0, -1
    s = fermion_sign(n, i)
    m = flip_bit(n, i)
    return s, m

@njit
def a_on_basis_state(n, i):
    if bit(n, i) == 0:
        return 0.0, -1
    s = fermion_sign(n, i)
    m = flip_bit(n, i)
    return s, m

@njit
def precompute_ntot(N):
    dim = 2**N
    ntot_arr = np.zeros(dim, dtype=np.int32)
    for n in range(dim):
        cnt = 0
        x = n
        while x:
            cnt += x & 1
            x >>= 1
        ntot_arr[n] = cnt
    return ntot_arr

@njit
def H_Kitaev_T(N, t, tmax,
                   omega_fun,  
                   delta_fun,                    
                   mu_fun,     
                   ntot_arr,
                   bonds_j, bonds_k,
                   bond_arm, bond_depth, n, buffer_sites):
    """
    Kitaev Hamiltonian defined by bonds_j, bonds_k.

    H = - sum_j mu_j(t) (n_j - 1/2)
        - sum_b omega_b(t) (c_j^dagger c_k + c_k^dagger c_j)
        + sum_b [Delta_b(t) c_j c_k + Delta_b^*(t) c_k^dagger c_j^dagger],

    where each bond b connects sites (j,k) = (bonds_j[b], bonds_k[b]).

    We allow omega_fun and delta_fun to depend on the bond label (arm, depth) to
    implement spatially local topological/trivial regions and their motion.
    """
    dim = 2**N
    Nb = bonds_j.shape[0]

    H = np.zeros((dim, dim), dtype=np.complex128)

    for n_state in range(dim):
        ntot = ntot_arr[n_state]

        #Site dependent mu
        diag_mu = 0.0
        for j in range(N):
            mu_j = mu_fun(j, t, tmax, n, buffer_sites)
            n_j = bit(n_state, j)  # 0 or 1
            diag_mu += - mu_j * (n_j - 0.5)
        H[n_state, n_state] += diag_mu

        # Bond terms
        for b in range(Nb):
            j = bonds_j[b]
            k = bonds_k[b]
            arm = bond_arm[b]
            depth = bond_depth[b]

            w = omega_fun(b, arm, depth, t, tmax, n, buffer_sites)
            d = delta_fun(b, arm, depth, t, tmax, n, buffer_sites)

            # hopping: -w (a_j^† a_k + a_k^† a_j)

            # a_j^† a_k
            a1, m1 = a_on_basis_state(n_state, k)
            if m1 != -1:
                a2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    H[m2, n_state] += -w * a1 * a2

            # a_k^† a_j
            a3, m3 = a_on_basis_state(n_state, j)
            if m3 != -1:
                a4, m4 = a_dag_on_basis_state(m3, k)
                if m4 != -1:
                    H[m4, n_state] += -w * a3 * a4

            # pairing: d a_j a_k + d^* a_k^† a_j^†

            # a_j a_k
            a5, m5 = a_on_basis_state(n_state, k)
            if m5 != -1:
                a6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    H[m6, n_state] += d * a5 * a6

            # c_k^† c_j^†
            a7, m7 = a_dag_on_basis_state(n_state, j)
            if m7 != -1:
                a8, m8 = a_dag_on_basis_state(m7, k)
                if m8 != -1:
                    H[m8, n_state] += d.conjugate() * a7 * a8

    return H

@njit
def precompute_couplings_Kitaev_T(
    N, tau_grid, tmax, Nb,
    bond_arm, bond_depth,
    n, buffer_sites,
    omega_fun, delta_fun,
    mu_fun,                # NEW: mu_fun(site, t, tmax)
):
    Nt = len(tau_grid)

    omega_tab = np.empty((Nt, Nb), dtype=np.float64)
    delta_tab = np.empty((Nt, Nb), dtype=np.complex128)
    mu_tab    = np.empty((Nt, N), dtype=np.float64)  # site-based

    for it in range(Nt):
        tau = tau_grid[it]
        t = tau * tmax

        # bonds: omega, delta
        for b in range(Nb):
            arm = bond_arm[b]
            depth = bond_depth[b]
            omega_tab[it, b] = omega_fun(b, arm, depth, t, tmax, n, buffer_sites)
            delta_tab[it, b] = delta_fun(b, arm, depth, t, tmax, n, buffer_sites)

        # sites: mu
        for j in range(N):
            mu_tab[it, j] = mu_fun(j, t, tmax, n, buffer_sites)

    return omega_tab, delta_tab, mu_tab


def H_Kitaev_sparse_cached_T(N, omega_b, delta_b, mu_j,
                                 ntot_arr, bonds_j, bonds_k):
    """
    Sparse Kitaev Hamiltonian on a general graph.

    Arguments
    ---------
    N          : number of sites
    mu         : scalar chemical potential (can be made site-dependent if needed)
    omega_b    : array of length Nb with hopping amplitude on each bond.
    delta_b    : array of length Nb with pairing amplitude on each bond.
    ntot_arr   : precomputed particle number in each basis state.
    bonds_j,k  : arrays of int, endpoints of each bond.

    Returns
    -------
    H_sparse   : csr_matrix of shape (2^N, 2^N)
    """
    dim = 2**N
    Nb = bonds_j.shape[0]

    rows = []
    cols = []
    data = []

    for n_state in range(dim):
        # diagonal chemical potential term: sum over sites
        diag_mu = 0.0
        for j in range(N):
            n_j = bit(n_state, j)
            diag_mu += - mu_j[j] * (n_j - 0.5)

        # diagonal chemical potential term
        rows.append(n_state)
        cols.append(n_state)
        data.append(diag_mu)

        # off-diagonal bond terms
        for b in range(Nb):
            j = int(bonds_j[b])
            k = int(bonds_k[b])
            w = omega_b[b]
            d = delta_b[b]

            # a_j^† a_k
            a1, m1 = a_on_basis_state(n_state, k)
            if m1 != -1:
                a2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    val = -w * a1 * a2
                    if val != 0.0:
                        rows.append(m2)
                        cols.append(n_state)
                        data.append(val)

            # a_k^† a_j
            a3, m3 = a_on_basis_state(n_state, j)
            if m3 != -1:
                a4, m4 = a_dag_on_basis_state(m3, k)
                if m4 != -1:
                    val = -w * a3 * a4
                    if val != 0.0:
                        rows.append(m4)
                        cols.append(n_state)
                        data.append(val)

            # a_j a_k
            a5, m5 = a_on_basis_state(n_state, k)
            if m5 != -1:
                a6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    val = d * a5 * a6
                    if val != 0.0:
                        rows.append(m6)
                        cols.append(n_state)
                        data.append(val)

            # a_k^† a_j^†
            a7, m7 = a_dag_on_basis_state(n_state, j)
            if m7 != -1:
                a8, m8 = a_dag_on_basis_state(m7, k)
                if m8 != -1:
                    val = np.conjugate(d) * a7 * a8
                    if val != 0.0:
                        rows.append(m8)
                        cols.append(n_state)
                        data.append(val)

    H_sparse = csr_matrix((np.array(data, dtype=np.complex128),
                           (np.array(rows, dtype=np.int32),
                            np.array(cols, dtype=np.int32))),
                          shape=(dim, dim))
    return H_sparse


"Adiabatic integrators and helper functions"
def rhs_phase(a, phase_int, A, lambd):
    phase_matrix = np.exp(1j * phase_int)
    M = phase_matrix * A
    np.fill_diagonal(M, 0.0)

    da_dt = -M @ a
    delta_lambda = lambd[:, None] - lambd[None, :]
    dI_dt = delta_lambda

    return da_dt, dI_dt

def evolution_step_RK4(a_n, phase_int, A, lambd, dt):
    k1_a, k1_I = rhs_phase(a_n, phase_int, A, lambd)

    a2 = a_n + 0.5 * dt * k1_a
    I2 = phase_int + 0.5 * dt * k1_I
    k2_a, k2_I = rhs_phase(a2, I2, A, lambd)

    a3 = a_n + 0.5 * dt * k2_a
    I3 = phase_int + 0.5 * dt * k2_I
    k3_a, k3_I = rhs_phase(a3, I3, A, lambd)

    a4 = a_n + dt * k3_a
    I4 = phase_int + dt * k3_I
    k4_a, k4_I = rhs_phase(a4, I4, A, lambd)

    a_next = a_n + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
    phase_int_next = phase_int + (dt / 6.0) * (k1_I + 2*k2_I + 2*k3_I + k4_I)

    return a_next, phase_int_next

mu0 = 0.5
delta0 = 2.0
omega0 = 1.0

"Time dependencies"

# --- Cutting a 1D chain protocol ---

@njit
def omega_Eivind_4_2(j, N, t, tmax):
    j_c = N//2  -1 #-1 to cut the central bond
    if j == j_c:
        return 2 * omega0 * np.abs(t/tmax - 0.5) 
    else:
        return omega0

@njit
def mu_Eivind_4_2(t, tmax, n, buffer_sites):
    return mu0  

@njit
def delta_Eivind_4_2(j, N, t, tmax):
    j_c = N//2  -1 #-1 to cut the central bond
    if j == j_c:
        return 2 * delta0 * np.abs(t/tmax - 0.5)
    else:
        return delta0

@njit
def omega_chain_cut(b, arm, depth, t, tmax, n, buffer_sites):
    """
    Wrapper for omega_Eivind_4_2 on a 1D chain.
    Here:
      - n = chain length N
      - depth = bond index j
    """
    N = n
    j = depth
    return omega_Eivind_4_2(j, N, t, tmax)

@njit
def delta_chain_cut(b, arm, depth, t, tmax, n, buffer_sites):
    """
    Wrapper for delta_Eivind_4_2 on a 1D chain.
    """
    N = n
    j = depth
    return delta_Eivind_4_2(j, N, t, tmax)

@njit
def mu_chain_cut(site, t, tmax, n, buffer_sites):
    """
    Uniform mu in time and space for the chain, as specified.
    """
    return mu_Eivind_4_2(t, tmax)


# --- T-junction protocols ---

@njit
def mu_T(site, t, tmax, n, buffer_sites):
    return mu0

@njit
def smooth_step(x):
    """
    Simple clipped linear step:
    0 for x <= 0, 1 for x >= 1, linear in between.
    """
    if x <= 0.0:
        return 0.0
    elif x >= 1.0:
        return 1.0
    else:
        return x

@njit
def smooth_ramp(tau, t0, t1, v0, v1):
    """
    Smoothly ramp from v0 at tau=t0 to v1 at tau=t1 using smooth_step.
    Outside [t0,t1] return v0 (tau<t0) or v1 (tau>t1).
    """
    if tau <= t0:
        return v0
    if tau >= t1:
        return v1
    x = (tau - t0) / (t1 - t0)  # in [0,1]
    s = smooth_step(x)
    return v0 + (v1 - v0) * s


@njit
def topo_slide_one_step(arm, depth, s, arm_target, d_in, d_out):
    """
    Grow a topological segment on arm_target inward by one bond, without shrinking:
      initial: depths [d_in, d_out] = 1
      final:   depths [d_in-1, d_out] = 1

    s in [0,1]:
      - s=0:  [d_in..d_out] = 1, depth d_in-1 = 0
      - 0<s<1: depth d_in-1 ramps 0 -> 1, [d_in..d_out] stay 1
      - s=1:  [d_in-1..d_out] = 1

    This keeps the segment contiguous and never turns any existing bond off.
    """
    if arm != arm_target:
        return 0.0

    # Clamp s
    if s <= 0.0:
        if depth >= d_in and depth <= d_out:
            return 1.0
        else:
            return 0.0

    if s >= 1.0:
        if depth >= d_in - 1 and depth <= d_out:
            return 1.0
        else:
            return 0.0

    # 0 < s < 1
    if depth == d_in - 1:
        # new inner bond: 0 -> 1
        x = s  # in (0,1)
        return smooth_step(x)

    if depth >= d_in and depth <= d_out:
        # original segment stays fully 1
        return 1.0

    return 0.0


@njit
def static_topo_with_buffer(depth, buffer_sites):
    # topological outside buffer, trivial inside
    if depth < buffer_sites:
        return 0.0
    return 1.0

@njit
def weight_move(arm, depth, tau_local, arm_from, arm_to, n):
    """
    tau_local in [0,1] controls a single move of the topological segment
    from arm_from to arm_to.

    For each depth (bond distance from center):
    - Source (arm_from):
        outer bonds turn off first, inner bonds last.
    - Target (arm_to):
        inner bonds turn on first, outer bonds last.
    """

    max_depth = n - 2
    movable_min = 0
    movable_max = max_depth
    n_steps = movable_max - movable_min + 1
    if n_steps <= 0:
        return 0.0

    depth_idx = depth - movable_min            # 0 .. n_steps-1

    # ---- Source arm: outer -> inner ----
    if arm == arm_from:
        # depth_idx = 0 is inner; we want it to be LAST.
        # So define depth_rel such that outer (large depth_idx) has small depth_rel.
        depth_rel = (movable_max - depth_idx) / max(1, n_steps - 1)  # 0 for outermost, 1 for innermost

        switch_center = 0.1 + 0.8 * depth_rel  # outer switches early, inner late
        switch_width  = 0.5 / n_steps

        if tau_local < switch_center - switch_width:
            return 1.0
        if tau_local > switch_center + switch_width:
            return 0.0

        # in transition window: map tau_local linearly to [0,1], downwards
        x = (switch_center + switch_width - tau_local) / (2.0 * switch_width)
        return smooth_step(x)

    # ---- Target arm: inner -> outer ----
    if arm == arm_to:
        # For the target we want the opposite: inner bonds on first, outer last.
        # So depth_rel = 0 at inner; 1 at outer is fine.
        depth_rel = depth_idx / max(1, n_steps - 1)  # 0 at inner, 1 at outer

        switch_center = 0.1 + 0.8 * depth_rel  # inner switches early, outer late
        switch_width  = 0.5 / n_steps

        if tau_local < switch_center - switch_width:
            return 0.0
        if tau_local > switch_center + switch_width:
            return 1.0

        x = (tau_local - (switch_center - switch_width)) / (2.0 * switch_width)
        return smooth_step(x)

    # other arms: trivial during this move
    return 0.0

@njit
def topo_weight_move(arm, depth, tau, n, buffer_sites):
    """
    'move_arm' protocol:
      - At tau = 0: arms A(0) and B(1) are topological (outside buffer),
                    arm C(2) trivial.
      - For tau in [0,1]: move the topological segment from B(1) to C(2),
                          while A(0) stays topological.

    We use a sliding substage on B that takes a fixed fraction of this
    protocol, independent of absolute tau.
    """

    arm_A = 0
    arm_B = 1
    arm_C = 2

    # A stays static topological (buffered) for the whole protocol
    if arm == arm_A:
        return static_topo_with_buffer(depth, buffer_sites)

    # Use the whole tau ∈ [0,1] as a single "stage" with stage-local coordinate s
    s = tau  # stage-local parameter in [0,1]

    # Choose fraction of THIS stage used to slide B inward
    slide_frac = 0.2  # e.g. first 20% of tau is sliding

    if s < slide_frac:
        # ---- Sliding substage on B: only B is active here, C remains trivial ----

        if arm == arm_C:
            # C stays trivial during the slide
            return 0.0

        # At the beginning of the slide, the B segment is [buffer_sites .. n-2]
        d_in = buffer_sites
        d_out = n - 2

        # Map s ∈ [0, slide_frac] -> s_slide ∈ [0,1] for the sliding function
        s_slide = s / slide_frac

        return topo_slide_one_step(arm, depth, s_slide,
                                   arm_target=arm_B,
                                   d_in=d_in, d_out=d_out)

    # ---- After slide: use remaining part of the stage to transfer B -> C ----
    # At s = slide_frac, B's segment has been shifted inward by one bond.
    s_transfer = (s - slide_frac) / (1.0 - slide_frac)  # in [0,1]

    # At tau very close to 0, ensure B initially matches static_topo_with_buffer
    if tau < 1e-8 and arm == arm_B:
        return static_topo_with_buffer(depth, buffer_sites)

    # At tau very close to 1, ensure final C matches static_topo_with_buffer
    if tau > 1.0 - 1e-8 and arm == arm_C:
        return static_topo_with_buffer(depth, buffer_sites)

    # Main B -> C transfer (no buffer enforced here, because we WANT the center bond)
    return weight_move(arm, depth, s_transfer,
                       arm_from=arm_B, arm_to=arm_C, n=n)

@njit
def delta_move(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_move(arm, depth, tau, n, buffer_sites)
    return delta0 * w

@njit
def omega_move(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_move(arm, depth, tau, n, buffer_sites)
    return omega0 * w

@njit
def topo_weight_half_circuit(arm, depth, tau, n, buffer_sites):
    """
    'half_circuit' protocol:
      - Initially: arms A(0) and B(1) topological (outside buffer), C trivial.
      - Stage 1: move B -> C (with smooth growth of B depth 0).
      - Stage 2: move A -> B (with smooth growth of A depth 0).
      - Stage 3: move C -> A.
    Ends with arms B and C topological (swapped compared to start).
    """

    arm_A = 0
    arm_B = 1
    arm_C = 2

    tau1_end = 1.0 / 3.0
    tau2_end = 2.0 / 3.0

    # ------------------------
    # Stage 1: B -> C, A static
    # ------------------------
    if tau < tau1_end:
        # A static: buffered topological
        if arm == arm_A:
            return static_topo_with_buffer(depth, buffer_sites)

        # Stage-local coordinate in [0,1]
        s = tau / tau1_end

        slide_frac = 0.5  # fraction of Stage 1 used to slide B inward

        if s < slide_frac:
            # Slide B inward: [buffer..n-2] -> [buffer-1..n-3]
            if arm == arm_C:
                # C trivial during slide
                return 0.0

            d_in = buffer_sites
            d_out = n - 2
            s_slide = s / slide_frac
            return topo_slide_one_step(arm, depth, s_slide,
                                       arm_target=arm_B,
                                       d_in=d_in, d_out=d_out)
        else:
            # After slide: B->C transfer
            s_transfer = (s - slide_frac) / (1.0 - slide_frac)
            return weight_move(arm, depth, s_transfer,
                               arm_from=arm_B, arm_to=arm_C, n=n)

    # ------------------------
    # Stage 2: A -> B, C static
    # ------------------------
    if tau < tau2_end:
        # C static: buffered topological
        if arm == arm_C:
            return static_topo_with_buffer(depth, buffer_sites)

        # Stage-local coordinate in [0,1]
        s = (tau - tau1_end) / (tau2_end - tau1_end)

        slide_frac = 0.5  # fraction of Stage 2 used to slide A inward

        if s < slide_frac:
            # Slide A inward: [buffer..n-2] -> [buffer-1..n-3]
            d_in = buffer_sites
            d_out = n - 2
            s_slide = s / slide_frac
            return topo_slide_one_step(arm, depth, s_slide,
                                       arm_target=arm_A,
                                       d_in=d_in, d_out=d_out)
        else:
            # After slide: A->B transfer
            s_transfer = (s - slide_frac) / (1.0 - slide_frac)
            return weight_move(arm, depth, s_transfer,
                               arm_from=arm_A, arm_to=arm_B, n=n)

    # ------------------------
    # Stage 3: C -> A, B static
    # ------------------------
    # Stage 3 uses weight_move directly: by now C already touches the center
    s = (tau - tau2_end) / (1.0 - tau2_end)  # in [0,1]

    # B static: buffered
    if arm == arm_B:
        return static_topo_with_buffer(depth, buffer_sites)

    return weight_move(arm, depth, s,
                       arm_from=arm_C, arm_to=arm_A, n=n)

@njit
def delta_half_circuit(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_half_circuit(arm, depth, tau, n, buffer_sites)
    return delta0 * w

@njit
def omega_half_circuit(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_half_circuit(arm, depth, tau, n, buffer_sites)
    return omega0 * w

@njit
def topo_weight_full_circuit(arm, depth, tau, n, buffer_sites):
    """
    'full_circuit' protocol: half_circuit done twice.
    """
    if tau < 0.5:
        tau_local = 2.0 * tau          # first half_circuit
        return topo_weight_half_circuit(arm, depth, tau_local, n, buffer_sites)
    else:
        tau_local = 2.0 * (tau - 0.5)  # second half_circuit
        return topo_weight_half_circuit(arm, depth, tau_local, n, buffer_sites)

@njit
def delta_full_circuit(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_full_circuit(arm, depth, tau, n, buffer_sites)
    return delta0 * w

@njit
def omega_full_circuit(b, arm, depth, t, tmax, n, buffer_sites):
    tau = t / tmax
    w = topo_weight_full_circuit(arm, depth, tau, n, buffer_sites)
    return omega0 * w

# --- T-chain "left/right switch" protocol: vary mu on T-junction ---
@njit
def smoothstep_poly(q):
    "Function from dibajyoti"
    if q <= 0.0:
        return 0.0
    elif q >= 1.0:
        return 1.0
    # r(q) = q^2 (3 - 2q)
    return q*q*(3.0 - 2.0*q)

@njit
def mu_leg_keyboard(x, t, tmax, L, tau, alpha, mu_triv, mu_topo, reverse):
    """
    Implements Eq. (A2) Dibajyoti on a single leg of length L, site index x in [0,L-1].

    reverse = False: trivial -> topo (use r)
    reverse = True : topo -> trivial (use 1 - r)
    """
    # dimensionless total protocol time in Eq. A2:
    # q = t / [tau * (1 + alpha (L-1))] - alpha x
    q = (t / (tau * (1.0 + alpha*(L-1)))) - alpha*float(x)
    s = smoothstep_poly(q)
    if reverse:
        s = 1.0 - s
    return mu_triv + (mu_topo - mu_triv) * s

mu_topo = 0.05
mu_triv = 8.0

@njit
def site_to_leg_and_x(site, n):
    """
    Map global site index to (leg, x) with x=0 at center.

    leg: 0 = left(A), 1 = right(B), 2 = bottom(C)
    """
    if site == 0:
        return 0, 0  # treat center as left-leg x=0 (we'll override if needed)

    # left: 1..n-1
    if 1 <= site <= n-1:
        return 0, site  # x = site

    # right: n..2n-2 => shift by (n-1)
    if n <= site <= 2*n-2:
        return 1, site - (n-1)

    # bottom: 2n-1..3n-3 => shift by (2n-1)
    if 2*n-1 <= site <= 3*n-3:
        return 2, site - (2*n-1)

    # default: put anywhere (shouldn't happen for valid T-geometry)
    return 0, 0

@njit
def mu_T_piano(site, t, tmax, n, buffer_sites):
    """
    Keyboard-style chemical potential protocol on a T-junction.

    - n: arm length (including center), must match the 'n' used in the model.
    - mu_topo = 0.05, mu_triv = 8.0 as in the article.
    - omega and delta are kept constant; only mu varies.
    """
    # YOU MUST KEEP THIS CONSISTENT WITH YOUR MODEL
    L = n   # leg length in 'x' (we use x=0..n-1)
    alpha = 1.0   # site-by-site delay; alpha ≳ 1 => sequential
    tau_leg = tmax / 4.0   # characteristic ramp time per leg; tune as needed

    # Map site -> (leg, x)
    leg, x = site_to_leg_and_x(site, n)

    # Total protocol time split into two stages:
    #   Stage 1: move boundary on right leg, t ∈ [0, tmax/2]
    #   Stage 2: restore right leg & move boundary on left leg, t ∈ [tmax/2, tmax]
    half = 0.5 * tmax

    # Keep bottom leg always topological:
    if leg == 2:
        return mu_topo

    # Option: keep center always topological
    if site == 0:
        return mu_topo

    if t < half:
        # -------- Stage 1: move trivial region along the RIGHT leg towards center --------
        if leg == 1:
            # Right leg: topo -> trivial starting from outer sites (x large) going inward (x small).
            # "reverse=True" means we ramp from topo to triv.
            # To have outer sites change first, we can reinterpret x -> (L-1-x) if desired.
            # Simple version: treat x=0 near center, x=L-1 outer; alpha>0 ramps x=0 first.
            # For "outer first", use an effective index x' = (L-1 - x):
            x_eff = (L - 1) - x
            # Local time for this stage: t ∈ [0,half] mapped to [0, 2*tau_leg] for flexibility
            t_local = t
            return mu_leg_keyboard(x_eff, t_local, half, L, tau_leg, alpha,
                                   mu_triv, mu_topo, reverse=True)
        else:
            # Left leg stays fully topological in Stage 1
            return mu_topo

    else:
        # -------- Stage 2: restore right leg to topo, move trivial region into LEFT leg --------
        t_stage2 = t - half

        if leg == 1:
            # Right leg: trivial -> topo (we "heal" it) with the same keyboard form.
            # Now we can ramp inner sites first or outer first; choose inner first:
            x_eff = x
            return mu_leg_keyboard(x_eff, t_stage2, half, L, tau_leg, alpha,
                                   mu_triv, mu_topo, reverse=False)

        elif leg == 0:
            # Left leg: topo -> trivial starting from inner sites (near center) outwards.
            x_eff = x  # x=0 center, x increases outward
            return mu_leg_keyboard(x_eff, t_stage2, half, L, tau_leg, alpha,
                                   mu_triv, mu_topo, reverse=True)

        else:
            # bottom (already caught) or others: topo
            return mu_topo


@njit
def omega_T_chain_switch(b, arm, depth, t, tmax, n, buffer_sites):
    omega0 = 1
    return omega0

@njit
def delta_T_chain_switch(b, arm, depth, t, tmax, n, buffer_sites):
    delta0 = 1
    return delta0

def bond_x_positions(model):
    """
    Map each bond to a 1D x-coordinate: x = arm*(n-1) + depth.
    Returns:
      x : (Nb,) array of ints
    """
    n = model.n
    Nb = model.bonds_j.shape[0]
    x = np.empty(Nb, dtype=int)
    for b in range(Nb):
        arm = model.bond_arm[b]
        depth = model.bond_depth[b]
        x[b] = arm * (n - 1) + depth
    return x

def adiabatic_approx(
    geometry="T",
    protocol="full_circuit",
):
    """
    Unified driver:
      - geometry: "T" (T-junction) or "chain" (1D chain of length n)
      - protocol:
          * "move_arm", "half_circuit", "full_circuit" (T-junction, existing)
          * "cut_chain" (1D chain: cutting central bond)
          * "T_chain_switch" (T-junction: left/right arm switch via mu)
    """

    # --- common parameters ---
    n = 4          # For T: arm length including center site; for chain: total length N
    buffer = 1
    if geometry == "T":
        N = 3*n - 2
    elif geometry == "chain":
        N = n
    else:
        raise ValueError(f"Unknown geometry: {geometry}")

    dt_tau = 1e-3
    Nt_tau = int(1.0/dt_tau) + 1
    # Nt_tau = 1000
    dt_tau = 1 / (Nt_tau - 1) 
    stride = 10

    tmax_list = [0.01, 0.1, 0.5, 1, 5, 10, 20]
    trunc = 10

    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic_precomputed"
    exp_degen = trunc
    return_full = False
    Hamrep = "sparse"

    # --- choose time-dependent couplings and mu based on protocol and geometry ---

    # default: T-junction movement protocols (your existing ones)
    omega_fun = None
    delta_fun = None
    mu_fun    = None

    if geometry == "T":
        # T-junction geometry
        if protocol == "move_arm":
            omega_fun = omega_move
            delta_fun = delta_move
            mu_fun    = mu_T
        elif protocol == "T_piano":
            omega_fun = omega_T_chain_switch  # or define a trivial wrapper returning omega0
            delta_fun = delta_T_chain_switch  # or constant delta0
            mu_fun    = mu_T_piano
        elif protocol == "full_circuit":
            omega_fun = omega_full_circuit
            delta_fun = delta_full_circuit
            mu_fun    = mu_T
        elif protocol == "T_chain_switch":
            # NEW: switching left/right arm purely via mu
            omega_fun = omega_T_chain_switch
            delta_fun = delta_T_chain_switch
            mu_fun    = mu_T_piano
        else:
            raise ValueError(f"Unknown protocol for T geometry: {protocol}")

    elif geometry == "chain":
        # 1D chain geometry
        if protocol == "cut_chain":
            # NEW: cutting a chain using your Eivind_4_2 functions
            omega_fun = omega_chain_cut
            delta_fun = delta_chain_cut
            mu_fun    = mu_chain_cut
        else:
            raise ValueError(f"Unknown protocol for chain geometry: {protocol}")

    # --- construct model ---
    model = Kitaev(
        dt=dt_tau,
        n=n,
        stride=stride,
        tmax=1.0,
        Hamiltonian=Hamiltonian_type,
        Integration=Integrator,
        Evolution=Evolution_type,
        omega_fun=omega_fun,
        delta_fun=delta_fun,
        mu_fun=mu_fun,
        exp_degen=exp_degen,
        return_full=return_full,
        trunc_dim=trunc,
        HamRepresentation=Hamrep,
        buffer_sites=buffer,
        geometry=geometry,   # NEW
    )

    model.build_geometry()

    _ = model.build_H(0.0)

    model.prepare_precomputed(Nt_tau)

    tau_grid, omega_tab, delta_tab, mu_tab = model.precompute_couplings(Nt_tau)

    np.save(f"{geometry}_{protocol}_tau_run_trunc={trunc}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", tau_grid)
    np.save(f"{geometry}_{protocol}_omega_run_trunc={trunc}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", omega_tab)
    np.save(f"{geometry}_{protocol}_delta_run_trunc={trunc}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", delta_tab)
    np.save(f"{geometry}_{protocol}_mu_run_trunc={trunc}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", mu_tab)

    x = bond_x_positions(model)             # shape (Nb,)
    np.save(f"{geometry}_{protocol}_x_run_trunc={trunc}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", x)

    plt.figure(figsize=(12,4))

    # --- omega ---
    plt.subplot(1,3,1)
    plt.imshow(
        omega_tab,
        aspect='auto', origin='lower',
        extent=[x.min(), x.max(), tau_grid[0], tau_grid[-1]]
    )
    plt.colorbar(label=r"$\omega$")
    plt.xlabel("bond index")
    plt.ylabel(r"$\tau$")
    plt.title(r"$\omega(b,\tau)$")

    # --- |Delta| ---
    plt.subplot(1,3,2)
    plt.imshow(
        np.abs(delta_tab),
        aspect='auto', origin='lower',
        extent=[x.min(), x.max(), tau_grid[0], tau_grid[-1]]
    )
    plt.colorbar(label=r"$|\Delta|$")
    plt.xlabel("bond index")
    plt.ylabel(r"$\tau$")
    plt.title(r"$|\Delta(b,\tau)|$")

    # --- mu(j, tau) ---
    plt.subplot(1,3,3)
    site_indices = np.arange(model.N)
    plt.imshow(
        mu_tab,
        aspect='auto', origin='lower',
        extent=[site_indices.min(), site_indices.max(), tau_grid[0], tau_grid[-1]]
    )
    plt.colorbar(label=r"$\mu$")
    plt.xlabel("site index")
    plt.ylabel(r"$\tau$")
    plt.title(r"$\mu(j,\tau)$")

    plt.tight_layout()
    plt.savefig(
        f"{geometry}_omega_delta_mu_heatmap_{protocol}_{len(tmax_list)}_"
        f"{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_"
        f"{Integrator}_{N}_{dt_tau}_{Hamrep}.pdf"
    )
    plt.show()

    # --- adiabatic evolution for multiple tmax ---
    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
        model.tmax = tmax_

        print(f"Running Adiabatic_precomputed (trunc={trunc}) for geometry={geometry}, protocol={protocol}, tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        T_run, a_t_reduced, b_t, P0_orig_phys = model.run(T=None)
        t_global_end = time.perf_counter()
        print(f"  Time: {t_global_end - t_global_start:.3f} s\n")

        tau = T_run / tmax_

        np.save(f"{geometry}_{protocol}_T_run_trunc={trunc}_{tmax_}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", T_run)
        np.save(f"{geometry}_{protocol}_a_t_reduced_trunc={trunc}_{tmax_}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", a_t_reduced)
        np.save(f"{geometry}_{protocol}_psi_t_trunc={trunc}_{tmax_}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", b_t)
        np.save(f"{geometry}_{protocol}_c_t_orig_trunc={trunc}_{tmax_}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamiltonian_type}.npy", P0_orig_phys)

        P0_adiab = np.abs(a_t_reduced[:,0])**2
        P0_orig = P0_orig_phys

        # plt.subplot(1,2,2)
        plt.plot(tau, P0_adiab, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |a_0|^2$")
        plt.title(f"Adiabatic basis (truncated to {trunc} lowest energy levels)")
        plt.legend()

        # plt.subplot(1,2,1)
        # plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        # plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        # plt.ylabel(r"$P = |n_0|^2$")
        # plt.title("Original basis (projected onto truncated space)")
        # plt.legend()

    plt.tight_layout()
    plt.savefig(f"{geometry}_{protocol}_trunc={trunc}_{len(tmax_list)}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_N={N}_dt={dt_tau}_{Hamrep}.pdf")
    plt.show()


if __name__ == "__main__":
    # Examples:
    # T-junction full circuit (old behavior)
    # adiabatic_approx(geometry="T", protocol="full_circuit")

    # 1D chain cutting protocol
    # adiabatic_approx(geometry="chain", protocol="cut_chain")

    # T-chain left/right switch protocol
    adiabatic_approx(geometry="T", protocol="T_chain_switch")


    