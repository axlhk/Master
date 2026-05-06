import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh, ArpackNoConvergence
from numba import njit
import time 


class Kitaev:
    "dt     :       timestep (i fysisk tid eller i tau, avhengig av bruk)"
    "tmax   :       Total time T"
    "N      :       number of sites"
    "Stride :       How often the loops print progress"
    "Hamiltonian:   Kitaev or BdG"
    "Integration:   (Euler, Euler_Cromer) or RK4"
    "Evolution:     Adiabatic, Adiabatic_precomputed eller Schrodinger, Schrodinger_precomputed"
    "omega, mu, delta: functions(j, N, t, tmax)" 
    "return_full:   True or False, to return all a_t's, or just the gs."
    "trunc_dim:     Dimensjon på truncert lavenergirom"
    "warm_start:    Bruk forrige gs som startvektor i eigsh under precompute"
    "HamRepresentation = dense or sparse"

    def __init__(self, dt, N, stride, tmax,
                 Hamiltonian, Integration, Evolution,
                 omega_fun, mu_fun, delta_fun,
                 exp_degen, return_full,
                 trunc_dim=None,
                 warm_start=True,
                 HamRepresentation="dense"):

        self.dt   = dt
        self.N    = N
        self.stride = stride
        self.tmax = tmax

        self.exp_degen = exp_degen
        self.return_full = return_full
        self.warm_start = warm_start

        self.omega_fun = omega_fun
        self.mu_fun    = mu_fun
        self.delta_fun = delta_fun

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
            self.dim = 2**N
            self.ntot = precompute_ntot(N)
            self.ham_is_sparse = (HamRepresentation == "sparse")
            
            if self.ham_rep == "dense":
                self.ham_builder = H_Kitaev_Chain
                self.ham_is_sparse = False
            elif self.ham_rep == "sparse":
                self.ham_builder = H_Kitaev_Chain_sparse_cached
                self.ham_is_sparse = True
            else:
                raise ValueError(
                    f"Unknown HamRepresentation: {self.ham_rep}. "
                    "Use 'dense' or 'sparse'."
                )
                
        elif Hamiltonian == "BdG":
            self.dim = 2*N
            self.ham_is_sparse = False
            if self.ham_rep == "dense":
                self.ham_builder = H_BdG
            elif self.ham_rep == "sparse":
                # Only dense BdG implemented so far
                raise ValueError("Sparse BdG not implemented. Use HamRepresentation='dense'.")
            else:
                raise ValueError(
                    f"Unknown HamRepresentation: {self.ham_rep}. "
                    "Use 'dense' or 'sparse'."
                )
        else:
            raise ValueError(f"Unknown Hamiltonian type: {Hamiltonian}. Use either Kitaev or BdG")

        # trunceringsdimensjon i adiabatiske ligninger
        if trunc_dim is None:
            self.trunc_dim = self.dim
        else:
            self.trunc_dim = min(trunc_dim, self.dim)

        # Choose integrator for the evolution type
        if Integration == "Euler":
            self.stepper = evolution_step_euler          # used by run_adiabatic
            self.sch_stepper = schrodinger_step_euler    # used by _run_schrodinger
        elif Integration == "Euler_Cromer":
            self.stepper = evolution_step_euler_cromer
            self.sch_stepper = None  # or raise later if used
        elif Integration == "RK4":
            self.stepper = evolution_step_RK4
            self.sch_stepper = schrodinger_step_RK4
        else:
            raise ValueError(f"Unknown integrator: {Integration}. Use either Euler or RK4. Euler_Cromer not yet implemented.")

        # Evolusjonstype
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
        
    def run(self, T):    # For easy calling
        return self.run_method(T)
    
    def Diag_H(self, H):
        """
        Diagonalise Hamiltonian H (dense or sparse).
        Always converts to dense before calling eigh.
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
            # Always use dense H_Kitaev_Chain here
            H = H_Kitaev_Chain(self.N, t, self.tmax,
                            self.omega_fun, self.mu_fun, self.delta_fun,
                            self.ntot)
        elif self.ham_type == "BdG":
            H = H_BdG(self.N, t, self.tmax,
                    self.omega_fun, self.mu_fun, self.delta_fun)
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
        Kjør precompute én gang og lagre på denne instansen.
        Må kalles før Evolution='Adiabatic_precomputed' brukes.
        Bruker sparse H + eigsh med mulighet for warm start.
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
        """
        Precompute E(τ) og V(τ) for τ ∈ [0, 1], på et Nt_tau-gitter.
        Trunkerer til self.trunc_dim laveste energitilstander ved hver τ.

        Bruker sparse H og eigsh for å hente kun trunc_dim laveste
        eigenpar. Warm start: bruker forrige gs som startvektor v0 ved
        neste τ hvis self.warm_start = True.
        """
        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        dim_trunc = self.trunc_dim
        dim_full = self.dim

        E_store = np.zeros((Nt_tau, dim_trunc))
        V_store = np.zeros((Nt_tau, dim_full, dim_trunc), dtype=complex)

        #Precompute couplings for all tau
        omega_tab, delta_tab = precompute_couplings_Kitaev(self.N, tau_grid, 1.0)

        t0_all = time.perf_counter()

        v0 = None   # startvektor for eigsh
        for idx, tau in enumerate(tau_grid):
            t_dummy = tau

            # Bygg sparse H direkte
            if self.ham_type == "Kitaev":
                mu = self.mu_fun(t_dummy, 1.0)
                H_sparse = H_Kitaev_Chain_sparse_cached(
                    self.N,
                    mu,
                    omega_tab[idx],   # 1D array length N-1
                    delta_tab[idx],   # 1D array length N-1
                    self.ntot
                )
            else:
                raise ValueError("precompute_eigensystem_tau only implemented for Kitaev here")
            # else:
            #     # (BdG-varianten kan også gjøres sparse på tilsvarende måte om ønskelig)
            #     H_dense = H_BdG(self.N, t_dummy, 1.0,
            #                     self.omega_fun, self.mu_fun, self.delta_fun)
            #     H_sparse = csr_matrix(H_dense)

            # Diagonaliser laveste trunc_dim egenverdier
            # (eigsh gir dem ikke nødvendigvis sortert)
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
                    # enkel fallback: fyll opp med NaN / 0 – eller evt. ta en fallback-eigh
                    dim_here = E_part.shape[0]
                    E_tmp = np.full(dim_trunc, np.nan, dtype=float)
                    V_tmp = np.zeros((dim_full, dim_trunc), dtype=complex)
                    E_tmp[:dim_here] = E_part
                    V_tmp[:, :dim_here] = V_part
                    E_part, V_part = E_tmp, V_tmp

            # sorter etter energi (stigende)
            idx_sorted = np.argsort(E_part)
            E_sorted = E_part[idx_sorted]
            V_sorted = V_part[:, idx_sorted]

            E_store[idx] = E_sorted
            V_store[idx] = V_sorted

            # Warm start: bruk gs som startvektor neste gang
            if self.warm_start:
                v0 = V_sorted[:, 0]

            if 10 and (idx % 10 == 0):
                print(f"[Precompute] Step {idx} / {Nt_tau-1}. tau = {tau:.3f}")

        t1_all = time.perf_counter()
        self.time_precompute += (t1_all - t0_all)

        return tau_grid, E_store, V_store

    def run_adiabatic_precomputed(self, T, E_store, V_store):
        """
        Adiabatic evolusjon der eigenpar (E,V) er precomputet på samme
        tidsgitter (tau-grid) som T er basert på.

        Forutsetter:
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

        #Finn fysisk grunntilstand ved t=0 i full basis
        H0_full = self.build_H(0.0)             # bruker tett H_Kitaev_Chain
        E0_full, V0_full = self.Diag_H(H0_full)
        psi0 = V0_full[:, 0]                    # fysisk gs ved t=0
        # preallocer P0_orig_phys
        P0_orig_phys = np.zeros(Nt, dtype=float)
        P0_orig_phys[0] = 1.0                   # <psi0|psi0>^2

        # initiale eigenpar (truncerte)
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

        # Viktig: phase_int må være dim_trunc × dim_trunc
        phase_int = np.zeros((dim_trunc, dim_trunc), dtype=complex)

        A = np.empty((dim_trunc, dim_trunc), dtype=complex)
        lambda_vec = np.empty(dim_trunc, dtype=complex)
        B_mat = np.empty((dim_trunc, dim_trunc), dtype=complex)  # for overlaps if you want

        # n = 0: forward difference
        A0 = psi_dotpsi_forward(V0, V1, dt)
        lambda0 = lambda_from_E_and_A(E0, A0)
        a_t[1], phase_int = self.stepper(a_t[0], phase_int, A0, lambda0, dt)

        #beregn fysisk psi(t1) og projiser på psi0
        I_k_1 = -phase_int[0, :]
        phase_factors_1 = np.exp(-1j * I_k_1)
        a_phys_1 = phase_factors_1 * a_t[1]     # fys. koeff i instantan basis ved t1
        psi_trunc_1 = V1 @ a_phys_1             # |psi(t1)> i full basis (truncert)
        P0_orig_phys[1] = np.abs(np.vdot(psi0, psi_trunc_1))**2

        V_prev, V_curr = V0, V1
        E_curr = E1

        t_loop_start = time.perf_counter()

        # tidsutvikling (ingen Diag_H/build_H her!)
        for n in range(1, Nt - 1):
            # 1) Get raw (unsorted) eigenpairs at next step
            E_next_raw = E_store[n+1]
            V_next_raw = V_store[n+1]

            # 2) Match and fix phases relative to current basis
            V_next_matched, perm = match_eigenvectors(V_curr, V_next_raw)
            E_next = E_next_raw[perm]
            V_next = fix_phases(V_curr, V_next_matched)

            # 3) Central Berry connection and lambda in-place
            central_A_inplace(V_prev, V_curr, V_next, dt, A)
            lambda_from_E_and_A_inplace(E_curr, A, lambda_vec)

            # 4) Adiabatic step in truncated basis
            a_t[n+1], phase_int = self.stepper(a_t[n], phase_int, A, lambda_vec, dt)

            # 5) Reconstruct physical amplitudes in original basis
            I_k = -phase_int[0, :]
            phase_factors = np.exp(-1j * I_k)
            a_phys = phase_factors * a_t[n+1]

            B_mat[:] = V0.conj().T @ V_curr
            b_t[n+1] = B_mat @ a_phys

            # 6) Physical projection onto psi0
            psi_trunc = V_curr @ a_phys
            P0_orig_phys[n+1] = np.abs(np.vdot(psi0, psi_trunc))**2

            # 7) Rotate for next iteration
            V_prev, V_curr = V_curr, V_next
            E_curr = E_next

            E_all[n+1] = E_next

            if self.stride and (n % self.stride == 0):
                print(f"[Adiabatic-precomputed] Step {n} / {Nt-1}. T = {self.tmax}")


        t_loop_end = time.perf_counter()
        self.time_evolution_loop += (t_loop_end - t_loop_start)

        # reduksjon til exp_degen laveste innenfor trunc_dim
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
        Kjøres når Evolution='Adiabatic_precomputed'.
        Ignorerer T_dummy. Bruker precomputed tau_grid/E_store/V_store.
        """
        if self.tau_grid is None or self.E_store is None or self.V_store is None:
            raise RuntimeError(
                "Precomputed eigensystem mangler. "
                "Kall prepare_precomputed(Nt_tau) før du bruker Adiabatic_precomputed."
            )

        # fysisk tidsgitter for denne tmax
        T = self.tau_grid * self.tmax
        dt_phys = T[1] - T[0]

        # midlertidig sett fysisk dt inn i self.dt (brukes av stepper)
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
        Dense precompute of E(τ), V(τ) for τ in [0,1].
        This *always* uses the chosen Hamiltonian type (Kitaev or BdG),
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
                H = H_Kitaev_Chain(self.N, tau, 1.0,
                                self.omega_fun, self.mu_fun, self.delta_fun,
                                self.ntot)
            elif self.ham_type == "BdG":
                H = H_BdG(self.N, tau, 1.0,
                        self.omega_fun, self.mu_fun, self.delta_fun)
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

    def run_schrodinger_with_precomputed(self, _=None):
        """
        Schrödinger evolution using precomputed dense eigenbases on the same tau-grid.

        Requires:
        - self.tau_grid, self.E_store_dense, self.V_store_dense from prepare_precomputed_dense().
        """
        if not hasattr(self, "tau_grid") or not hasattr(self, "V_store_dense"):
            raise RuntimeError("Call prepare_precomputed_dense(Nt_tau) first.")

        tau_grid = self.tau_grid
        V_store = self.V_store_dense
        E_store = self.E_store_dense

        Nt = len(tau_grid)
        dim = self.dim

        # physical time grid consistent with tau_grid and current tmax
        T = tau_grid * self.tmax
        dt = T[1] - T[0]

        psi_t = np.zeros((Nt, dim), dtype=complex)
        a_t_inst = np.zeros((Nt, dim), dtype=complex)  # instantaneous coeffs

        # initial state: ground state at t=0 from H(0)
        H0 = self.build_H(0.0)
        E0, V0 = self.Diag_H(H0)
        psi0 = V0[:, 0]
        psi_t[0] = psi0

        # consistency: at tau=0, V_store_dense[0] should be close to V0
        # but we won't rely on ordering matching; just use V_store as is.

        # instantaneous coeffs at t=0
        a_t_inst[0] = V_store[0].conj().T @ psi_t[0]

        for n in range(Nt - 1):
            t_n = T[n]

            if self.sch_stepper is None:
                raise NotImplementedError("Schrodinger evolution not implemented for this integrator.")

            psi_t[n+1] = self.sch_stepper(psi_t[n], t_n, dt, self.build_H)

            # optional renormalisation to suppress norm drift
            norm = np.linalg.norm(psi_t[n+1])
            psi_t[n+1] /= norm

            # project onto precomputed instantaneous eigenbasis at step n+1
            V_inst = V_store[n+1]
            a_t_inst[n+1] = V_inst.conj().T @ psi_t[n+1]

            if self.stride and (n % self.stride == 0):
                print(f"[Schrodinger-precomputed] Step {n} / {Nt-1}. T = {self.tmax}, norm={norm}")
        
        c_t_orig = psi_t @ V0.conj()

        return T, a_t_inst, psi_t, c_t_orig

    def _run_schrodinger(self, T):
        """
        Schrödinger evolution in the original basis.
        """
        dt  = self.dt
        dim = self.dim
        Nt  = len(T)
        E_all = np.zeros((Nt, dim))

        psi_t = np.zeros((Nt, dim), dtype=complex)
        a_t = np.zeros((Nt, dim), dtype=complex)

        H0 = self.build_H(T[0])
        E0, V0 = self.Diag_H(H0)
        E_all[0] = E0

        psi0 = V0[:, 0]
        psi_t[0] = psi0
        a_t[0] = V0.conj().T @ psi0

        for n in range(Nt - 1):
            t_n = T[n]

            if self.sch_stepper is None:
                raise NotImplementedError(
                    "Schrödinger evolution not implemented for this integrator."
                )

            psi_t[n+1] = self.sch_stepper(psi_t[n], t_n, dt, self.build_H)

            t_next = T[n+1]
            H_next = self.build_H(t_next)
            E_next, V_next = self.Diag_H(H_next)

            E_all[n+1] = E_next
            a_t[n+1] = V_next.conj().T @ psi_t[n+1]

            if self.stride and (n % self.stride == 0):
                print(f"[Schrödinger] Step {n} / {Nt-1}. T = {self.tmax}")

        c_t = psi_t @ V0.conj()

        k = self.exp_degen
        a_t_reduced = np.zeros((Nt, k), dtype=complex)

        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]
            idx_sorted = np.argsort(E)
            idx_gs = idx_sorted[:k]
            a_t_reduced[n, :] = d[idx_gs]

        return T, a_t_reduced, c_t


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
def H_Kitaev_Chain(N, t, tmax, omega_fun, mu_fun, delta_fun, ntot_arr):
    dim = 2**N
    mu = mu_fun(t, tmax)

    H = np.zeros((dim, dim), dtype=np.complex128)

    for n in range(dim):
        ntot = ntot_arr[n]
        H[n, n] += - mu * (ntot - N/2)

        for j in range(N-1):
            c1, m1 = a_on_basis_state(n, j + 1)
            if m1 != -1:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    H[m2, n] += -omega_fun(j, N, t, tmax) * c1 * c2

            c3, m3 = a_on_basis_state(n, j)
            if m3 != -1:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 != -1:
                    H[m4, n] += -omega_fun(j, N, t, tmax) * c3 * c4

            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 != -1:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    H[m6, n] += delta_fun(j, N, t, tmax) * c5 * c6

            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 != -1:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 != -1:
                    H[m8, n] += delta_fun(j, N, t, tmax).conjugate() * c7 * c8

    return H

@njit
def precompute_couplings_Kitaev(N, tau_grid, tmax):
    """
    Precompute omega_j(t) and delta_j(t) on the tau grid, for a Kitaev chain.
    Assumes omega_Eivind_4_2 and delta_Eivind_4_2 are @njit.
    """
    Nt = len(tau_grid)
    omega_tab = np.empty((Nt, N-1), dtype=np.float64)
    delta_tab = np.empty((Nt, N-1), dtype=np.complex128)

    for it in range(Nt):
        tau = tau_grid[it]
        t = tau * tmax   # use your chosen mapping here
        for j in range(N-1):
            omega_tab[it, j] = omega_Eivind_4_2(j, N, t, tmax)
            delta_tab[it, j] = delta_Eivind_4_2(j, N, t, tmax)
    return omega_tab, delta_tab

def H_Kitaev_Chain_sparse_cached(N, mu, omega_j, delta_j, ntot_arr):
    dim = 2**N

    rows = []
    cols = []
    data = []

    for n in range(dim):
        ntot = int(ntot_arr[n])
        rows.append(n)
        cols.append(n)
        data.append(- mu * (ntot - N/2))

        for j in range(N-1):
            w = omega_j[j]
            d = delta_j[j]

            c1, m1 = a_on_basis_state(n, j + 1)
            if m1 != -1:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    val = -w * c1 * c2
                    if val != 0.0:
                        rows.append(m2)
                        cols.append(n)
                        data.append(val)

            c3, m3 = a_on_basis_state(n, j)
            if m3 != -1:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 != -1:
                    val = -w * c3 * c4
                    if val != 0.0:
                        rows.append(m4)
                        cols.append(n)
                        data.append(val)

            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 != -1:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    val = d * c5 * c6
                    if val != 0.0:
                        rows.append(m6)
                        cols.append(n)
                        data.append(val)

            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 != -1:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 != -1:
                    val = d.conjugate() * c7 * c8
                    if val != 0.0:
                        rows.append(m8)
                        cols.append(n)
                        data.append(val)

    H_sparse = csr_matrix((np.array(data, dtype=np.complex128),
                           (np.array(rows, dtype=np.int32),
                            np.array(cols, dtype=np.int32))),
                          shape=(dim, dim))
    return H_sparse


def H_BdG(N, t, tmax, omega_fun, mu_fun, delta_fun):
    mu = mu_fun(t, tmax)

    h = np.zeros((N, N), dtype=complex)
    for j in range(N):
        h[j, j] = -mu
        if j < N - 1:
            h[j, j+1] = -omega_fun(j, N, t, tmax)
            h[j+1, j] = -omega_fun(j, N, t, tmax)

    Delta = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        Delta[j, j+1] = delta_fun(j, N, t, tmax)
        Delta[j+1, j] = -delta_fun(j, N, t, tmax)

    upper = np.hstack((h, Delta))
    lower = np.hstack((Delta.conj().T, -h.T))
    H_bdg = np.vstack((upper, lower))

    return H_bdg


"Schrodinger integrators and helper functions"
def rhs_schrodinger(c, H):
    return -1j * (H @ c)

def schrodinger_step_euler(c_n, t_n, dt, build_H):
    H_n = build_H(t_n)
    dc_dt = rhs_schrodinger(c_n, H_n)
    c_next = c_n + dt * dc_dt
    return c_next

def schrodinger_step_RK4(c_n, t_n, dt, build_H):
    H1 = build_H(t_n)
    k1 = rhs_schrodinger(c_n, H1)

    t2 = t_n + 0.5*dt
    H2 = build_H(t2)
    c2 = c_n + 0.5*dt*k1
    k2 = rhs_schrodinger(c2, H2)

    c3 = c_n + 0.5*dt*k2
    k3 = rhs_schrodinger(c3, H2)

    t4 = t_n + dt
    H4 = build_H(t4)
    c4 = c_n + dt*k3
    k4 = rhs_schrodinger(c4, H4)

    c_next = c_n + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return c_next


"Adiabatic integrators and helper functions"
def evolution_step_euler(a_n, phase_int, A, lambd, dt):
    phase_matrix = np.exp(-1j * phase_int)
    M = phase_matrix * A
    np.fill_diagonal(M, 0.0)

    rhs = M @ a_n
    a_next = a_n + dt * rhs

    delta_lambda = lambd[:, None] - lambd[None, :]
    phase_int_next = phase_int + delta_lambda * dt

    return a_next, phase_int_next

def evolution_step_euler_cromer():
    a_next = 1
    phase_int_next = 1
    return a_next, phase_int_next

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
@njit
def omega_Eivind_4_2(j, N, t, tmax):
    j_c = N//2 - 1  #-1 to cut the central bond
    if j == j_c:
        return 2 * omega0 * np.abs(t/tmax - 0.5)
    else:
        return omega0

@njit
def mu_Eivind_4_2(t, tmax):
    return mu0  

@njit
def delta_Eivind_4_2(j, N, t, tmax):
    j_c = N//2 - 1 #-1 to cut the central bond
    if j == j_c:
        return 2 * delta0 * np.abs(t/tmax - 0.5)
    else:
        return delta0


"Eksempel-funksjoner (truncation, adiabatic_approx) - oppdatert til ny __init__"
def truncation():   #Used to find appropriate truncation 
    dt_tau = 1e-2
    Nt_tau = int(1.0/dt_tau) + 1

    eps = 1e-5 #Want to keep (1-eps) of the total probability
    k_needed_all = []

    tmax_list = [0.01, 0.1, 1]  #Can ignore higher tmax; the slower the change, the more adiabatic

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic_precomputed"
    exp_degen = 4
    return_full = True
    stride = 10
    trunc_dim = 2**N - 2  # her vil vi se hele halen; du kan sette til f.eks. 200 også

    model = Kitaev(dt=dt_tau,
                   N=N,
                   stride=stride,
                   tmax=1.0,
                   Hamiltonian=Hamiltonian_type,
                   Integration=Integrator,
                   Evolution=Evolution_type,
                   omega_fun=omega_Eivind_4_2,
                   mu_fun=mu_Eivind_4_2,
                   delta_fun=delta_Eivind_4_2,
                   exp_degen=exp_degen,
                   return_full=return_full,
                   trunc_dim=trunc_dim)

    _ = model.build_H(0.0)

    model.prepare_precomputed(Nt_tau)

    plt.figure(figsize=(6,4))

    for tmax_ in tmax_list:
        model.tmax = tmax_

        print(f"Running adiabatic (precomputed) for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        T_run, a_t_reduced, b_t, a_t_full, E_all = model.run(T=None)
        t_global_end = time.perf_counter()
        print(f"  Tid: {t_global_end - t_global_start:.3f} s\n")

        a_final = a_t_full[-1]
        E_final = E_all[-1]

        idx_sorted = np.argsort(E_final)
        probs_sorted = np.abs(a_final[idx_sorted])**2

        cum = np.cumsum(probs_sorted)
        tail = 1.0 - cum
        k_needed = np.argmax(tail < eps)
        k_needed_all.append(k_needed + 1)

        n_idx = np.arange(len(probs_sorted))
        plt.scatter(n_idx, probs_sorted,
                    alpha=0.7,
                    s=15,
                    label=f"T = {tmax_}")

    k_safe = max(k_needed_all)
    print("Safe truncation dimension for eps =", eps, "is k_safe =", k_safe)
    
    plt.yscale("log")
    plt.xlabel("tilstandsindeks n (sortert etter energi)")
    plt.ylabel(r"$|a_n(T_\mathrm{final})|^2$")
    plt.title("Fordeling av amplituder i instantan egenbasis ved sluttid")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"Truncation_distribution_{Hamiltonian_type}_{N}.pdf")
    plt.show()

def adiabatic_approx():
    dt_tau = 1 / 50
    Nt_tau = int(1.0/dt_tau) + 1
    stride = 10

    # tmax_list = [0.01, 0.1, 0.5, 1, 5, 10, 20]
    tmax_list = [0.01]
    trunc = 150 #136 for eps = 1e-4

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic_precomputed"
    exp_degen = 1
    return_full = False
    Hamrep = "sparse"   #sparse, dense

    model = Kitaev(dt=dt_tau,
                   N=N,
                   stride=stride,
                   tmax=1.0,
                   Hamiltonian=Hamiltonian_type,
                   Integration=Integrator,
                   Evolution=Evolution_type,
                   omega_fun=omega_Eivind_4_2,
                   mu_fun=mu_Eivind_4_2,
                   delta_fun=delta_Eivind_4_2,
                   exp_degen=exp_degen,
                   return_full=return_full,
                   trunc_dim=trunc,
                   HamRepresentation=Hamrep)

    _ = model.build_H(0.0)

    model.prepare_precomputed(Nt_tau)

    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
        model.tmax = tmax_

        print(f"Running Adiabatic_precomputed (trunc={trunc}) for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        T_run, a_t_reduced, b_t, P0_orig_phys = model.run(T=None)
        t_global_end = time.perf_counter()
        print(f"  Tid: {t_global_end - t_global_start:.3f} s\n")

        tau = T_run / tmax_

        P0_adiab = np.sum(np.abs(a_t_reduced)**2, axis=1)
        P0_orig = P0_orig_phys
        # P0_orig = np.sum(np.abs(b_t[:, :exp_degen])**2, axis=1)

        plt.subplot(1,2,2)
        plt.plot(tau, P0_adiab, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = \sum_{k<\mathrm{exp\_degen}} |a_k|^2$")
        plt.title("Adiabatisk basis (truncert)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |n_0|^2$")
        plt.title("Original basis (projisert på truncert rom)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"__eivind_{len(tmax_list)}_{omega0}_{delta0}_{mu0}_{Evolution_type}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{Hamrep}.pdf")
    plt.show()

def schrodinger():
    dt_tau = 1e-2   #dt = 1e-2 N = 10 takes about 330s per tmax
    Nt_tau = int(1.0/dt_tau) + 1
    tau_grid = np.linspace(0.0, 1.0, Nt_tau)

    stride = 10

    tmax_list = [0.01, 0.5]   #Inaccurate for tmax 1 and up, using dt = 1e-2

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Schrodinger"
    exp_degen = 4
    return_full = False

    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
        # Physical time grid corresponding to same tau-grid
        T = tau_grid * tmax_
        dt_phys = T[1] - T[0]

        model = Kitaev(dt=dt_phys,
                       N=N,
                       stride=stride,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=Evolution_type,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full,
                       trunc_dim=None)   # full Hilbert space

        print(f"Running Schrodinger for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        T_run, a_t_reduced, c_t = model.run(T)
        t_global_end = time.perf_counter()
        print(f"  Tid: {t_global_end - t_global_start:.3f} s\n")

        # tau is exactly tau_grid (up to float roundoff)
        tau = T_run / tmax_

        # Instantaneous gs-subspace probability (like your P0_adiab)
        P0_inst = np.sum(np.abs(a_t_reduced)**2, axis=1)

        # Original basis: exact probability to remain in ground state at t=0
        # c_t = psi_t @ V0.conj(), so c_t[:,0] is overlap with psi0
        P0_orig = np.abs(c_t[:, 0])**2

        plt.subplot(1,2,2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = \sum_{k<\mathrm{exp\_degen}} |a_k|^2$")
        plt.title("Instantan basis (Schrödinger)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |\langle \psi_0 | \psi(t)\rangle|^2$")
        plt.title("Original basis (Schrödinger)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{len(tmax_list)}_Schrodinger_{Hamiltonian_type}_{Integrator}_{N}.pdf")
    plt.show()

def schrodinger_approx_with_instantaneous():
    "Approximate as we interpolate for V(tau), but this should hold as H(t)=H(tau)"
    dt_tau = 1e-3                   #For dt = 1e-2 N = 10 the precompute is about 330s. Each tmax about 10. 
    Nt_tau = int(1.0/dt_tau)        #dt = 1e-2 accurate norm up until tmax = 5, norm of tmax = 10 is about 0.96
                                    #dt = 5e-3 is stable for tmax = 10, but norm is 0.96-0.99 for tmax = 20
                                    #dt = 1e-3 is stable enough for tmax = 20
                                    #dt = 1e-4 is unable to run; requires 156. GiB??!!
    stride = 10
    tmax_list = [0.01, 0.1, 0.5, 1, 5, 10, 20]

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    exp_degen = 1
    return_full = False
    evolution = "Schrodinger_precomputed"
    HamRep = "dense"   #dense, sparse

    # 1) One model used only to precompute instantaneous eigenpairs in τ
    pre_model = Kitaev(dt=dt_tau,
                       N=N,
                       stride=stride,
                       tmax=1.0,  # irrelevant in precompute_dense; we pass tmax=1.0 explicitly there
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=evolution,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full,
                       trunc_dim=None,
                       HamRepresentation=HamRep)

    print("Precomputing dense eigenpairs on tau-grid (shared for all tmax)...")
    t0 = time.perf_counter()
    pre_model.prepare_precomputed_dense(Nt_tau)
    t1 = time.perf_counter()
    print(f"  Time to precompute: {t1 - t0:.3f} s\n")

    plt.figure(figsize=(8,4))

    # 2) For each tmax, make a model that REUSES the precomputed τ-eigenpairs
    for tmax_ in tmax_list:
        model = Kitaev(dt=dt_tau,          # dt in τ; run_schrodinger_with_precomputed will convert
                       N=N,
                       stride=stride,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution="Schrodinger_precomputed",
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full,
                       trunc_dim=None,
                       HamRepresentation=HamRep)

        # Share precomputed τ-grid and eigenpairs
        model.tau_grid = pre_model.tau_grid
        model.E_store_dense = pre_model.E_store_dense
        model.V_store_dense = pre_model.V_store_dense

        print(f"Running Schrödinger (with precomputed basis) for tmax = {tmax_}...")
        t0_ = time.perf_counter()
        # Note: run_schrodinger_with_precomputed ignores its argument
        T_run, a_t_inst, psi_t, c_t_orig = model.run(T=None)
        t1_ = time.perf_counter()
        print(f"  Time to run: {t1_ - t0_:.3f} s\n")

        tau = T_run / tmax_

        # instantaneous gs-subspace probability
        P0_inst = np.sum(np.abs(a_t_inst[:, :exp_degen])**2, axis=1)

        # original-basis probability in psi0 (ground state at t=0)
        P0_orig = np.zeros_like(tau)        
        psi0 = psi_t[0]
        for n in range(len(tau)):
            P0_orig[n] = np.abs(np.vdot(psi0, psi_t[n]))**2     #OOPS: psi_t is actually in Fock basis. Use c_t_orig. (They coincide for the ground state so it's fine)

        np.save(f"_T_run_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy", T_run)
        np.save(f"_a_t_inst_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy", a_t_inst)
        np.save(f"_psi_t_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy", psi_t)

        plt.subplot(1,2,2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = \sum_{k<\mathrm{exp\_degen}} |a_k|^2$")
        plt.title("Instantan basis (Schrödinger)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |\langle \psi_0 | \psi(t)\rangle|^2$")
        plt.title("Original basis (Schrödinger)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"_eivind_{len(tmax_list)}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.pdf")
    plt.show()

def plot_delete():
    dt_tau = 5e-3                   #For dt = 1e-2 N = 10 the precompute is about 330s. Each tmax about 10. 
    Nt_tau = int(1.0/dt_tau) + 1    #dt = 1e-2 accurate norm up until tmax = 5, norm of tmax = 10 is about 0.96
                                    #dt = 5e-3 is stable for tmax = 10, but norm is 0.96-0.99 for tmax = 20
                                    #dt = 1e-4 is unable to run; requires 156. GiB??!!
    tmax_list = [0.01, 0.1, 0.5, 1, 5, 10]

    for tmax_ in tmax_list:
        T_run = np.load(f"T_run_{tmax_}.npy")
        a_t_inst = np.load(f"a_t_inst_{tmax_}.npy")
        psi_t = np.load(f"psi_t_{tmax_}.npy")

        tau = T_run / tmax_


        P0_inst = np.sum(np.abs(a_t_inst[:, :1])**2, axis=1)

        # original-basis probability in psi0 (ground state at t=0)
        P0_orig = np.zeros_like(tau)
        psi0 = psi_t[0]
        for n in range(len(tau)):
            P0_orig[n] = np.abs(np.vdot(psi0, psi_t[n]))**2

        P0_inst = np.sqrt(P0_inst)
        P0_orig = np.sqrt(P0_orig)

        plt.subplot(1,2,2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = \sum_{k<\mathrm{exp\_degen}} |a_k|^2$")
        plt.title("Instantan basis (Schrödinger)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |\langle \psi_0 | \psi(t)\rangle|^2$")
        plt.title("Original basis (Schrödinger)")
        plt.legend()

    plt.show()

def plot_delete_():
    """
    Plot probabilities of original and instantaneous ground states for all tmax
    using the saved data from schrodinger_approx_with_instantaneous():

        T_run_{tmax}.npy   : time grid (length Nt)
        a_t_inst_{tmax}.npy: instantaneous coefficients in eigenbasis of H(t)
        psi_t_{tmax}.npy   : wavefunction in original Fock basis

    For each tmax we compute:
      P_orig_gs(t)  = |<E0(0) | psi(t)>|^2
      P_inst_gs(t)  = |a_0^{(inst)}(t)|^2
    and plot all tmax curves on the same two subplots.
    """

    # Same N and Hamiltonian as in schrodinger_approx_with_instantaneous()
    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    stride = 0
    exp_degen = 1
    return_full = False

    # tmax values for which you have saved data
    tmax_list = [0.01, 0.1, 0.5, 1, 5, 10]

    # Prepare figure with two panels: original basis / instantaneous basis
    plt.figure(figsize=(8, 4))

    ax_orig = plt.subplot(1, 2, 1)
    ax_inst = plt.subplot(1, 2, 2)

    for tmax_ in tmax_list:
        # Load saved evolution data
        T_run    = np.load(f"T_run_{tmax_}.npy")       # (Nt,)
        a_t_inst = np.load(f"a_t_inst_{tmax_}.npy")    # (Nt, dim)
        psi_t    = np.load(f"psi_t_{tmax_}.npy")       # (Nt, dim)

        Nt  = len(T_run)
        dim = a_t_inst.shape[1]
        tau = T_run / tmax_

        # Build a helper model just to get H(0) and its ground state
        dt_phys = T_run[1] - T_run[0]
        model = Kitaev(dt=dt_phys,
                       N=N,
                       stride=stride,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution="Schrodinger",
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full,
                       trunc_dim=None)

        # Original ground state |E0(0)>
        H0 = model.build_H(T_run[0])
        E0, V0 = model.Diag_H(H0)
        psi0 = V0[:, 0]

        # P_orig_gs(t) = |<psi0 | psi(t)>|^2
        P_orig_gs = np.zeros(Nt)
        for n in range(Nt):
            P_orig_gs[n] = np.abs(np.vdot(psi0, psi_t[n]))**2

        # P_inst_gs(t) = |a_0^{(inst)}(t)|^2  (index 0 = instantaneous ground state)
        P_inst_gs = np.abs(a_t_inst[:, 0])**2

        # Plot on common axes for all tmax
        ax_orig.plot(tau, P_orig_gs, label=f"T = {tmax_}")
        ax_inst.plot(tau, P_inst_gs, label=f"T = {tmax_}")

    # Label and formatting
    ax_orig.set_xlabel(r"$\tau = t / T$")
    ax_orig.set_ylabel(r"$P_{\text{orig,gs}}$")
    ax_orig.set_title("Original ground state probability")
    ax_orig.legend()

    ax_inst.set_xlabel(r"$\tau = t / T$")
    ax_inst.set_ylabel(r"$P_{\text{inst,gs}}$")
    ax_inst.set_title("Instantaneous ground state probability")
    ax_inst.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Kjør f.eks.:
    # truncation()  #NOT OPTIMIZED FOR THIS CODE, SEE Truncation_Optimized.py
    # adiabatic_approx()
    # schrodinger()
    schrodinger_approx_with_instantaneous()
    # plot_delete()
    # plot_delete_()



"Python time analysis / profiling"
"Få kode av Eivind"
"- Match resultater"
"Bygg om kode til å lagre ALLE filer"
"Les opp på T junctions"