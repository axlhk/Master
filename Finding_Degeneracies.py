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
    "Evolution:     Adiabatic, Adiabatic_precomputed eller Schrodinger"
    "omega, mu, delta: functions(j, N, t, tmax)" 
    "return_full:   True or False, to return all a_t's/E_all, or just P0-inst etc."
    "trunc_dim:     Dimensjon på truncert lavenergirom"
    "warm_start:    Bruk forrige gs som startvektor i eigsh under precompute"
    "eps_degen:     Energitorskel for å definere gs-degenerasjon"

    def __init__(self, dt, N, stride, tmax,
                 Hamiltonian, Integration, Evolution,
                 omega_fun, mu_fun, delta_fun,
                 return_full,
                 trunc_dim=None,
                 warm_start=True,
                 eps_degen=1e-7):

        self.dt   = dt
        self.N    = N
        self.stride = stride
        self.tmax = tmax

        self.return_full = return_full
        self.warm_start = warm_start
        self.eps_degen = eps_degen

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

        # Precomputed-data (for Schrodinger_precomputed)
        self.E_store_dense = None
        self.V_store_dense = None

        # Choose Hamiltonian
        if Hamiltonian == "Kitaev":
            self.ham_builder = H_Kitaev_Chain
            self.dim = 2**N
            self.ntot = precompute_ntot(N)
        elif Hamiltonian == "BdG":
            self.ham_builder = H_BdG
            self.dim = 2*N
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
        t0 = time.perf_counter()
        E, V = np.linalg.eigh(H)
        t1 = time.perf_counter()
        self.time_diag_H += (t1 - t0)
        return E, V

    def build_H(self, t):
        t0 = time.perf_counter()
        if self.ham_builder is H_Kitaev_Chain:
            H = self.ham_builder(self.N, t, self.tmax,
                                 self.omega_fun, self.mu_fun, self.delta_fun,
                                 self.ntot)
        else:
            H = self.ham_builder(self.N, t, self.tmax,
                                 self.omega_fun, self.mu_fun, self.delta_fun)
        t1 = time.perf_counter()
        self.time_build_H += (t1 - t0)
        return H

    def initial_eigenpairs(self, T):
        H0 = self.build_H(T[0])
        E0, V0 = self.Diag_H(H0)

        H1 = self.build_H(T[1])
        E1, V1 = self.Diag_H(H1)

        return E0, V0, E1, V1

    # Degeneracy detection 
    def find_gs_manifold_indices(self, E, eps=None):
        """
        Given eigenvalues E, return indices of all states whose energy
        lies within eps of the minimum.
        """
        if eps is None:
            eps = self.eps_degen

        E = np.asarray(E)
        idx_sorted = np.argsort(E)
        E_sorted = E[idx_sorted]
        E0 = E_sorted[0]
        mask = (E_sorted - E0) < eps
        idx_degen_sorted = idx_sorted[mask]
        return idx_degen_sorted

    def compute_P0_inst_from_precomputed(self, a_t_inst, E_store_dense=None, eps=None):
        """
        For Schrödinger_precomputed: given a_t_inst (coeffs in precomputed
        instantaneous eigenbasis), compute gs-manifold probability and degeneracy.
        """
        if E_store_dense is None:
            if self.E_store_dense is None:
                raise RuntimeError("E_store_dense not found. Did you call prepare_precomputed_dense?")
            E_store_dense = self.E_store_dense

        if eps is None:
            eps = self.eps_degen

        a_t_inst = np.asarray(a_t_inst)
        Nt = a_t_inst.shape[0]

        P0_inst = np.zeros(Nt, dtype=float)
        degeneracies = np.zeros(Nt, dtype=int)

        for n in range(Nt):
            E_n = E_store_dense[n]
            idx_degen = self.find_gs_manifold_indices(E_n, eps=eps)
            degeneracies[n] = len(idx_degen)
            P0_inst[n] = np.sum(np.abs(a_t_inst[n, idx_degen])**2)

        return P0_inst, degeneracies

    def compute_P0_inst_from_adiabatic(self, a_t, E_all, eps=None):
        """
        Generic helper: from adiabatic amplitudes a_t and eigenvalues E_all,
        compute gs-manifold probability and degeneracy.
        """
        if eps is None:
            eps = self.eps_degen

        a_t = np.asarray(a_t)
        E_all = np.asarray(E_all)
        Nt = a_t.shape[0]

        P0_inst = np.zeros(Nt, dtype=float)
        degeneracies = np.zeros(Nt, dtype=int)

        for n in range(Nt):
            E_n = E_all[n]
            idx_degen = self.find_gs_manifold_indices(E_n, eps=eps)
            degeneracies[n] = len(idx_degen)
            P0_inst[n] = np.sum(np.abs(a_t[n, idx_degen])**2)

        return P0_inst, degeneracies

    #Precompute for adiabatic (sparse) 

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

        t0_all = time.perf_counter()

        v0 = None   # startvektor for eigsh
        for idx, tau in enumerate(tau_grid):
            t_dummy = tau

            # Bygg sparse H direkte
            if self.ham_builder is H_Kitaev_Chain:
                H_sparse = H_Kitaev_Chain_sparse(self.N, t_dummy, 1.0,
                                                 self.omega_fun, self.mu_fun,
                                                 self.delta_fun, self.ntot)
            else:
                H_dense = H_BdG(self.N, t_dummy, 1.0,
                                self.omega_fun, self.mu_fun, self.delta_fun)
                H_sparse = csr_matrix(H_dense)

            try:
                if self.warm_start and v0 is not None:
                    E_part, V_part = eigsh(H_sparse, k=dim_trunc, which="SA",
                                           v0=v0, tol=1e-10)
                else:
                    E_part, V_part = eigsh(H_sparse, k=dim_trunc, which="SA",
                                           tol=1e-10)
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

            idx_sorted = np.argsort(E_part)
            E_sorted = E_part[idx_sorted]
            V_sorted = V_part[:, idx_sorted]

            E_store[idx] = E_sorted
            V_store[idx] = V_sorted

            if self.warm_start:
                v0 = V_sorted[:, 0]

            if self.stride and (idx % 10 == 0):
                print(f"[Precompute] Step {idx} / {Nt_tau-1}. tau = {tau:.3f}")

        t1_all = time.perf_counter()
        self.time_precompute += (t1_all - t0_all)

        return tau_grid, E_store, V_store

    #Adiabatic with precomputed eigenpairs 
    def run_adiabatic_precomputed(self, T, E_store, V_store):
        """
        Adiabatic evolusjon der eigenpar (E,V) er precomputet på samme
        tidsgitter (tau-grid) som T er basert på.

        Returnerer:
        - T: fysisk tidsgitter
        - P0_inst: sum |a_k(t)|^2 over gs-manifold (instantan basis)
        - P0_orig_phys: |<psi0 | psi(t)>|^2 i full basis (psi0 gs ved t=0)
        - degeneracies: gs-degenerasjon per tid
        - (valgfritt) a_t, E_all hvis return_full=True
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

        # Finn fysisk grunntilstand ved t=0 i full basis
        H0_full = self.build_H(0.0)
        E0_full, V0_full = self.Diag_H(H0_full)
        psi0 = V0_full[:, 0]
        P0_orig_phys = np.zeros(Nt, dtype=float)
        P0_orig_phys[0] = 1.0

        # initiale eigenpar (truncerte)
        E0_raw = E_store[0]
        V0_raw = V_store[0]

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

        # n = 0: forward difference
        A0 = psi_dotpsi_forward(V0, V1, dt)
        lambda0 = lambda_from_E_and_A(E0, A0)
        a_t[1], phase_int = self.stepper(a_t[0], phase_int, A0, lambda0, dt)

        # psi(t1) i full basis
        I_k_1 = -phase_int[0, :]
        phase_factors_1 = np.exp(-1j * I_k_1)
        a_phys_1 = phase_factors_1 * a_t[1]
        psi_trunc_1 = V1 @ a_phys_1
        P0_orig_phys[1] = np.abs(np.vdot(psi0, psi_trunc_1))**2

        V_prev, V_curr = V0, V1
        E_curr = E1

        t_loop_start = time.perf_counter()

        for n in range(1, Nt - 1):
            E_next_raw = E_store[n+1]
            V_next_raw = V_store[n+1]

            V_next_matched, perm = match_eigenvectors(V_curr, V_next_raw)
            E_next = E_next_raw[perm]
            V_next_fixed = fix_phases(V_curr, V_next_matched)
            V_next = V_next_fixed

            E_all[n+1] = E_next

            A_n = central_A(V_prev, V_curr, V_next, dt)
            lambda_n = lambda_from_E_and_A(E_curr, A_n)

            a_t[n+1], phase_int = self.stepper(a_t[n], phase_int, A_n, lambda_n, dt)

            V_prev, V_curr = V_curr, V_next
            E_curr = E_next

            # rekonstruksjon i original basis
            I_k = -phase_int[0, :]
            phase_factors = np.exp(-1j * I_k)
            a_phys = phase_factors * a_t[n+1]

            B = V0.conj().T @ V_curr
            b_t[n+1] = B @ a_phys

            psi_trunc = V_curr @ a_phys
            P0_orig_phys[n+1] = np.abs(np.vdot(psi0, psi_trunc))**2

            if self.stride and (n % self.stride == 0):
                print(f"[Adiabatic-precomputed] Step {n} / {Nt-1}. T = {self.tmax}")

        t_loop_end = time.perf_counter()
        self.time_evolution_loop += (t_loop_end - t_loop_start)

        # gs-manifold sannsynlighet + degenerasjon
        P0_inst, degeneracies = self.compute_P0_inst_from_adiabatic(a_t, E_all)

        if self.return_full:
            return T, P0_inst, P0_orig_phys, degeneracies, a_t, E_all
        else:
            return T, P0_inst, P0_orig_phys, degeneracies

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

        T = self.tau_grid * self.tmax
        dt_phys = T[1] - T[0]

        dt_old = self.dt
        self.dt = dt_phys

        result = self.run_adiabatic_precomputed(T, self.E_store, self.V_store)

        self.dt = dt_old
        return result

    #Adiabatic (dense, no precompute) 

    def run_adiabatic(self, T):
        """
        Run time evolution on time grid T (full dim).
        Return:
        - T
        - P0_inst: gs-manifold probability
        - b_t: coefficients in basis of eigenstates at t=0
        - degeneracies: detected gs degeneracy vs time
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

            I_k = -phase_int[0, :]
            phase_factors = np.exp(-1j * I_k)
            a_phys = phase_factors * a_t[n+1]

            B = V0.conj().T @ V_curr
            b_t[n+1] = B @ a_phys

            if self.stride and (n % self.stride == 0):
                print(f"[Adiabatic] Step {n} / {Nt-1}. T = {self.tmax}")
        t_loop_end = time.perf_counter()

        self.time_evolution_loop += (t_loop_end - t_loop_start)

        P0_inst, degeneracies = self.compute_P0_inst_from_adiabatic(a_t, E_all)

        return T, P0_inst, b_t, degeneracies

    #Precompute dense for Schrödinger_precomputed 

    def prepare_precomputed_dense(self, Nt_tau):
        """
        Dense precompute of E(τ), V(τ) for τ in [0,1] on a Nt_tau grid.
        Uses t = τ, tmax = 1.0 in the Hamiltonian, so the eigenpairs depend
        only on τ = t/tmax and can be reused for all physical tmax.
        """
        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        dim = self.dim

        E_store = np.zeros((Nt_tau, dim))
        V_store = np.zeros((Nt_tau, dim, dim), dtype=complex)

        t0_all = time.perf_counter()
        for idx, tau in enumerate(tau_grid):
            if self.ham_builder is H_Kitaev_Chain:
                H = H_Kitaev_Chain(self.N, tau, 1.0,
                                   self.omega_fun, self.mu_fun, self.delta_fun,
                                   self.ntot)
            else:
                H = H_BdG(self.N, tau, 1.0,
                          self.omega_fun, self.mu_fun, self.delta_fun)

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

    #Schrödinger with precomputed eigenbasis

    def run_schrodinger_with_precomputed(self, _=None):
        """
        Schrödinger evolution using precomputed dense eigenbases on the same tau-grid.

        Requires:
        - self.tau_grid, self.E_store_dense, self.V_store_dense from prepare_precomputed_dense().
        Returns:
        - T, a_t_inst, psi_t
          (use compute_P0_inst_from_precomputed to get P0_inst and degeneracies)
        """
        if self.tau_grid is None or self.V_store_dense is None:
            raise RuntimeError("Call prepare_precomputed_dense(Nt_tau) first.")

        tau_grid = self.tau_grid
        V_store = self.V_store_dense

        Nt = len(tau_grid)
        dim = self.dim

        T = tau_grid * self.tmax
        dt = T[1] - T[0]

        psi_t = np.zeros((Nt, dim), dtype=complex)
        a_t_inst = np.zeros((Nt, dim), dtype=complex)

        H0 = self.build_H(0.0)
        E0, V0 = self.Diag_H(H0)
        psi0 = V0[:, 0]
        psi_t[0] = psi0

        a_t_inst[0] = V_store[0].conj().T @ psi_t[0]

        for n in range(Nt - 1):
            t_n = T[n]

            if self.sch_stepper is None:
                raise NotImplementedError("Schrodinger evolution not implemented for this integrator.")

            psi_t[n+1] = self.sch_stepper(psi_t[n], t_n, dt, self.build_H)

            norm = np.linalg.norm(psi_t[n+1])
            psi_t[n+1] /= norm

            V_inst = V_store[n+1]
            a_t_inst[n+1] = V_inst.conj().T @ psi_t[n+1]

            if self.stride and (n % self.stride == 0):
                print(f"[Schrodinger-precomputed] Step {n} / {Nt-1}. T = {self.tmax}, norm={norm}")

        return T, a_t_inst, psi_t

    #Schrödinger (dense, no precompute) 

    def _run_schrodinger(self, T):
        """
        Schrödinger evolution in the original basis.

        Returns:
        - T
        - P0_inst: gs-manifold probability in instantaneous basis
        - c_t: overlaps with eigenbasis at t=0 (c_t[:,0] is overlap with psi0)
        - degeneracies: detected gs degeneracy vs time
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

        P0_inst, degeneracies = self.compute_P0_inst_from_adiabatic(a_t, E_all)

        return T, P0_inst, c_t, degeneracies

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

def H_Kitaev_Chain_sparse(N, t, tmax, omega_fun, mu_fun, delta_fun, ntot_arr):
    """
    Ikke-numba-versjon som bygger en sparse (CSR) matrise for H_Kitaev.
    Brukes i precompute sammen med eigsh.
    """
    dim = 2**N
    mu = mu_fun(t, tmax)

    rows = []
    cols = []
    data = []

    for n in range(dim):
        ntot = int(ntot_arr[n])
        rows.append(n)
        cols.append(n)
        data.append(- mu * (ntot - N/2))

        for j in range(N-1):
            c1, m1 = a_on_basis_state(n, j + 1)
            if m1 != -1:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    val = -omega_fun(j, N, t, tmax) * c1 * c2
                    if val != 0.0:
                        rows.append(m2)
                        cols.append(n)
                        data.append(val)

            c3, m3 = a_on_basis_state(n, j)
            if m3 != -1:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 != -1:
                    val = -omega_fun(j, N, t, tmax) * c3 * c4
                    if val != 0.0:
                        rows.append(m4)
                        cols.append(n)
                        data.append(val)

            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 != -1:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    val = delta_fun(j, N, t, tmax) * c5 * c6
                    if val != 0.0:
                        rows.append(m6)
                        cols.append(n)
                        data.append(val)

            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 != -1:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 != -1:
                    val = delta_fun(j, N, t, tmax).conjugate() * c7 * c8
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


"Time dependencies"
@njit
def omega_Eivind_4_2(j, N, t, tmax):
    omega0 = 1
    j_c = N//2 - 1
    if j == j_c:
        return 2 * omega0 * np.abs(t/tmax - 0.5)
    else:
        return omega0

@njit
def mu_Eivind_4_2(t, tmax):
    mu0 = 0.5
    # return mu0
    return 2 * mu0  #For å matche Eivind, anderledes notasjon fra meg ( / Kitaev)

@njit
def delta_Eivind_4_2(j, N, t, tmax):
    delta0 = 2
    j_c = N//2 - 1
    if j == j_c:
        return 2 * delta0 * np.abs(t/tmax - 0.5)
    else:
        return delta0


def truncation():   #Used to find appropriate truncation 
    dt_tau = 1e-2
    Nt_tau = int(1.0/dt_tau) + 1

    eps = 1e-5 #Want to keep (1-eps) of the total probability
    k_needed_all = []

    tmax_list = [0.01, 0.1, 1]

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic_precomputed"
    return_full = True
    stride = 10
    trunc_dim = 2**N - 2

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
                   return_full=return_full,
                   trunc_dim=trunc_dim)

    _ = model.build_H(0.0)
    model.prepare_precomputed(Nt_tau)

    plt.figure(figsize=(6,4))

    for tmax_ in tmax_list:
        model.tmax = tmax_

        print(f"Running adiabatic (precomputed) for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        # full return: T, P0_inst, P0_orig_phys, degeneracies, a_t, E_all
        T_run, P0_inst, P0_orig_phys, degeneracies, a_t_full, E_all = model.run(T=None)
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
    dt_tau = 1e-2
    Nt_tau = int(1.0/dt_tau) + 1
    stride = 10

    tmax_list = [0.01, 0.1, 0.5, 1, 5, 10, 20]
    trunc = 136 # example truncation

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic_precomputed"
    return_full = False

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
                   return_full=return_full,
                   trunc_dim=trunc)

    _ = model.build_H(0.0)
    model.prepare_precomputed(Nt_tau)

    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
        model.tmax = tmax_

        print(f"Running Adiabatic_precomputed (trunc={trunc}) for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        # T_run, P0_inst, P0_orig_phys, degeneracies
        T_run, P0_adiab, P0_orig_phys, degeneracies = model.run(T=None)
        t_global_end = time.perf_counter()
        print(f"  Tid: {t_global_end - t_global_start:.3f} s\n")

        tau = T_run / tmax_

        plt.subplot(1,2,2)
        plt.plot(tau, P0_adiab, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P_0^\text{inst}$")
        plt.title("Adiabatisk basis (truncert)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig_phys, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = | \langle \psi_0 | \psi(t)\rangle |^2$")
        plt.title("Original basis (projisert på truncert rom)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{len(tmax_list)}_Adiabatic_precomputed_trunc{trunc}_{Hamiltonian_type}_{Integrator}_{N}.pdf")
    plt.show()

def schrodinger():
    dt_tau = 1e-2
    Nt_tau = int(1.0/dt_tau) + 1
    tau_grid = np.linspace(0.0, 1.0, Nt_tau)

    stride = 10

    tmax_list = [0.01, 0.5]

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Schrodinger"
    return_full = False  # not used here

    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
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
                       return_full=return_full,
                       trunc_dim=None)

        print(f"Running Schrodinger for tmax = {tmax_}...")
        t_global_start = time.perf_counter()
        # T_run, P0_inst, c_t, degeneracies
        T_run, P0_inst, c_t, degeneracies = model.run(T)
        t_global_end = time.perf_counter()
        print(f"  Tid: {t_global_end - t_global_start:.3f} s\n")

        tau = T_run / tmax_

        P0_orig = np.abs(c_t[:, 0])**2

        plt.subplot(1,2,2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P_0^\text{inst}$")
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
    dt_tau = 1e-2
    Nt_tau = int(1.0/dt_tau) + 1
    stride = 10
    tmax_list = [0.01, 0.1, 0.5, 1]

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    return_full = False
    evolution = "Schrodinger_precomputed"
    eps_degen_ = 1e-2   #Takes the states within the range lowest energy + eps as the lowest state (degenerations)

    pre_model = Kitaev(dt=dt_tau,
                       N=N,
                       stride=stride,
                       tmax=1.0,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=evolution,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       return_full=return_full,
                       trunc_dim=None,
                       eps_degen = eps_degen_)

    print("Precomputing dense eigenpairs on tau-grid (shared for all tmax)...")
    t0 = time.perf_counter()
    pre_model.prepare_precomputed_dense(Nt_tau)
    t1 = time.perf_counter()
    print(f"  Time to precompute: {t1 - t0:.3f} s\n")

    plt.figure(figsize=(8,4))

    for tmax_ in tmax_list:
        model = Kitaev(dt=dt_tau,
                       N=N,
                       stride=stride,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution="Schrodinger_precomputed",
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       return_full=return_full,
                       trunc_dim=None,
                       eps_degen=eps_degen_)

        model.tau_grid = pre_model.tau_grid
        model.E_store_dense = pre_model.E_store_dense
        model.V_store_dense = pre_model.V_store_dense

        print(f"Running Schrödinger (with precomputed basis) for tmax = {tmax_}...")
        t0_ = time.perf_counter()
        T_run, a_t_inst, psi_t = model.run(T=None)
        t1_ = time.perf_counter()
        print(f"  Time to run: {t1_ - t0_:.3f} s\n")

        tau = T_run / tmax_

        P0_inst, degeneracies = model.compute_P0_inst_from_precomputed(a_t_inst)

        print(f"Degeneracies in gs = {degeneracies}")

        P0_orig = np.zeros_like(tau)
        psi0 = psi_t[0]
        for n in range(len(tau)):
            P0_orig[n] = np.abs(np.vdot(psi0, psi_t[n]))**2

        plt.subplot(1,2,2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P_0^\text{inst}$")
        plt.title("Instantan basis (Schrödinger)")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |\langle \psi_0 | \psi(t)\rangle|^2$")
        plt.title("Original basis (Schrödinger)")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{len(tmax_list)}_Schrodinger_precomputed_{Hamiltonian_type}_{Integrator}_{N}.pdf")
    plt.show()



if __name__ == "__main__":
    # truncation()
    # adiabatic_approx()
    # schrodinger()
    schrodinger_approx_with_instantaneous()
