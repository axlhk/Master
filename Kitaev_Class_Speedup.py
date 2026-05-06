import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from numba import njit
import time 


class Kitaev:
    "dt     :       timestep"
    "tmax   :       Total time T"
    "N      :       number of sites"
    "Hamiltonian:   Kitaev or BdG"
    "Integration:   (Euler, Euler_Cromer) or RK4"
    "Evolution: Adiabatic or Schrodinger"
    "omega, mu, delta: functions(j, N, t, tmax)" 
    "return_full:   True or False, to return all a_t's, or just the gs."

    def __init__(self, dt, N, tmax, Hamiltonian, Integration, Evolution, omega_fun, mu_fun, delta_fun, exp_degen, return_full):

        self.dt   = dt
        self.N    = N
        self.tmax = tmax
        self.exp_degen = exp_degen
        self.return_full = return_full

        self.omega_fun = omega_fun
        self.mu_fun    = mu_fun
        self.delta_fun = delta_fun

        #For timing
        self.time_build_H = 0.0
        self.time_diag_H  = 0.0
        self.time_evolution_loop = 0.0

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
        
        # Choose run function for the evolution type 
        self.evolution_type = Evolution
        if Evolution == "Adiabatic":
            self.run_method = self.run_adiabatic
        elif Evolution == "Schrodinger":
            self.run_method = self._run_schrodinger
        else:
            raise ValueError(
                f"Unknown evolution_type: {Evolution}. Use Adiabatic or Schrodinger.")
        
    def run(self, T, stride=10):    #For easy calling
        return self.run_method(T, stride=stride)
    
    def Diag_H(self, H): 
        t0 = time.perf_counter()
        E, V = np.linalg.eigh(H)
        t1 = time.perf_counter()
        self.time_diag_H += (t1 - t0)
        return E, V
        # return eigsh(H, k = self.exp_degen, which = "SA")   #Smallest algebraic??

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

    def precompute_eigensystem_tau(self, Nt_tau):
        """
        Precompute E(τ) og V(τ) for τ ∈ [0, 1], på et Nt_tau-gitter.

        Vi bruker t = τ, tmax = 1.0 i H-byggeren. For din nåværende
        protokoll (omega, delta som funksjon av t/tmax) betyr det at
        disse eigenparene gjelder for alle tmax.
        """
        tau_grid = np.linspace(0.0, 1.0, Nt_tau)
        dim = self.dim

        E_store = np.zeros((Nt_tau, dim))
        V_store = np.zeros((Nt_tau, dim, dim), dtype=complex)

        # Viktig: vi vil ikke blande inn timingakkumulatorene her
        # (de brukes til runtime-measurements per run), så vi kaller
        # H_Kitaev_Chain "rått" i stedet for self.build_H
        for idx, tau in enumerate(tau_grid):
            t_dummy = tau
            if self.ham_builder is H_Kitaev_Chain:
                H = H_Kitaev_Chain(self.N, t_dummy, 1.0,
                                   self.omega_fun, self.mu_fun,
                                   self.delta_fun, self.ntot)
            else:
                H = H_BdG(self.N, t_dummy, 1.0,
                          self.omega_fun, self.mu_fun,
                          self.delta_fun)
            # Bruk Diag_H for å få samme numeriske rutine (eigh)
            E, V = self.Diag_H(H)
            E_store[idx] = E
            V_store[idx] = V

            
            if 10 and (idx % 10 == 0):
                print(f"[Precompute] idx = {idx} / {tau_grid.shape}. tau = {tau}")

        return tau_grid, E_store, V_store

    def run_adiabatic_precomputed(self, T, E_store, V_store, stride=10):
        """
        Adiabatic evolusjon der eigenpar (E,V) er precomputet på samme
        tidsgitter (tau-grid) som T er basert på.

        Forutsetter:
        - len(T) == E_store.shape[0] == V_store.shape[0]
        - E_store[n], V_store[n] tilsvarer eigenparene ved T[n].
        """
        dt  = self.dt
        dim = self.dim
        Nt  = len(T)

        if Nt != E_store.shape[0]:
            raise ValueError("T og E_store/V_store må ha samme lengde")

        a_t = np.zeros((Nt, dim), dtype=complex)
        a_t[0, 0] = 1.0

        b_t = np.zeros((Nt, dim), dtype=complex)
        b_t[0] = a_t[0]

        E_all = np.zeros((Nt, dim))

        # initiale eigenpar
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

        phase_int = np.zeros((dim, dim), dtype=complex)

        # n = 0: forward difference
        A0 = psi_dotpsi_forward(V0, V1, dt)
        lambda0 = lambda_from_E_and_A(E0, A0)
        a_t[1], phase_int = self.stepper(a_t[0], phase_int, A0, lambda0, dt)

        V_prev, V_curr = V0, V1
        E_curr = E1

        # tidsutvikling (ingen Diag_H/build_H her!)
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

            if stride and (n % stride == 0):
                print(f"[Adiabatic-precomputed] Step {n} / {Nt-1}. T = {self.tmax}")

        # reduksjon til exp_degen laveste
        k = self.exp_degen
        a_t_reduced = np.zeros((Nt, k), dtype=complex)
        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]
            idx_sorted = np.argsort(E)
            idx_gs = idx_sorted[:k]
            a_t_reduced[n, :] = d[idx_gs]

        if self.return_full:
            return T, a_t_reduced, b_t, a_t, E_all
        else:
            return T, a_t_reduced, b_t

    def run_adiabatic(self, T, stride):
        """
        Run time evolution on time grid T.
        Returns T, a_t, E_all, b_t (same meanings as before).
        """
        dt  = self.dt
        N   = self.N
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

            if stride and (n % stride == 0):
                print(f"[Adiabatic] Step {n} / {Nt-1}. T = {self.tmax}")
        t_loop_end = time.perf_counter()

        self.time_evolution_loop += (t_loop_end - t_loop_start)

        k = self.exp_degen
        a_t_reduced = np.zeros((Nt, k), dtype=complex)
        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]          # instantaneous coefficients at time T[n]
            idx_sorted = np.argsort(E)
            idx_gs = idx_sorted[:k]
            a_t_reduced[n, :] = d[idx_gs]

        # return reduced instantaneous amplitudes + full original-basis amplitudes
        return T, a_t_reduced, b_t

    def _run_schrodinger(self, T, stride):
        """
        Schrödinger evolution in the original basis:
            i d/dt |psi> = H(t) |psi>
        Returns:
            T, psi_t, a_t
        where
            psi_t[n] : coefficients in the original basis at time T[n]
            a_t[n]   : coefficients in instantaneous eigenbasis at T[n]
        """
        dt  = self.dt
        dim = self.dim
        Nt  = len(T)
        E_all = np.zeros((Nt, dim))               #For the degenerate states

        #basis coefficients
        psi_t = np.zeros((Nt, dim), dtype=complex)
        a_t = np.zeros((Nt, dim), dtype=complex)

        # at t = T[0], build H and instantaneous basis, then project psi0
        H0 = self.build_H(T[0])
        E0, V0 = self.Diag_H(H0)
        E_all[0] = E0

        psi0 = V0[:, 0]          #Ground state
        psi_t[0] = psi0
        a_t[0] = V0.conj().T @ psi0   # psi0 in instantaneous eigenbasis

        for n in range(Nt - 1):
            t_n = T[n]

            if self.sch_stepper is None:
                raise NotImplementedError(
                    "Schrödinger evolution not implemented for this integrator."
                )

            # Integrate psi in the original basis
            psi_t[n+1] = self.sch_stepper(psi_t[n], t_n, dt, self.build_H)

            # Build instantaneous basis at t_{n+1}
            t_next = T[n+1]
            H_next = self.build_H(t_next)
            E_next, V_next = self.Diag_H(H_next)

            E_all[n+1] = E_next           #Store eigenvalues

            # Project psi(t_{n+1}) onto instantaneous eigenbasis
            a_t[n+1] = V_next.conj().T @ psi_t[n+1]

            if stride and (n % stride == 0):
                print(f"[Schrödinger] Step {n} / {Nt-1}. T = {self.tmax}")

        c_t = psi_t @ V0.conj()     #original basis

        k = self.exp_degen
        a_t_reduced = np.zeros((Nt, k), dtype=complex)

        for n in range(Nt):
            E = E_all[n]
            d = a_t[n]                   # full instantan koeff-vektor ved T[n]
            idx_sorted = np.argsort(E)   # laveste energi først
            idx_gs = idx_sorted[:k]      # indekser til de k laveste nivåene
            a_t_reduced[n, :] = d[idx_gs]

        return T, a_t_reduced, c_t

"Functions used in run_adiabatic() in class"
def psi_dotpsi_forward(V0, V1, dt): #Initial psi_dot_psi for t0 and t1 by forward difference 
    overlap_01 = V0.conj().T @ V1
    A0 = (overlap_01 - np.eye(V0.shape[1], dtype=complex)) / dt
    return A0

def lambda_from_E_and_A(E, A): #Returns lamba as analytically defined (A = psi_dotpsi))
    return E - 1j * np.diag(A)

def match_eigenvectors(V_curr, V_next_raw): #eigh doesn't spit out eigenvectors in the right order. FROM CHAT. READ UP ON
    """
    Reorder columns of V_next_raw to best match V_curr by maximal overlap.
    Returns V_next_matched with columns permuted and phases unadjusted.
    """
    dim = V_curr.shape[1]
    # Overlap matrix O_{kl} = <ψ_k(curr)|ψ_l(next_raw)>
    O = V_curr.conj().T @ V_next_raw  # shape (dim, dim)

    # For each k in curr, find l in next_raw with max |O_{kl}|
    # This is a greedy matching; Hungarian algorithm would be optimal,
    # but greedy is usually fine if steps are small.
    used_next = set()
    perm = [-1] * dim
    for k in range(dim):
        overlaps = np.abs(O[k, :])
        # mask already used columns
        for l in used_next:
            overlaps[l] = -1.0
        l_best = np.argmax(overlaps)
        perm[k] = l_best
        used_next.add(l_best)

    # Apply permutation
    V_next_perm = V_next_raw[:, perm]
    return V_next_perm, perm

def fix_phases(V_curr, V_next): #Same as match_egenvectors but for phases.
    """
    Adjust phases of columns of V_next so that
    <ψ_k(curr) | ψ_k(next)> is real and positive.
    """
    dim = V_curr.shape[1]
    for k in range(dim):
        overlap = np.vdot(V_curr[:, k], V_next[:, k])  # <curr|next>
        phase = np.angle(overlap)
        V_next[:, k] *= np.exp(-1j * phase)
    return V_next

def central_A(V_prev, V_curr, V_next, dt):  #psi_dot_psi
    overlap_n_next = V_curr.conj().T @ V_next
    overlap_n_prev = V_curr.conj().T @ V_prev
    A_n = (overlap_n_next - overlap_n_prev) / (2 * dt)
    return A_n


"Kitaev Hamiltonian and helper functions"
@njit
def bit(n, i):
    """
    Return the occupation (0 or 1) of site i in basis state |n>,
    where i = 0,...,L-1.
    """
    return (n >> i) & 1

@njit
def flip_bit(n, i):
    """Flip bit i of n."""
    return n ^ (1 << i)

@njit
def fermion_sign(n, i):
    """
    Jordan-Wigner-sign: (-1)^{sum_{j<i} n_j}
    Teller antall 1-bits i de laveste i bitene.
    """
    mask = (1 << i) - 1
    x = n & mask
    cnt = 0
    while x:
        cnt += x & 1
        x >>= 1
    return -1 if (cnt % 2 == 1) else 1

# def fermion_sign(n, i): #Enforces anti-commutation
#     r"""
#     Jordan-Wigner sign: (-1)^{sum_{j < i} n_j}.
#     """
#     mask = (1 << i) - 1  # bits 0,...,i-1
#     num_ones = bin(n & mask).count("1")
#     return -1 if (num_ones % 2 == 1) else 1

@njit
def a_dag_on_basis_state(n, i):
    """
    Action of c_i^\dagger on |n>.

    Returns (coef, m), where result is coef * |m>, or (0, None) if annihilated.
    """
    if bit(n, i) == 1:
        return 0.0, -1  # already occupied
    s = fermion_sign(n, i)
    m = flip_bit(n, i)
    return s, m

@njit
def a_on_basis_state(n, i):
    """
    Action of c_i on |n>.

    Returns (coef, m), where result is coef * |m>, or (0, None) if annihilated.
    """
    if bit(n, i) == 0:
        return 0.0, -1  # empty site
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
        # Number term (diagonal)
        ntot = ntot_arr[n]
        H[n, n] += - mu * (ntot - N/2)


        # Hopping and pairing (off-diagonal)
        for j in range(N-1):
            # Omega terms
            # a_j^dagger a_{j+1}
            c1, m1 = a_on_basis_state(n, j + 1)
            if m1 != -1:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 != -1:
                    H[m2, n] += -omega_fun(j, N, t, tmax) * c1 * c2

            # a_{j+1}^dagger a_j
            c3, m3 = a_on_basis_state(n, j)
            if m3 != -1:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 != -1:
                    H[m4, n] += -omega_fun(j, N, t, tmax) * c3 * c4

            # Delta terms
            # a_j a_{j+1}
            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 != -1:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 != -1:
                    H[m6, n] += delta_fun(j, N, t, tmax) * c5 * c6

            # a_{j+1}^dagger a_j^dagger
            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 != -1:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 != -1:
                    H[m8, n] += delta_fun(j, N, t, tmax).conjugate() * c7 * c8

    return H


def H_BdG(N, t, tmax, omega_fun, mu_fun, delta_fun):    #FROM CHAT. READ UP ON
    mu = mu_fun(t, tmax)

    # Normal part h(t)
    h = np.zeros((N, N), dtype=complex)
    for j in range(N):
        h[j, j] = -mu
        if j < N - 1:
            h[j, j+1] = -omega_fun(j, N, t, tmax)
            h[j+1, j] = -omega_fun(j, N, t, tmax)

    # Pairing part Δ(t)
    Delta = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        Delta[j, j+1] = delta_fun(j, N, t, tmax)
        Delta[j+1, j] = -delta_fun(j, N, t, tmax)  # p-wave sign structure

    # BdG matrix
    upper = np.hstack((h, Delta))
    lower = np.hstack((Delta.conj().T, -h.T))
    H_bdg = np.vstack((upper, lower))

    return H_bdg


"Schrodinger integrators and helper functions"  #ALL FROM CHAT. READ UP ON
def rhs_schrodinger(c, H):
    """Right-hand side of Schrödinger equation: dc/dt = -i H c."""
    return -1j * (H @ c)

def schrodinger_step_euler(c_n, t_n, dt, build_H):
    """
    Explicit Euler step for Schrödinger evolution with
    piecewise-constant H(t) ≈ H(t_n) on [t_n, t_n+dt].
    """
    H_n = build_H(t_n)
    dc_dt = rhs_schrodinger(c_n, H_n)
    c_next = c_n + dt * dc_dt
    return c_next

def schrodinger_step_RK4(c_n, t_n, dt, build_H):
    """
    RK4 step for Schrödinger evolution with time-dependent H(t).
    """
    # k1
    H1 = build_H(t_n)
    k1 = rhs_schrodinger(c_n, H1)

    # k2
    t2 = t_n + 0.5*dt
    H2 = build_H(t2)
    c2 = c_n + 0.5*dt*k1
    k2 = rhs_schrodinger(c2, H2)

    # k3
    c3 = c_n + 0.5*dt*k2
    k3 = rhs_schrodinger(c3, H2)

    # k4
    t4 = t_n + dt
    H4 = build_H(t4)
    c4 = c_n + dt*k3
    k4 = rhs_schrodinger(c4, H4)

    c_next = c_n + (dt/6.0) * (k1 + 2*k2 + 2*k3 + k4)
    return c_next

"Adiabatic integrators and helper functions"
def evolution_step_euler(a_n, phase_int, A, lambd, dt):   #One step using Euler integration of the analytical expression for da(t)/dt
    phase_matrix = np.exp(-1j * phase_int)  #The exponential in the rhs of the ODE for a

    # M_{lk} = phase_{lk} * A_{lk}, but set diagonal to 0 (k != l only)
    M = phase_matrix * A
    np.fill_diagonal(M, 0.0)

    # RHS = sum_k M_{lk} a_k
    rhs = M @ a_n

    # Euler step
    a_next = a_n + dt * rhs

    # Update integral: delta_lambda_{lk} = lambda_l - lambda_k
    delta_lambda = lambd[:, None] - lambd[None, :]
    phase_int_next = phase_int + delta_lambda * dt

    return a_next, phase_int_next

def evolution_step_euler_cromer():  #Not implemented 
    a_next = 1
    phase_int_next = 1
    return a_next, phase_int_next

def rhs_phase(a, phase_int, A, lambd): #Right hand side of da/dt and the phase lambda_l - lambda_k
    # Phase factors e^{-i I}
    phase_matrix = np.exp(1j * phase_int)

    # M_{lk} = phase_{lk} * A_{lk}, zero diagonal
    M = phase_matrix * A
    np.fill_diagonal(M, 0.0)

    # da/dt = M @ a
    da_dt = -M @ a

    # dI/dt = Δλ = λ_l - λ_k
    delta_lambda = lambd[:, None] - lambd[None, :]
    dI_dt = delta_lambda

    return da_dt, dI_dt

def evolution_step_RK4(a_n, phase_int, A, lambd, dt):
    # k1
    k1_a, k1_I = rhs_phase(a_n, phase_int, A, lambd)

    # k2
    a2 = a_n + 0.5 * dt * k1_a
    I2 = phase_int + 0.5 * dt * k1_I
    k2_a, k2_I = rhs_phase(a2, I2, A, lambd)

    # k3
    a3 = a_n + 0.5 * dt * k2_a
    I3 = phase_int + 0.5 * dt * k2_I
    k3_a, k3_I = rhs_phase(a3, I3, A, lambd)

    # k4
    a4 = a_n + dt * k3_a
    I4 = phase_int + dt * k3_I
    k4_a, k4_I = rhs_phase(a4, I4, A, lambd)

    # Combine
    a_next = a_n + (dt / 6.0) * (k1_a + 2*k2_a + 2*k3_a + k4_a)
    phase_int_next = phase_int + (dt / 6.0) * (k1_I + 2*k2_I + 2*k3_I + k4_I)

    return a_next, phase_int_next

"Time dependencies"
@njit
def omega_Eivind_4_2(j, N, t, tmax):
    omega0 = 1

    """
    Hopping on bond j ↔ j+1 at time t.
    j runs from 0 to N-2.
    """
    j_c = N//2 - 1  # central bond index for even N
    if j == j_c:
        # central bond: ramp to 0 in the middle
        return 2 * omega0 * np.abs(t/tmax - 0.5)
    else:
        # all other bonds: keep constant (or weakly varying if you want)
        return omega0

@njit
def mu_Eivind_4_2(t, tmax):
    mu0 = 0.5
    return mu0

@njit
def delta_Eivind_4_2(j, N, t, tmax):
    """
    Pairing on bond j ↔ j+1 at time t.
    """
    delta0 = 2
    j_c = N//2 - 1
    if j == j_c:
        # central bond: ramp to 0 in the middle
        return 2 * delta0 * np.abs(t/tmax - 0.5)
    else:
        return delta0

def Eivind_1():
    dt_tau = 1e-2       # steg i tau = t/tmax
    Nt_tau = int(1.0/dt_tau) + 1   # samme for alle tmax

    t0 = 0.0
    tmax_list = [0.01, 0.1, 1, 5]  # eksempel

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic"
    exp_degen = 4
    return_full = False

    # 1) Bygg en "referansemodell" for precompute, med tmax = 1.0
    model_ref = Kitaev(dt=dt_tau,
                       N=N,
                       tmax=1.0,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=Evolution_type,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full)

    # Warmup Numba-kompilering
    _ = model_ref.build_H(0.0)

    print("Precomputing eigenpairs along tau...")
    tau_grid, E_store, V_store = model_ref.precompute_eigensystem_tau(Nt_tau)
    print("Done precomputing.\n")

    plt.figure(figsize=(8,4))

    for t_, tmax_ in enumerate(tmax_list):
        # 2) For hvert tmax: samme tau-grid, men andre fysiske tider
        T = tau_grid * tmax_

        # dt for denne kjøringen: fysisk tidsteg
        dt = T[1] - T[0]

        model = Kitaev(dt=dt,
                       N=N,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=Evolution_type,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen)

        print(f"Running adiabatic evolution (precomputed) for tmax = {tmax_}...")

        t_global_start = time.perf_counter()
        T_run, a_t, c_t = model.run_adiabatic_precomputed(T, E_store, V_store, stride=0)
        t_global_end = time.perf_counter()

        print(f"  Tid total (uten diagonaliseringskostnad per tmax): "
              f"{t_global_end - t_global_start:.3f} s\n")

        # Plotting
        tau = T_run / tmax_
        P0 = np.sum(np.abs(a_t)**2, axis=1)
        P0_orig = np.abs(c_t[:, 0])**2

        plt.subplot(1,2,2)
        plt.plot(tau, P0, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |a_0|^2$")
        plt.title("Adiabatic basis")
        plt.legend()

        plt.subplot(1,2,1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |n_0|^2$")
        plt.title("Original basis")
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{len(tmax_list)}_Adiabatic_{Hamiltonian_type}_{Integrator}_{N}_precomputed.pdf")
    plt.show()
# Eivind_1()
def truncation():
    dt_tau = 1e-2       # steg i tau = t/tmax
    Nt_tau = int(1.0/dt_tau) + 1   # samme for alle tmax

    eps = 1e-4  #Truncation: Will keep lowest energy states containing (1-epsilon) of the probability 
    k_needed_all = []

    tmax_list = [0.01, 0.1, 1, 5]

    N = 10
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    Evolution_type = "Adiabatic"
    exp_degen = 4
    return_full = True

    # 1) Referansemodell for precompute, med tmax = 1.0
    model_ref = Kitaev(dt=dt_tau,
                       N=N,
                       tmax=1.0,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=Evolution_type,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full)

    # Warmup Numba
    _ = model_ref.build_H(0.0)

    print("Precomputing eigenpairs along tau...")
    tau_grid, E_store, V_store = model_ref.precompute_eigensystem_tau(Nt_tau)
    print("Done precomputing.\n")

    plt.figure(figsize=(6,4))

    for t_, tmax_ in enumerate(tmax_list):
        # 2) Tidsgitter for denne tmax
        T = tau_grid * tmax_
        dt = T[1] - T[0]

        model = Kitaev(dt=dt,
                       N=N,
                       tmax=tmax_,
                       Hamiltonian=Hamiltonian_type,
                       Integration=Integrator,
                       Evolution=Evolution_type,
                       omega_fun=omega_Eivind_4_2,
                       mu_fun=mu_Eivind_4_2,
                       delta_fun=delta_Eivind_4_2,
                       exp_degen=exp_degen,
                       return_full=return_full)

        print(f"Running adiabatic evolution (precomputed) for tmax = {tmax_}...")

        t_global_start = time.perf_counter()
        T_run, a_t_reduced, b_t, a_t_full, E_all = model.run_adiabatic_precomputed(
            T, E_store, V_store, stride=0
        )
        t_global_end = time.perf_counter()

        print(f"  Tid total (uten diagonaliseringskostnad per tmax): "
              f"{t_global_end - t_global_start:.3f} s\n")

        # --- trunceringsanalyse ved sluttid ---
        a_final = a_t_full[-1]   # koeffisienter i instantan egenbasis ved sluttid
        E_final = E_all[-1]      # energier ved sluttid

        # sorter tilstander etter energi
        idx_sorted = np.argsort(E_final)
        probs_sorted = np.abs(a_final[idx_sorted])**2

        cum = np.cumsum(probs_sorted)
        # finn minste k der tail = 1 - cum[k] < eps
        tail = 1.0 - cum
        k_needed = np.argmax(tail < eps)  # første indeks hvor tail < eps
        k_needed_all.append(k_needed + 1) # +1 fordi k er 0-basert


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
# truncation()


#SPM:
"Hvordan håndterer jeg degenererte tilstander?"
"- Implementerer nå en forvented mengde degenerasjon som da tar de x antall laveste tilstandene"
"Bruke forskjellige dt for forskjellige tmax? - T = 0.001 blir ubrukelig mens T = 20 blir veldig bra"
"- Evt. en skalering dt = dt / tmax"
"Hvilken dt brukte Eivind?"

"Speedup av kode"
"- Kun finne de første egenverdi og vektorene - i.e de degenererte - trunkering"
"- Behandle H som en sparse matrise(?)"
"- Mer effektiv diagonalisering? - warm guess; adiabatisk utvikling?"
"- Finn laveste tilstander, spar dem, beregn koeffisienter a_t for forskjellige tider tmax."

"Feil(?) i utledningen min av adiabatisk; byttet om på - tegn og det fikset det"

"Hva jeg vil nå:"
"1. Speede up kode"
"2. Se på adiabatisk forenkling (trunkering av tilstander)"
"3. Se på T-junctions"
"4. Skrive om koden i oppgaven?"

"Plotte trunkeringskurve - |a_n|**2 som punkter"