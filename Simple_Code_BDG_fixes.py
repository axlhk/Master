import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh


"Functions and constants used in making the Hamiltonian"

omega0 = 1
mu0 = 0.5
delta0 = 2

def omega_(j, N, t, tmax):
    """
    Hopping on bond j ↔ j+1 at time t.
    j runs from 0 to N-2.
    """
    j_c = N/2 - 1  # central bond index for even N
    if j == j_c:
        # central bond: ramp to 0 in the middle
        return 2 * omega0 * np.abs(t/tmax - 0.5)
    else:
        # all other bonds: keep constant (or weakly varying if you want)
        return omega0

def mu_(t, tmax):
    mu_start = -3
    mu_stop = 3
    # return (mu_start + (mu_stop - mu_start )* t / tmax)
    return mu0

def delta_(j, N, t, tmax):
    """
    Pairing on bond j ↔ j+1 at time t.
    """
    j_c = N//2 - 1
    if j == j_c:
        # central bond: ramp to 0 in the middle
        return 2 * delta0 * np.abs(t/tmax - 0.5)
    else:
        return delta0

def bit(n, i):
    """
    Return the occupation (0 or 1) of site i in basis state |n>,
    where i = 0,...,L-1.
    """
    return (n >> i) & 1

def flip_bit(n, i):
    """Flip bit i of n."""
    return n ^ (1 << i)

def fermion_sign(n, i): #Enforces anti-commutation
    r"""
    Jordan-Wigner sign: (-1)^{sum_{j < i} n_j}.
    """
    mask = (1 << i) - 1  # bits 0,...,i-1
    num_ones = bin(n & mask).count("1")
    return -1 if (num_ones % 2 == 1) else 1

def a_dag_on_basis_state(n, i):
    """
    Action of c_i^\dagger on |n>.

    Returns (coef, m), where result is coef * |m>, or (0, None) if annihilated.
    """
    if bit(n, i) == 1:
        return 0.0, None  # already occupied
    s = fermion_sign(n, i)
    m = flip_bit(n, i)
    return s, m

def a_on_basis_state(n, i):
    """
    Action of c_i on |n>.

    Returns (coef, m), where result is coef * |m>, or (0, None) if annihilated.
    """
    if bit(n, i) == 0:
        return 0.0, None  # empty site
    s = fermion_sign(n, i)
    m = flip_bit(n, i)
    return s, m

def H_Kitaev_Chain(N, t, tmax):
    """
    Returns the Kitaev Hamiltonian in the occupational basis for a given time t.
    H = sum - omega ( a_j^dagger a_{j+1} + a_{j+1}^dagger a_j )
        - mu (n_j - 1/2)
        + delta a_j a_{j+1} + delta^* a_{j+1}^dagger a_j^dagger
    """
    dim = 2**N
    mu = mu_(t, tmax)

    H = np.zeros((dim, dim), dtype=complex)

    for n in range(dim):
        # Number term (diagonal)
        ntot = 0
        for j in range(N):
            ntot += bit(n, j)
        H[n, n] += - mu * (ntot - N/2)

        # Hopping and pairing (off-diagonal)
        for j in range(N-1):
            # Omega terms
            # a_j^dagger a_{j+1}
            c1, m1 = a_on_basis_state(n, j + 1)
            if m1 is not None:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 is not None:
                    H[m2, n] += -omega_(j, N, t, tmax) * c1 * c2

            # a_{j+1}^dagger a_j
            c3, m3 = a_on_basis_state(n, j)
            if m3 is not None:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 is not None:
                    H[m4, n] += -omega_(j, N, t, tmax) * c3 * c4

            # Delta terms
            # a_j a_{j+1}
            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 is not None:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 is not None:
                    H[m6, n] += delta_(j, N, t, tmax) * c5 * c6

            # a_{j+1}^dagger a_j^dagger
            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 is not None:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 is not None:
                    H[m8, n] += (delta_(j, N, t, tmax).conjugate()) * c7 * c8

    return H

def H_BdG(N, t, tmax):
    
    mu    = mu_(t, tmax)

    # Normal part h(t)
    h = np.zeros((N, N), dtype=complex)
    for j in range(N):
        h[j, j] = -mu
        if j < N - 1:
            h[j, j+1] = -omega_(j, N, t, tmax)
            h[j+1, j] = -omega_(j, N, t, tmax)

    # Pairing part Δ(t)
    Delta = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        Delta[j, j+1]   =  delta_(j, N, t, tmax)
        Delta[j+1, j]   = -delta_(j, N, t, tmax)  # p-wave sign structure

    # BdG matrix
    upper = np.hstack((h,      Delta))
    lower = np.hstack((Delta.conj().T, -h.T))
    H_bdg = np.vstack((upper, lower))

    return H_bdg

def Diag_H(H):
    """Diagonalize Hermitian H: returns eigenvalues E and eigenvectors V."""
    E, V = np.linalg.eigh(H)
    # E, V = eigsh(H)

    return E, V

def initial_eigenpairs(T, tmax, N, dim):    #Initial eigenpairs for t0 and t1
    H0 = H_Kitaev_Chain(N, T[0], tmax)
    # H0 = H_BdG(N, T[0], tmax)
    E0, V0 = Diag_H(H0)

    H1 = H_Kitaev_Chain(N, T[1], tmax)
    # H1 = H_BdG(N, T[1], tmax)
    E1, V1 = Diag_H(H1)

    return E0, V0, E1, V1

def psi_dotpsi_forward(V0, V1, dt): #Initial psi_dot_psi for t0 and t1 by forward difference 
    overlap_01 = V0.conj().T @ V1
    A0 = (overlap_01 - np.eye(V0.shape[1], dtype=complex)) / dt
    return A0

def lambda_from_E_and_A(E, A): #Returns lamba as analytically defined (A = psi_dotpsi))
    return E - 1j * np.diag(A)

def match_eigenvectors(V_curr, V_next_raw):
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

def fix_phases(V_curr, V_next):
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
    phase_matrix = np.exp(-1j * phase_int)

    # M_{lk} = phase_{lk} * A_{lk}, zero diagonal
    M = phase_matrix * A
    np.fill_diagonal(M, 0.0)

    # da/dt = M @ a
    da_dt = M @ a

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

def central_A(V_prev, V_curr, V_next, dt):  #Normal psi_dot_psi
    overlap_n_next = V_curr.conj().T @ V_next
    overlap_n_prev = V_curr.conj().T @ V_prev
    A_n = (overlap_n_next - overlap_n_prev) / (2 * dt)
    return A_n

def run_evolution(t_, T, tmax, N, dim):
    Nt = len(T)
    dt = T[1] - T[0]

    # Allocate outputs
    a_t = np.zeros((Nt, dim), dtype=complex)
    a_t[0, 0] = 1.0  # initial ground-state amplitude in instantaneous basis

    b_t = np.zeros((Nt, dim), dtype=complex) #original basis
    # b_t[0] = V0.conj().T @ (np.ones(dim, dtype=complex)*0)  # or just (1,0,...) since V0 basis = original
    b_t[0,0] = 1.0  # clear: initial state is original ground state

    E_all = np.zeros((Nt, dim))

    # --- Initial eigenpairs at t0 and t1 (raw) ---
    H0 = H_Kitaev_Chain(N, T[0], tmax)
    # H0 = H_BdG(N, T[0], tmax)
    E0_raw, V0_raw = Diag_H(H0)
    H1 = H_Kitaev_Chain(N, T[1], tmax)
    # H1 = H_BdG(N, T[1], tmax)
    E1_raw, V1_raw = Diag_H(H1)

    # Match and fix phases for V1 relative to V0
    V1_matched, perm01 = match_eigenvectors(V0_raw, V1_raw)
    E1 = E1_raw[perm01]
    V1_fixed = fix_phases(V0_raw, V1_matched)
    V0 = V0_raw
    V1 = V1_fixed
    E0 = E0_raw

    E_all[0] = E0
    E_all[1] = E1

    # Running integral of (lambda_l - lambda_k)
    phase_int = np.zeros((dim, dim), dtype=complex)

    # --- Step n = 0: forward difference for A0, then evolve a_1 ---
    # Use gauge-fixed V0, V1 here
    A0 = psi_dotpsi_forward(V0, V1, dt)
    lambda0 = lambda_from_E_and_A(E0, A0)

    a_t[1], phase_int = evolution_step_RK4(a_t[0], phase_int, A0, lambda0, dt)

    # Prepare rolling eigenpairs
    V_prev, V_curr = V0, V1
    E_curr = E1    # Consistent with V_curr

    # --- Main time loop: central differences, 1 <= n <= Nt-2 --- 
    for n in range(1, Nt - 1):
        t_next = T[n + 1]

        # Build and diagonalize H at t_{n+1}
        H_next = H_Kitaev_Chain(N, t_next, tmax)
        # H_next = H_BdG(N, t_next, tmax)
        E_next_raw, V_next_raw = Diag_H(H_next)

        # Match eigenvectors and fix phases relative to V_curr
        V_next_matched, perm = match_eigenvectors(V_curr, V_next_raw)
        E_next = E_next_raw[perm]
        V_next_fixed = fix_phases(V_curr, V_next_matched)
        V_next = V_next_fixed

        E_all[n + 1] = E_next   # store permuted eigenvalues

        # psi_dotpsi via central difference using gauge-fixed vectors
        A_n = central_A(V_prev, V_curr, V_next, dt)

        # lambda at t_n (consistent with E_curr and A_n)
        lambda_n = lambda_from_E_and_A(E_curr, A_n)

        # Evolve a_n -> a_{n+1}
        a_t[n + 1], phase_int = evolution_step_RK4(a_t[n], phase_int, A_n, lambda_n, dt)

        # Roll eigenpairs
        V_prev, V_curr = V_curr, V_next
        E_curr = E_next

        # Reconstruct I_k(t) from phase_int, using reference index 0
        I_k = -phase_int[0, :]                      # array length dim
        phase_factors = np.exp(-1j * I_k)
        a_phys = phase_factors * a_t[n+1]           # a_k * e^{-i I_k}

        # Overlaps between original (t=0) and current instantaneous basis
        B = V0.conj().T @ V_curr                    # shape (dim, dim)

        # Amplitudes in original eigenbasis
        b_t[n+1] = B @ a_phys

        if n % 5 == 0:
            print(f"Iteration {t_}. Step {n} / {Nt-1}")

    return T, a_t, E_all, b_t

def main():
    t0 = 0
    tmax_ = [0.1]

    for t_, tmax in enumerate(tmax_):
        dt = 5e-3
        Nt = int(tmax/dt) + 1 
        T = np.linspace(t0, tmax, Nt)

        N = 10         # number of sites
        dim = 2**N      #Hilbert-space dimension for Kitaev
        # dim = 2*N      # Hilbert-space dimension for BdG

        print("Running evolution...")
        T, a_t, E_all, b_t = run_evolution(t_, T, tmax, N, dim)

        np.savetxt(f'at_T{tmax}_Nt{Nt}_N{N}.txt', a_t)

        # Plot
        tau = T / tmax

        norm = np.sum(np.abs(a_t)**2, axis=1)
        P0 = np.abs(a_t[:, 0])**2 

        plt.subplot(1,2,2)
        for i in range(len(a_t[0,:]) - 1):
            Pi = np.abs(a_t[:, i])**2 #/ norm
            plt.plot(tau, Pi, label = fr"$|c_{i}(t)|^2$", alpha = (len(a_t[0,:]) - i) / (len(a_t[0,:])))
        # plt.plot(tau, P0, label= r"$a_0 t_\text{max} ="  f"{tmax}$")
        plt.plot(tau, norm, label=f"norm {tmax}", alpha=0.25)
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"P = $|a|^2$")
        plt.title("Adiabatic")
        # plt.legend()

        plt.subplot(1,2,1)
        P0_orig = np.abs(b_t[:, 0])**2     # original ground-state population
        plt.plot(tau, P0_orig, label="orig ground")
        plt.title("Original")

        plt.legend()
    plt.savefig(f"Kitaev_T={tmax}_Nt={Nt}_N={N}_dt={dt}.pdf")
    plt.show()

main()
