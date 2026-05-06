import numpy as np
import matplotlib.pyplot as plt


"Functions and constants used in making the Hamiltonian"

omega0 = 1
mu0 = 0.5
delta0 = 2

def omega_(t, tmax):
    w = 2 * omega0 * np.abs(t / tmax - 1/2)     # Eivind
    return w

def mu_(t, tmax):
    mu_start = -3
    mu_stop = 3
    # return (mu_start + (mu_stop - mu_start )* t / tmax)
    return mu0

def delta_(t, tmax):
    d = 2 * delta0 * np.abs(t / tmax - 1/2)     # Eiving
    return d

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

def H_Kitaev_Chain(N, t, tmax, dim):
    """
    Returns the Kitaev Hamiltonian in the occupational basis for a given time t.
    H = sum - omega ( a_j^dagger a_{j+1} + a_{j+1}^dagger a_j )
        - mu (n_j - 1/2)
        + delta a_j a_{j+1} + delta^* a_{j+1}^dagger a_j^dagger
    """
    omega = omega_(t, tmax)
    mu = mu_(t, tmax)
    delta = delta_(t, tmax)

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
                    H[m2, n] += -omega * c1 * c2

            # a_{j+1}^dagger a_j
            c3, m3 = a_on_basis_state(n, j)
            if m3 is not None:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 is not None:
                    H[m4, n] += -omega * c3 * c4

            # Delta terms
            # a_j a_{j+1}
            c5, m5 = a_on_basis_state(n, j + 1)
            if m5 is not None:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 is not None:
                    H[m6, n] += delta * c5 * c6

            # a_{j+1}^dagger a_j^dagger
            c7, m7 = a_dag_on_basis_state(n, j)
            if m7 is not None:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 is not None:
                    H[m8, n] += (delta.conjugate()) * c7 * c8

    return H

"Will switch to BdG simplification later. Massive spedup"
def H_BdG(N, t, tmax):
    
    omega = omega_(t, tmax)
    mu    = mu_(t, tmax)
    delta = delta_(t, tmax)

    # Normal part h(t)
    h = np.zeros((N, N), dtype=complex)
    for j in range(N):
        h[j, j] = -mu
        if j < N - 1:
            h[j, j+1] = -omega
            h[j+1, j] = -omega

    # Pairing part Δ(t)
    Delta = np.zeros((N, N), dtype=complex)
    for j in range(N - 1):
        Delta[j, j+1]   =  delta
        Delta[j+1, j]   = -delta  # p-wave sign structure

    # BdG matrix
    upper = np.hstack((h,      Delta))
    lower = np.hstack((Delta.conj().T, -h.T))
    H_bdg = np.vstack((upper, lower))

    return H_BdG

def Diag_H(H):
    """Diagonalize Hermitian H: returns eigenvalues E and eigenvectors V."""
    E, V = np.linalg.eigh(H)
    return E, V

def initial_eigenpairs(T, tmax, N, dim):    #Initial eigenpairs for t0 and t1
    H0 = H_Kitaev_Chain(N, T[0], tmax, dim)
    E0, V0 = Diag_H(H0)

    H1 = H_Kitaev_Chain(N, T[1], tmax, dim)
    E1, V1 = Diag_H(H1)

    return E0, V0, E1, V1

def psi_dotpsi_forward(V0, V1, dt): #Initial psi_dot_psi for t0 and t1 by forward difference 
    overlap_01 = V0.conj().T @ V1
    A0 = (overlap_01 - np.eye(V0.shape[1], dtype=complex)) / dt
    return A0

def lambda_from_E_and_A(E, A): #Returns lamba as analytically defined (A = psi_dotpsi))
    return E - 1j * np.diag(A)

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

def evolution_step_euler_cromer():
    a_next = 1
    phase_int_next = 1
    return a_next, phase_int_next

def evolution_step_RK4():
    a_next = 1
    phase_int_next = 1
    return a_next, phase_int_next

def central_A(V_prev, V_curr, V_next, dt):  #Normal psi_dot_psi
    overlap_n_next = V_curr.conj().T @ V_next
    overlap_n_prev = V_curr.conj().T @ V_prev
    A_n = (overlap_n_next - overlap_n_prev) / (2 * dt)
    return A_n

def run_evolution(T, tmax, N, dim):
    Nt = len(T)
    dt = T[1] - T[0]

    # Allocate outputs
    a_t = np.zeros((Nt, dim), dtype=complex)
    a_t[0, 0] = 1.0  # initial ground-state amplitude

    E_all = np.zeros((Nt, dim))

    # Initial eigenpairs at t0 and t1
    E0, V0, E1, V1 = initial_eigenpairs(T, tmax, N, dim)
    E_all[0] = E0
    E_all[1] = E1

    # Running integral of (lambda_l - lambda_k)
    phase_int = np.zeros((dim, dim), dtype=complex)

    # --- Step n = 0: forward difference for A0, then evolve a_1 ---
    A0 = psi_dotpsi_forward(V0, V1, dt)
    lambda0 = lambda_from_E_and_A(E0, A0)

    a_t[1], phase_int = evolution_step_euler(a_t[0], phase_int, A0, lambda0, dt)

    # Prepare rolling eigenpairs
    V_prev, V_curr = V0, V1
    E_curr = E1

    # --- Main time loop: central differences, 1 <= n <= Nt-2 --- 
    for n in range(1, Nt - 1):
        t_next = T[n + 1]

        # Build and diagonalize H at t_{n+1}
        H_next = H_Kitaev_Chain(N, t_next, tmax, dim)
        E_next, V_next = Diag_H(H_next)
        E_all[n + 1] = E_next

        # psi_dotpsi via central difference
        A_n = central_A(V_prev, V_curr, V_next, dt)

        # lambda at t_n
        lambda_n = lambda_from_E_and_A(E_curr, A_n)

        # Evolve a_n -> a_{n+1}
        a_t[n + 1], phase_int = evolution_step_euler(a_t[n], phase_int, A_n, lambda_n, dt)

        # Roll eigenpairs
        V_prev, V_curr = V_curr, V_next
        E_curr = E_next

        if n % 50 == 0:
            print(f"Step {n} / {Nt-1}")

    return T, a_t, E_all

def main():
    t0 = 0
    tmax = 1
    Nt = 1000

    N = 10          # number of sites
    dim = 2**N      # Hilbert-space dimension

    T = np.linspace(t0, tmax, Nt)

    print("Running evolution...")
    T, a_t, E_all = run_evolution(T, tmax, N, dim)

    np.savetxt(f'at_T{tmax}_Nt{Nt}_N{N}.txt', a_t)

    # Plot
    norm = np.sum(np.abs(a_t)**2, axis=1)
    P0 = np.abs(a_t[:, 0])**2
    P1 = np.abs(a_t[:, 1])**2
    P2 = np.abs(a_t[:, 2])**2

    plt.plot(T, P0, label=r"$|a_0(t)|^2$")
    plt.plot(T, P1, label=r"$|a_1(t)|^2$", alpha=0.5)
    plt.plot(T, P2, label=r"$|a_2(t)|^2$", alpha=0.5)
    plt.plot(T, norm, label="norm", alpha=0.25)
    plt.xlabel("T")
    plt.ylabel("P")
    plt.legend()
    plt.savefig(f"at_T{tmax}_Nt{Nt}_N{N}.pdf")
    plt.show()

main()
