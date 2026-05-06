import numpy as np
import matplotlib.pyplot as plt


"Functions and constants used in making the Hamiltonian"

omega0 = 1
mu0 = 0.5
delta0 = 2

def omega_(t, tmax):
    w = 2 * omega0 * np.abs(t / tmax - 1/2)     #Eivind
    return w
    # return (omega_start + (omega_stop - omega_start )* t / tmax)
    # return omega0 

def mu_(t, tmax):
    mu_start = -3
    mu_stop = 3
    # return (mu_start + (mu_stop - mu_start )* t / tmax)
    return mu0

def delta_(t, tmax):
    d = 2 * delta0 * np.abs(t / tmax - 1/2)     #Eiving
    return d
    # return delta0

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

def H_Kitaev_Chain(N, t, tmax, dim): #Returns the original Kitaev Hamiltonian in the occupational basis
    "H = sum - omega ( a_j^dagger a_j+1 + a_j+1^dagger + a_j ) "    
    " - mu (n_j - 1/2)" #n_j = a_j^dagger a_j. Number operator
    "+ delta a_j a_j+1 + delta^* a_j+1^dagger a_j"

    omega = omega_(t, tmax)
    mu = mu_(t, tmax)
    delta = delta_(t, tmax)

    H = np.zeros((dim, dim), dtype=complex)

    for n in range(dim): 
        ntot = 0
        for j in range(N):     
            nj = np.sum(bit(n,j))
            ntot += nj
        H[n,n] += - mu * (ntot - N/2)        #mu_term (diagonal)
        
        for j in range(N-1):    # omega and delta terms (off-diagonal)

            "Omega"
            #a_j^dagger a_j+1 on state
            c1, m1 = a_on_basis_state(n, j + 1) #c is coefficient, m is bit flip
            if m1 is not None:
                c2, m2 = a_dag_on_basis_state(m1, j)
                if m2 is not None:
                    H[m2, n] += -omega * c1 * c2

            #a_j+1^dagger a_j on state
            c3, m3 = a_on_basis_state(n, j) #c is coefficient, m is bit flip
            if m3 is not None:
                c4, m4 = a_dag_on_basis_state(m3, j + 1)
                if m4 is not None:
                    H[m4, n] += -omega * c3 * c4
            
            "Delta"
            #a_j a_j+1
            c5, m5 = a_on_basis_state(n, j + 1) #c is coefficient, m is bit flip
            if m5 is not None:
                c6, m6 = a_on_basis_state(m5, j)
                if m6 is not None:
                    H[m6, n] += delta * c5 * c6

            #a_j+1^dagger a_j^dagger
            c7, m7 = a_dag_on_basis_state(n, j) #c is coefficient, m is bit flip
            if m7 is not None:
                c8, m8 = a_dag_on_basis_state(m7, j + 1)
                if m8 is not None:
                    H[m8, n] += (delta.conjugate()) * c7 * c8

    # Enforce Hermiticity
    # H = 0.5 * (H + H.conj().T)

    return H

"Will switch to BdG simplification later. Massive spedup"
def H_BdG(N):
    
    H = 1

    return H

def Diag_H(H):    #Returns the diagonal and the eigenvectors of the Kitaev chain matrix

    E, V = np.linalg.eigh(H)    #.eigh eigen hermitian

    return E, V

def psi_dotpsi(dt, V):
    "V is of the form (Nt, dim, dim) dtype complex"

    Nt, dim, _ = V.shape

    A = np.zeros((Nt, dim, dim), dtype=complex)

    for n in range(1, Nt - 1):
        V_prev = V[n-1]
        V_next = V[n+1]
        V_n = V[n]

        for l in range(dim):
            psi_l = V_n[:, l]
            for k in range(dim):
                psi_k_prev = V_prev[:, k]
                psi_k_next = V_next[:, k]
                overlap_prev = np.vdot(psi_l, psi_k_prev)
                overlap_next = np.vdot(psi_l, psi_k_next)
                A[n, l, k] = (overlap_next - overlap_prev) / (2 * dt)

    # forward difference at n = 0
    V0 = V[0]
    V1 = V[1]
    for l in range(dim):
        psi_l0 = V0[:, l]
        for k in range(dim):
            psi_k1 = V1[:, k]
            overlap = np.vdot(psi_l0, psi_k1)
            A[0, l, k] = (overlap - np.vdot(psi_l0, V0[:, k])) / dt

    #backward difference at n = Nt-1
    if Nt > 1:
        VN = V[-1]
        VN_1 = V[-2]
        for l in range(dim):
            psi_l = VN[:, l]
            for k in range(dim):
                psi_k_prev = VN_1[:, k]
                overlap = np.vdot(psi_l, VN[:, k])  # <psi_l(t_N)|psi_k(t_N)> = delta_lk
                A[-1, l, k] = (overlap - np.vdot(psi_l, psi_k_prev)) / dt

    return A

def lamba(E, psi_psidot):   #Returns lamba as analytically defined
    Nt, dim = E.shape
    lamba_ = np.zeros_like(E, dtype=complex)
    for n in range(Nt):
        for k in range(dim):
            lamba_[n, k] = E[n, k] - 1j * psi_psidot[n, k, k]
    
    return lamba_

def phase_int(lamba_, Nt, dt, dim):    #Returns the integral in the exponential in da/dt
    int = np.zeros((Nt, dim, dim), dtype=complex)

    for t in range(1, Nt):  #For each t we find the vectorised lamba_l - lamba_k and then sum up all contributions as an integral
        delta_lambda = lamba_[t-1][:, None] - lamba_[t-1][None, :]
        int[t] = int[t-1] + delta_lambda * dt
    
    return int

def evolution(Nt, dt, dim, lamba_, psi_psidot): #Returns the integral of adot following the equation given earlier. Using euler-cromer

    exp_int = phase_int(lamba_, Nt, dt, dim)

    a = np.zeros((Nt, dim), dtype=complex)
    a[0, 0] = 1 #Ground state

    #Simple euler integration
    for n in range(0, Nt - 1):
        for l in range(dim):
            rhs = 0.0 + 0.0j
            for k in range(dim):
                if k == l:
                    continue
                phase = np.exp(-1j * exp_int[n, l, k])
                rhs += a[n, k] * phase * psi_psidot[n, l, k]
            a[n+1, l] = a[n, l] + dt * rhs
        if n % 200 == 0:
            print(f"Evolution iteration {n}")

    return a

def main():
    t0 = 0
    tmax = 1
    Nt = 1000
    T = np.linspace(t0, tmax, Nt)
    dt = T[1] - T[0]

    N = 10          #Number of sites
    dim = 2**N      #For original Kitaev chain

    E = np.zeros((Nt, dim))
    V = np.zeros((Nt, dim, dim), dtype=complex)
    
    "1) Make the Hamiltonian and diagonalise it for all time steps"
    print(f"1)")

    print(f"Starting H")

    for n, t in enumerate(T):                 #For all time steps
        H = H_Kitaev_Chain(N, t, tmax, dim)   #Build the Hamiltonians
        E_, V_ = Diag_H(H)                    #Get eigenvalues and eigenvectors

        E[n] = E_
        V[n] = V_
        if n % 10 == 0:
            print(f"Hamiltonian iteration {n} / {Nt}")
    
    print(f"Hamiltonian made and diagonalised")

    "2) Get psi_dotpsi using numerical derivation"
    print(f"2)")

    print(f"Finding psi_dot_psi")

    psi_psidot = psi_dotpsi(dt, V)       #Get psi_dotpsi

    print(f"Found")

    "3) Get lambda according to the analytical expression"
    print(f"3)")

    lamba_ = lamba(E, psi_psidot)        #Get lambda

    print(f"Lambda found")

    "4) Evolve the states a using numerical integration"
    print(f"4)")
    print(f"Starting time evolution")

    a_t = evolution(Nt, dt, dim, lamba_, psi_psidot)

    "5) Plot the results for a(t)"

    norm = np.sum(np.abs(a_t)**2, axis = 1)

    P0 = np.abs(a_t[:, 0])**2 #/ norm
    P1 = np.abs(a_t[:, 1])**2 #/ norm
    P2 = np.abs(a_t[:, 2])**2 #/ norm

    plt.plot(T, P0, label = r"$|a_0(t)|^2$")
    plt.plot(T, P1, label = r"$|a_1(t)|^2$", alpha = 0.5)
    plt.plot(T, P2, label = r"$|a_2(t)|^2$", alpha = 0.5)
    plt.plot(T, norm, label = "norm", alpha = 0.25)
    plt.xlabel("T")
    plt.ylabel("P")
    plt.legend()
    plt.show()

main()