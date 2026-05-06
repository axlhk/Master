import numpy as np
import matplotlib.pyplot as plt


"Functions and constants used in making the Hamiltonian"

omega0 = 1
mu0 = 0.5
delta0 = 2

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

def H_Kitaev_Chain(N, dim): #Returns the original Kitaev Hamiltonian in the occupational basis
    "H = sum - omega ( a_j^dagger a_j+1 + a_j+1^dagger + a_j ) "    
    " - mu (n_j - 1/2)" #n_j = a_j^dagger a_j. Number operator
    "+ delta a_j a_j+1 + delta^* a_j+1^dagger a_j"

    omega = omega0
    mu = mu0
    delta = delta0

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

def evolution_c(H, dt, Nt, dim): 
    "Evolves the schrodinger equation using euler integration"

    "Defining the initial state"
    # psi = np.zeros(dim, dtype = complex)
    # psi[0] = np.sqrt(0.05)

    psi = np.random.randn(dim) + 1j * np.random.randn(dim)  #Randomly initiated
    psi = psi / np.linalg.norm(psi)                       #Normalized

    "Defining the time dependent array"
    psi_t = np.zeros((Nt, dim), dtype = complex)   
    psi_t[0] = psi   #Filling initial state

    for nt in range(Nt - 1):
        psi = psi - 1j * dt * (H @ psi)
        psi_t[nt + 1] = psi
    return psi_t

def Diag_H(H):    #Returns the diagonal and the eigenvectors of the Kitaev chain matrix

    E, V = np.linalg.eigh(H)    #.eigh eigen hermitian

    return E, V

def to_eigenbasis(V, psi_t):
    
    Nt, dim = psi_t.shape
    c_t = np.zeros_like(psi_t, dtype=complex)

    # V0^\dagger = V0.conj().T
    Vdag = V.conj().T

    for n in range(Nt):
        c_t[n] = Vdag @ psi_t[n]

    return c_t

def main():
    t0 = 0
    tmax = 5
    Nt = 1000
    T = np.linspace(t0, tmax, Nt)
    dt = T[1] - T[0]

    N = 6           #Number of sites
    dim = 2**N      #For original Kitaev chain

    H = np.zeros((dim, dim), dtype = complex)
    E = np.zeros((dim))
    V = np.zeros((dim, dim), dtype=complex)
    
    "1) Make the Hamiltonian and diagonalise it for all time steps"
    print(f"Starting H")

    H = H_Kitaev_Chain(N, dim)   #Build the Hamiltonian
    E, V = Diag_H(H)                       #Get eigenvalues and eigenvectors
    
    print(f"Hamiltonian made and diagonalised")
    
    "2) Evolves and finds the psi(t) using numerical integration"

    print(f"Starting psi(t)")

    psi_t = evolution_c(H, dt, Nt, dim)
    
    print(f"Found psi(t)")
    
    "Finds the c_k(t) by projecting psi onto eigenbasis"

    c_t = to_eigenbasis(V, psi_t)

    "Plot the results for c(t)"

    norm = np.sum(np.abs(c_t)**2, axis=1)

    for i in range(len(c_t[0,:]) - 1):
        Pi = np.abs(c_t[:, i])**2 #/ norm
        plt.plot(T, Pi, label = fr"$|c_{i}(t)|^2$", alpha = (len(c_t[0,:]) - i) / (len(c_t[0,:])))

    # P0 = np.abs(c_t[:, 0])**2 
    # P1 = np.abs(c_t[:, 1])**2 
    # P2 = np.abs(c_t[:, 2])**2 

    # plt.plot(T, P0, label = r"$|c_0(t)|^2$")
    # plt.plot(T, P1, label = r"$|c_1(t)|^2$", alpha = 0.5)
    # plt.plot(T, P2, label = r"$|c_2(t)|^2$", alpha = 0.5)

    # plt.plot(T, norm, label = "norm", alpha = 0.25)
    plt.xlabel("T")
    plt.ylabel("P")
    # plt.legend()
    plt.show()

main()