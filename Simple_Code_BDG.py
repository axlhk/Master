import numpy as np
import matplotlib.pyplot as plt


"Functions and constants used in making the Hamiltonian"

omega0 = 1
mu0 = 0.5
delta0 = 2

def omega_(t, tmax):
    w = 2 * omega0 * np.abs(t / tmax - 1/2)     # Eivind
    # return omega0
    return w

def mu_(t, tmax):
    mu_start = -3
    mu_stop = 3
    # return (mu_start + (mu_stop - mu_start )* t / tmax)
    return mu0

def delta_(t, tmax):
    d = 2 * delta0 * np.abs(t / tmax - 1/2)     # Eiving
    # return delta0
    return d

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

    return H_bdg

def Diag_H(H):
    """Diagonalize Hermitian H: returns eigenvalues E and eigenvectors V."""
    E, V = np.linalg.eigh(H)
    return E, V

def initial_eigenpairs(T, tmax, N, dim):    #Initial eigenpairs for t0 and t1
    H0 = H_BdG(N, T[0], tmax)
    E0, V0 = Diag_H(H0)

    H1 = H_BdG(N, T[1], tmax)
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

    a_t[1], phase_int = evolution_step_RK4(a_t[0], phase_int, A0, lambda0, dt)

    # Prepare rolling eigenpairs
    V_prev, V_curr = V0, V1
    E_curr = E1

    # --- Main time loop: central differences, 1 <= n <= Nt-2 --- 
    for n in range(1, Nt - 1):
        t_next = T[n + 1]

        # Build and diagonalize H at t_{n+1}
        H_next = H_BdG(N, t_next, tmax)
        E_next, V_next = Diag_H(H_next)
        E_all[n + 1] = E_next

        # psi_dotpsi via central difference
        A_n = central_A(V_prev, V_curr, V_next, dt)

        # lambda at t_n
        lambda_n = lambda_from_E_and_A(E_curr, A_n)

        # Evolve a_n -> a_{n+1}
        a_t[n + 1], phase_int = evolution_step_RK4(a_t[n], phase_int, A_n, lambda_n, dt)

        # Roll eigenpairs
        V_prev, V_curr = V_curr, V_next
        E_curr = E_next

        if n % 200 == 0:
            print(f"Iteration {t_}. Step {n} / {Nt-1}")

    return T, a_t, E_all
 
def main():
    t0 = 0
    tmax_ = [0.1, 20]

    for t_, tmax in enumerate(tmax_):
        dt = 1e-3
        Nt = int(tmax/dt) + 1 
        T = np.linspace(t0, tmax, Nt)

        N = 10         # number of sites
        dim = 2*N      # Hilbert-space dimension

        print("Running evolution...")
        T, a_t, E_all = run_evolution(t_, T, tmax, N, dim)

        np.savetxt(f'at_T{tmax}_Nt{Nt}_N{N}.txt', a_t)

        # Plot
        tau = T / tmax

        norm = np.sum(np.abs(a_t)**2, axis=1)
        P0 = np.abs(a_t[:, 0])**2 
        P1 = np.abs(a_t[:, 1])**2
        P2 = np.abs(a_t[:, 2])**2

        plt.plot(tau, P0, label= r"$t_\text{max} ="  f"{tmax}$")
        # plt.plot(tau, P1, label=r"$|a_1(t)|^2$", alpha=0.5)
        # plt.plot(tau, P2, label=r"$|a_2(t)|^2$", alpha=0.5)
        # plt.plot(tau, norm, label="norm", alpha=0.25)
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"P = $|a_0|^2$")
        plt.legend()
    plt.savefig(f"at_T{tmax}_Nt{Nt}_N{N}.pdf")
    plt.show()

main()
