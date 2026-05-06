import numpy as np
from Kitaev_AdiabatixAprx_Sparse import (
    Kitaev,
    omega_Eivind_4_2,
    mu_Eivind_4_2,
    delta_Eivind_4_2,
)

def compute_P_orig_for_T(
    tmax,
    psi_file,
    out_file,
    N=10,
    omega0=1.0,
    delta0=2.0,
    mu0=0.5,
    dt_tau=1e-3,
):
    """
    Given:
      - tmax (e.g. 20.0),
      - psi_file: saved psi_t for that tmax,
    compute P_orig[n, j] = |<E_j(0)|psi(t_n)>|^2
    and save to out_file as a .npy array.
    """

    # --- load your existing evolution data ---
    psi_t = np.load(psi_file)          # shape (Nt, dim)
    Nt, dim = psi_t.shape

    # time step in physical time (consistent with your run)
    T_run = np.linspace(0.0, tmax, Nt)
    dt_phys = T_run[1] - T_run[0]

    # --- build model and diagonalize H(0) to get eigenbasis |E_j(0)> ---
    model = Kitaev(
        dt=dt_phys,
        N=N,
        stride=0,
        tmax=tmax,
        Hamiltonian="Kitaev",
        Integration="RK4",
        Evolution="Schrodinger",
        omega_fun=omega_Eivind_4_2,
        mu_fun=mu_Eivind_4_2,
        delta_fun=delta_Eivind_4_2,
        exp_degen=1,
        return_full=False,
        trunc_dim=None,
    )

    H0 = model.build_H(0.0)
    E0, V0 = model.Diag_H(H0)   # V0[:, j] = |E_j(0)>

    # --- project psi_t onto that eigenbasis ---
    # c_t[n, j] = <E_j(0) | psi(t_n)>  (same as psi_t @ V0.conj())
    c_t = psi_t @ V0.conj()
    P_orig = np.abs(c_t)**2

    # --- save ---
    np.save(out_file, P_orig)
    print(f"Saved P_orig with shape {P_orig.shape} to {out_file}")

if __name__ == "__main__":
    # Example for T = 20, matching your existing filenames
    tmax = 20.0
    psi_file = r"C:\Users\axlkl\Fag\Master\_psi_t_20_1.0_2.0_0.5_Schrodinger_precomputed_Kitaev_RK4_10_0.001_dense.npy"
    out_file = r"C:\Users\axlkl\Fag\Master\_P_orig_20_1.0_2.0_0.5_Schrodinger_precomputed_Kitaev_RK4_10_0.001_dense.npy"

    compute_P_orig_for_T(tmax, psi_file, out_file)
