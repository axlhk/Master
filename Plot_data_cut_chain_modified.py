import numpy as np
import matplotlib.pyplot as plt


def cutting_comparison():
    # ---------- EIVINDS DATA (NPZ) ----------
    his_file = r"C:\Users\axlkl\Fag\Master\cut_time_evolution\N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=5_t=20_onandoff=True.npz"
    data = np.load(his_file)


    his_psi   = data['arr_0']  # transfer_components_original, shape ~ (1001, 1024)
    his_adiab = data['arr_1']  # transfer_components_instantaneous, shape ~ (1001, 1024)

    # ---------- YOUR DATA (NPY) ----------
    # Use the exact filenames for T=20
    my_psi_file   = r"C:\Users\axlkl\Fag\Master\chain_cut_chain_psi_t_trunc=10_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_N=10_dt=0.001001001001001001_Kitaev.npy"
    my_adiab_file = r"C:\Users\axlkl\Fag\Master\eivind_cut_chain_cut_chain_a_t_reduced_trunc=10_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_N=10_dt=0.001001001001001001_Kitaev.npy"

    # my_psi_file   = r"C:\Users\axlkl\Fag\Master\chain_cut_chain_psi_t_trunc=50_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_N=10_dt=0.001001001001001001_Kitaev.npy"
    # my_adiab_file = r"C:\Users\axlkl\Fag\Master\chain_cut_chain_a_t_reduced_trunc=50_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_N=10_dt=0.001001001001001001_Kitaev.npy"

    # my_adiab_file = r"C:\Users\axlkl\Fag\Master\1_a_t_reduced_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_10_0.001001001001001001_Kitaev.npy"
    
    my_psi   = np.load(my_psi_file)      # e.g. (1000, 150)
    my_adiab = np.load(my_adiab_file)    # now observed as (1001, 1)
    my_adiab = np.abs(my_adiab**2)

    # ---------- Possible transpose for his data ----------
    # If his data is stored as (n_eigvals, n_steps) we transpose to (n_steps, n_eigvals)
    # This keeps your original heuristic:
    if his_psi.shape[0] != my_psi.shape[0] and his_psi.shape[1] == my_psi.shape[0]:
        his_psi   = his_psi.T
        his_adiab = his_adiab.T

    # ---------- Align timesteps ----------
    # Use the minimum number of timesteps across all four arrays.
    # This effectively "drops the last value" from any array that has 1001 steps.
    n_steps = min(his_psi.shape[0], his_adiab.shape[0],
                  my_psi.shape[0],  my_adiab.shape[0])

    his_psi   = his_psi[:n_steps, :]
    his_adiab = his_adiab[:n_steps, :]
    my_psi    = my_psi[:n_steps, :]
    my_adiab  = my_adiab[:n_steps, :]

    tau = np.linspace(0.0, 1.0, n_steps)

    # ---------- Align number of eigenstates/components ----------
    # For psi: his has 1024, you have 150 -> truncate his to 150.
    # For adiabatic: his has 1024, you have 1 -> everything gets truncated to 1 (GS only).
    n_states_psi   = min(his_psi.shape[1],   my_psi.shape[1])
    n_states_adiab = min(his_adiab.shape[1], my_adiab.shape[1])

    his_psi   = his_psi[:,   :n_states_psi]
    my_psi    = my_psi[:,    :n_states_psi]
    his_adiab = his_adiab[:, :n_states_adiab]
    my_adiab  = my_adiab[:,  :n_states_adiab]

    # Final sanity checks
    assert his_psi.shape   == my_psi.shape
    assert his_adiab.shape == my_adiab.shape

    # ---------- Differences ----------
    diff_psi   = his_psi   - my_psi   # psi coefficients difference
    diff_adiab = his_adiab - my_adiab # adiabatic basis coefficients difference

    # We mostly care about the ground state, so set components = 1
    components = 1

    # Difference in original basis coefficients (psi)
    plt.figure(figsize=(8, 5))
    for i in range(min(components, diff_psi.shape[1])):
        plt.plot(tau, diff_psi[:, i], label=f"component {i}")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\Delta \psi$ coefficient (his - mine)")
    plt.title(r"Difference in original basis coefficients at $T_{\max} = 20$")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("eivind_cut___eivind_diff_psi_1.pdf")
    plt.show()

    # Difference in adiabatic basis coefficients
    plt.figure(figsize=(8, 5))
    for i in range(min(components, diff_adiab.shape[1])):
        plt.plot(tau, diff_adiab[:, i], label=f"component {i}")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Difference in adiabatic coefficient (his - mine)")
    plt.title(r"Difference in adiabatic basis coefficients at $T_{\max} = 20$")
    # plt.legend()
    plt.tight_layout()
    plt.savefig("eivind_cut__trunc=10__eivind_diff_instantaneous_1.pdf")
    plt.show()

    # ---------- Direct comparison of ground state component ----------
    components_ = 1
    plt.figure(figsize=(8, 5))
    for i in range(min(components_, my_psi.shape[1])):
        plt.plot(tau, my_psi[:, i], label="mine")
        plt.plot(tau, his_psi[:, i], label="his")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Ground state in original basis for $T = 20$")
    plt.title(r"Ground-state coefficient comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig("eivind_cut__trunc=10__eivind_original_basis_gs.pdf")
    plt.show()

    for i in range(min(components_, diff_adiab.shape[1])):
        plt.plot(tau, my_adiab[:, i], label=f"mine")
        plt.plot(tau, his_adiab[:, i], label=f"his")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Ground state in adiabatic basis for T = 20")
    plt.title(r"")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"eivind_cut___eivind_instantaneous_basis.pdf")
    plt.show()


def plot_schrodinger_cut():
    T_list = [0.01, 0.5, 1, 5, 10, 20]

    omega0 = 1.0
    delta0 = 2.0
    mu0 = 0.5
    evolution = "Schrodinger_precomputed"
    Hamiltonian_type = "Kitaev"
    Integrator = "RK4"
    N = 10
    dt_tau = 1e-3
    HamRep = "dense"

    plt.figure(figsize=(10, 4))

    for tmax_ in T_list:
        T = np.load(
            f"T_run_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_"
            f"{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy"
        )
        a_t = np.load(
            f"a_t_inst_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_"
            f"{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy"
        )
        psi_t = np.load(
            f"psi_t_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_"
            f"{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy"
        )
        # c_t = np.load(f"_c_t_orig_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy") #Havent run class after adding c_t. It can easily be found in cutting_comparison()

        tau = T / tmax_

        # instantaneous gs-subspace probability (GS assumed index 0)
        P0_inst = np.sum(np.abs(a_t[:, :1])**2, axis=1)

        # original-basis probability in psi0 (ground state at t=0)
        P0_orig = np.zeros_like(tau)
        psi0 = psi_t[0]
        for n in range(len(tau)):
            P0_orig[n] = np.abs(np.vdot(psi0, psi_t[n]))**2
            # OBS: psi_t is actually in Fock basis. Use c_t_orig for original basis (Eivind).
            # They coincide for the ground state so it's fine.

        plt.subplot(1, 2, 2)
        plt.plot(tau, P0_inst, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = \sum_{k<\mathrm{exp\_degen}} |a_k|^2$")
        plt.title("Instantan basis (Schrödinger)")
        plt.legend()

        plt.subplot(1, 2, 1)
        plt.plot(tau, P0_orig, label=f"T = {tmax_}")
        plt.xlabel(r"$\tau = \frac{T}{t_\text{max}}$")
        plt.ylabel(r"$P = |\langle \psi_0 | \psi(t)\rangle|^2$")
        plt.title("Original basis (Schrödinger)")
        plt.legend()

    plt.tight_layout()
    # plt.savefig(f"_schrodinger_cut_{len(T_list)}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.pdf")
    plt.show()


if __name__ == "__main__":
    cutting_comparison()
    # plot_schrodinger_cut()
