import numpy as np
import matplotlib.pyplot as plt

def cutting_comparison():
    # ---------- EIVINDS DATA (NPZ) ----------
    his_file = r"C:\Users\axlkl\Fag\Master\cut_time_evolution\N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=5_t=20_onandoff=True.npz"
    data = np.load(his_file)

    # Inspect keys once to see what's what
    # print(data.files)

    # From his save call:
    # np.savez(...,
    #   transfer_components_original,        -> arr_0
    #   transfer_components_instantaneous,   -> arr_1
    #   self.eigvals_original,               -> arr_2
    #   instantaneous_eigvals,               -> arr_3
    #   self.parity_array_eigvecs,           -> arr_4
    #   cut_site,                            -> arr_5 (scalar)
    #   n_timesteps,                         -> arr_6 (scalar)
    #   n_eigvals,                           -> arr_7 (scalar)
    #   t,                                   -> arr_8 (scalar, should be 20)
    #   on_and_off                           -> arr_9
    # )

    his_psi   = data['arr_0']  # transfer_components_original
    his_adiab = data['arr_1']  # transfer_components_instantaneous

    # print("his_psi shape:",   his_psi.shape)
    # print("his_adiab shape:", his_adiab.shape)

    n_steps = his_psi.shape[0]  # should be 1000
    tau = np.linspace(0.0, 1.0, n_steps)

    # ---------- YOUR DATA (NPY) ----------
    # Use the exact filenames for T=20

    my_psi_file   = r"C:\Users\axlkl\Fag\Master\_P_orig_20_1.0_2.0_0.5_Schrodinger_precomputed_Kitaev_RK4_10_0.001_dense.npy"    # fill in
    my_adiab_file = r"C:\Users\axlkl\Fag\Master\_a_t_inst_20_1.0_2.0_0.5_Schrodinger_precomputed_Kitaev_RK4_10_0.001_dense.npy"   # fill in

    my_psi   = np.load(my_psi_file)
    my_adiab = np.load(my_adiab_file)

    my_adiab = np.abs(my_adiab**2)


    #If data needs transposing to match
    if his_psi.shape[0] != my_psi.shape[0] and his_psi.shape[1] == my_psi.shape[0]:
        his_psi   = his_psi.T
        his_adiab = his_adiab.T

    # assert his_psi.shape == my_psi.shape
    # assert his_adiab.shape == my_adiab.shape
    diff_psi   = his_psi   - my_psi   # psi coefficients difference
    diff_adiab = his_adiab - my_adiab # adiabatic basis coefficients difference

    components = 1
    plt.figure(figsize=(8, 5))
    for i in range(min(components, diff_psi.shape[1])):  # first up to 5 components
        plt.plot(tau, diff_psi[:, i], label=f"component {i}")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\Delta \psi$ coefficient (his - mine)")
    plt.title(r"Difference in original basis coefficients at $T_{\max} = 20$")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"__eivind_diff_psi_{components}.pdf")
    plt.show()

    plt.figure(figsize=(8, 5))
    for i in range(min(components, diff_adiab.shape[1])):  # first up to 5 components
        plt.plot(tau, diff_adiab[:, i], label=f"component {i}")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\Delta$ adiabatic coefficient (his - mine)")
    plt.title(r"Difference in adiabatic basis coefficients at $T_{\max} = 20$")
    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"__eivind_diff_instantaneous{components}.pdf")
    plt.show()

    components_ = 1
    for i in range(min(components_, diff_adiab.shape[1])):
        plt.plot(tau, my_psi[:, i], label=f"mine")
        plt.plot(tau, his_psi[:, i], label=f"his")
    plt.xlabel(r"$\tau$")
    plt.ylabel(r"Ground state in original basis for T = 20")
    plt.title(r"")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"__eivind_original_basis.pdf")
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

    for tmax_ in T_list:
        T = np.load(f"T_run_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy")
        a_t = np.load(f"a_t_inst_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy")
        psi_t = np.load(f"psi_t_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy")
        # c_t = np.load(f"_c_t_orig_{tmax_}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.npy") #Havent run class after adding c_t. It can easily be found in cutting_comparison() 

        tau = T / tmax_

        # instantaneous gs-subspace probability
        P0_inst = np.sum(np.abs(a_t[:,:1])**2, axis=1)

        # original-basis probability in psi0 (ground state at t=0)
        P0_orig = np.zeros_like(tau)        
        psi0 = psi_t[0]
        for n in range(len(tau)):
            P0_orig[n] = np.abs(np.vdot(psi0, psi_t[n]))**2     #OBS: psi_t is actually in Fock basis. Use c_t_orig for original basis (Eivind). (They coincide for the ground state so it's fine)


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
    # plt.savefig(f"_schrodinger_cut_{len(T_list)}_{omega0}_{delta0}_{mu0}_{evolution}_{Hamiltonian_type}_{Integrator}_{N}_{dt_tau}_{HamRep}.pdf")
    plt.show()


if __name__ == "__main__":
    # cutting_comparison()
    plot_schrodinger_cut()