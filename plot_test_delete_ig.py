import numpy as np
import matplotlib.pyplot as plt

def cutting_comparison():
    adiab_schrod_file = r"C:\Users\axlkl\Fag\Master\1_a_t_inst_20_1.0_2.0_0.5_Schrodinger_precomputed_Kitaev_RK4_6_0.001_dense.npy"
    adiab_adiab_file = r"C:\Users\axlkl\Fag\Master\10_a_t_reduced_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_6_0.001001001001001001_Kitaev.npy"
    adiab_adiab_file2 = r"C:\Users\axlkl\Fag\Master\chain_cut_chain_a_t_reduced_trunc=10_20_1.0_2.0_0.5_Adiabatic_precomputed_Kitaev_RK4_N=6_dt=0.001001001001001001_Kitaev.npy"

    adiab_schrod = np.load(adiab_schrod_file)
    adiab_adiab = np.load(adiab_adiab_file)
    adiab_adiab2 = np.load(adiab_adiab_file)

    n_steps = adiab_schrod.shape[0]  
    tau = np.linspace(0.0, 1.0, n_steps)

    adiab_schrod = np.abs(adiab_schrod**2)
    adiab_adiab = np.abs(adiab_adiab**2)
    adiab_adiab2 = np.abs(adiab_adiab**2)

    components = 1
    plt.figure(figsize=(8, 5))
    for i in range(min(components, adiab_schrod.shape[1])):  # first up to 5 components
        plt.plot(tau, adiab_schrod[:, i], label=f"adiab_schrod")

    for i in range(min(components, adiab_adiab.shape[1])):  # first up to 5 components
        plt.plot(tau, adiab_adiab[:, i], label=f"adiab_adiab")

    for i in range(min(components, adiab_adiab.shape[1])):  # first up to 5 components
        plt.plot(tau, adiab_adiab[:, i], label=f"adiab_adiab2")

    plt.xlabel(r"$\tau$")
    plt.ylabel(r"$\Delta \psi$ coefficient (his - mine)")
    plt.title(r"Difference in original basis coefficients at $T_{\max} = 20$")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cutting_comparison()
    # plot_schrodinger_cut()