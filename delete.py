import numpy as np
import matplotlib.pyplot as plt

def simple_plot_cut_time_evolution(path, file,
                                   cutoff_orig=1e-7,
                                   title=True):
    """
    Minimal plot of cut time evolution results from the npz file
    produced by KitaevChain.time_evolution(disorder_type='cut', save=True).

    path: directory where the file is located (e.g. "cut_time_evolution/").
    file: filename, e.g.
          "cut_time_evolution/N=10_mu0=0.0_delta0=1.0_t0=1.0_ntimesteps=100_neigvals=1024_cutsite=4_t=1.0_onandoff=False.npz"
    cutoff_orig: only plot eigenstates that reach probability >= cutoff_orig at some time.
    """

    data = np.load(r"C:\Users\axlkl\Fag\Master\cut_time_evolution\N=10_mu0=0.0_delta0=1.0_t0=1.0_ntimesteps=100_neigvals=1024_cutsite=4_t=0.01_onandoff=False.npz")

    # Unpack in same order as in time_evolution() / load_and_plot_cut_time_evolution()
    transfer_components_original = data['arr_0']   # shape: (n_timesteps, n_eigvals)
    transfer_components_instantaneous = data['arr_1']  # unused here, but present
    eigvals_original = data['arr_2']              # shape: (n_eigvals,)
    instantaneous_eigvals = data['arr_3']         # unused here
    parity_array_eigvecs = data['arr_4']          # unused here
    cut_site = data['arr_5']
    n_timesteps = int(data['arr_6'])
    n_eigvals = int(data['arr_7'])
    t_total = float(data['arr_8'])
    on_and_off = bool(data['arr_9'])

    # Time grid
    times = np.linspace(0.0, t_total, n_timesteps, endpoint=False)

    # Decide which eigenstates to plot (same logic as original loader, but simpler)
    index_list = []
    for i in range(n_eigvals):
        if np.any(transfer_components_original[:, i] >= cutoff_orig):
            index_list.append(i)

    # Simple plot
    plt.figure(figsize=(6, 4))
    for idx in index_list:
        plt.plot(times,
                 transfer_components_original[:, idx],
                 label=f"i={idx}, E≈{eigvals_original[idx]:.3f}")

    plt.xlabel("Time")
    plt.ylabel("Probability")
    if title:
        plt.title(f"Cut at bond {int(cut_site)+1}, T={t_total}, on/off={on_and_off}")

    # If there are many states, you may want to comment this out
    if len(index_list) <= 10:
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # The same name construction as in run_and_plot_cut_time_evolution_N10_dtau:
    N = 10
    mu0 = 0.0
    delta0 = 1.0
    t0 = 1.0
    dtau = 1e-2
    n_timesteps = int(1.0 / dtau)
    T = 1.0
    on_and_off = False
    n_eigvals = 2**N
    cut_site = N//2 - 1

    name = (
        f"cut_time_evolution/N={N}_mu0={mu0}_delta0={delta0}_t0={t0}"
        f"_ntimesteps={n_timesteps}_neigvals={n_eigvals}"
        f"_cutsite={cut_site}_t={T}_onandoff={on_and_off}"
    )
    file = name + ".npz"

    simple_plot_cut_time_evolution(path="", file=file)
