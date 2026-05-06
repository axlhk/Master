import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib as mpl
import matplotlib.font_manager as font_manager

sim_folder = "local_time_evolution/"

save_path = 'E:/Repositories/Masteroppgave/Masteroppgave/Scripts/Kitaev chain simulator/Figures/' + sim_folder
data_path = 'E:/Repositories/Masteroppgave/Masteroppgave/Scripts/Kitaev chain simulator/Data/' + sim_folder

# Cut time evolution files
"""
files = [data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=20_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=10_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=5_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=1_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=0.5_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=0.1_onandoff=True.npz", \
         data_path + "N=10_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=1000_neigvals=1024_cutsite=4_t=0.01_onandoff=True.npz"]
"""

# Local time evolution files
files = [data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=20000_neigvals=256_localsite=[3 4]_maxstrength=10_t=20_onandoff=True.npz", \
         data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=20000_neigvals=256_localsite=[3 4]_maxstrength=10_t=10_onandoff=True.npz", \
         data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=20000_neigvals=256_localsite=[3 4]_maxstrength=10_t=5_onandoff=True.npz", \
         data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=20000_neigvals=256_localsite=[3 4]_maxstrength=10_t=1_onandoff=True.npz", \
         data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=20000_neigvals=256_localsite=[3 4]_maxstrength=10_t=0.1_onandoff=True.npz", \
         data_path + "N=8_mu0=0.5_delta0=2.0_t0=1.0_ntimesteps=2000_neigvals=256_localsite=[3 4]_maxstrength=10_t=0.01_onandoff=True.npz"]

title = True
save = False

type = "local"

n_points_plot = 100

cutoff_orig = 1e-8
cutoff_inst = 1e-8

# Matplotlib parameters for latex
if save:
    plt.rcParams["figure.figsize"] = [12/1.7,5/1.6]
    plt.rcParams["font.size"] = 11
    plt.rcParams["figure.autolayout"] = True
    plt.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname = mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    plt.rcParams['font.serif'] = cmfont.get_name()
    plt.rcParams['mathtext.fontset'] = 'cm'
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['axes.formatter.use_mathtext'] = True

n_files = len(files)

# Load data
transfer_components_original_list = []
transfer_components_instantaneous_list = []
eigvals_original_list = []
instantaneous_eigvals_list = []
parity_array_eigvecs_list = []
cut_site_list = []
max_strength_list = []
n_timesteps_list = []
n_eigvals_list = []
t_list = []
on_and_off_list = []
if type == "cut":
    for i in range(n_files):
        data = np.load(files[i])
        transfer_components_original_list.append(data['arr_0'])
        transfer_components_instantaneous_list.append(data['arr_1'])
        eigvals_original_list.append(data['arr_2'])
        instantaneous_eigvals_list.append(data['arr_3'])
        parity_array_eigvecs_list.append(data['arr_4'])
        cut_site_list.append(data['arr_5'])
        n_timesteps_list.append(data['arr_6'])
        n_eigvals_list.append(data['arr_7'])
        t_list.append(data['arr_8'])
        on_and_off_list.append(data['arr_9'])
elif type == "local":
    for i in range(n_files):
        data = np.load(files[i])
        transfer_components_original_list.append(data['arr_0'])
        transfer_components_instantaneous_list.append(data['arr_1'])
        eigvals_original_list.append(data['arr_2'])
        instantaneous_eigvals_list.append(data['arr_3'])
        parity_array_eigvecs_list.append(data['arr_4'])
        cut_site_list.append(data['arr_5'])
        max_strength_list.append(data['arr_6'])
        n_timesteps_list.append(data['arr_7'])
        n_eigvals_list.append(data['arr_8'])
        t_list.append(data['arr_9'])
        on_and_off_list.append(data['arr_10'])



# Generate partial name for saving
save_name = save_path + "ground_state_probabilities"
for i in range(n_files):
    save_name += "_t={}".format(t_list[i])

if np.all(np.array(n_eigvals_list)==n_eigvals_list[0]):
    n_eigvals = n_eigvals_list[0]
    original_ground_state_probabilities = np.zeros((n_points_plot, n_files))
    instantaneous_ground_state_probabilities = np.zeros((n_points_plot, n_files))
else:
    sys.exit("System sizes are mismatched. Check files loaded.")

for i in range(n_files):
    original_ground_state_probabilities[:,i] = transfer_components_original_list[i][::n_timesteps_list[i]//n_points_plot,0]
    instantaneous_ground_state_probabilities[:,i] = transfer_components_instantaneous_list[i][::n_timesteps_list[i]//n_points_plot,0]

times = np.linspace(0,1,n_points_plot)

norm = mpl.colors.Normalize(vmin=np.min(np.log(np.array(t_list))), vmax=np.max(np.log(np.array(t_list))))
cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.viridis)
cmap.set_array([])

"""
f1, ax1 = plt.subplots()
f2, ax2 = plt.subplots()

for i in range(n_files):
    ax1.plot(times, original_ground_state_probabilities[:,i], color=cmap.to_rgba(np.log(t_list[i])), label=r"$T={}$".format(t_list[i]))
    ax2.plot(times, instantaneous_ground_state_probabilities[:,i], color=cmap.to_rgba(np.log(t_list[i])), label=r"$T={}$".format(t_list[i]))

ax1.set_xlabel(r"$\tau/T$")
ax2.set_xlabel(r"$\tau/T$")

ax1.set_ylabel(r"$|c_1|^2$")
ax2.set_ylabel(r"$|d_1|^2$")

box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])

ax1.legend(loc='center left', bbox_to_anchor=(1, 0.5), handlelength=0.5)
ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), handlelength=0.5)

if title:
    ax1.set_title("Original")
    ax2.set_title("Instantaneous")

if save:
    f1.savefig(save_name + "_original.pdf", bbox_inches='tight', pad_inches=0)
    f2.savefig(save_name + "_instantaneous.pdf", bbox_inches='tight', pad_inches=0)
else:
    plt.show()
"""

f1, (ax1,ax2) = plt.subplots(1,2)

for i in range(n_files):
    ax1.plot(times, original_ground_state_probabilities[:,i], color=cmap.to_rgba(np.log(t_list[i])), label=r"$T={}$".format(t_list[i]))
    ax2.plot(times, instantaneous_ground_state_probabilities[:,i], color=cmap.to_rgba(np.log(t_list[i])), label=r"$T={}$".format(t_list[i]))

ax1.set_xlabel(r"$\tau/T$")
ax2.set_xlabel(r"$\tau/T$")

ax1.set_ylabel(r"$|c_1|^2$")
ax2.set_ylabel(r"$|d_1|^2$")

box1 = ax1.get_position()
ax1.set_position([box1.x0, box1.y0, box1.width * 0.95, box1.height])

box2 = ax2.get_position()
ax2.set_position([box2.x0, box2.y0, box2.width * 0.95, box2.height])

ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), handlelength=0.5)

if title:
    ax1.set_title("Original")
    ax2.set_title("Instantaneous")

if save:
    f1.savefig(save_name + ".pdf", bbox_inches='tight', pad_inches=0)
else:
    plt.show()
