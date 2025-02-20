import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import time


optimizer_list = ['sgd','lsvrg_batch','prospect_batch','primaldual_batch'] 

loss_name = 'superquantile_lite'
dir = './result/dro/'
optimizer_dict_cvar = {}
optimizer_dict_cvar_var = {}
optimizer_subopt_std_cvar = {}
for optimizer_name in optimizer_list:
    file_name = loss_name+'_'+ optimizer_name + '.npz'
    loaded_data = np.load(dir+file_name)
    train_loss = loaded_data['array1']
    train_time = loaded_data['array2']
    train_subopt = loaded_data['array3']
    test_acc = loaded_data['array4']
    test_dro = loaded_data['array5']
    optimizer_dict_cvar[optimizer_name] = [train_loss,train_time,train_subopt,test_acc, test_dro]

    file_name2 = loss_name+'_'+ optimizer_name + '_variance.npz'
    loaded_data2 = np.load(dir+file_name2)
    train_loss_var = loaded_data2['array1']
    train_subopt_var = loaded_data2['array2']
    test_acc_var = loaded_data2['array3']
    test_dro_var = loaded_data2['array4']
    optimizer_dict_cvar_var[optimizer_name] = [train_loss_var,train_subopt_var,test_acc_var,test_dro_var]

if loss_name == 'superquantile_lite':
    opt_loss = 1.369862774988842
    loss_0 = 1.6094379124340992
    for optimizer_name in optimizer_list:
        optimizer_dict_cvar[optimizer_name][2] = (optimizer_dict_cvar[optimizer_name][0]-opt_loss)/(optimizer_dict_cvar[optimizer_name][0][0]-opt_loss)
        optimizer_dict_cvar_var[optimizer_name][1] = (optimizer_dict_cvar_var[optimizer_name][0]-opt_loss)/(optimizer_dict_cvar[optimizer_name][0][0]-opt_loss)

for optimizer_name in optimizer_list:
    train_subopt_var = optimizer_dict_cvar_var[optimizer_name][1]
    train_subopt_std = np.std(train_subopt_var, ddof=1, axis=0)
    optimizer_subopt_std_cvar[optimizer_name] = train_subopt_std


loss_name = 'extremile'
dir = './result/dro/'
optimizer_dict_extremile = {}
optimizer_dict_extremile_var = {}
optimizer_subopt_std_extremile = {}
for optimizer_name in optimizer_list:
    file_name = loss_name+'_'+ optimizer_name + '.npz'
    loaded_data = np.load(dir+file_name)
    train_loss = loaded_data['array1']
    train_time = loaded_data['array2']
    train_subopt = loaded_data['array3']
    test_acc = loaded_data['array4']
    test_dro = loaded_data['array5']
    optimizer_dict_extremile[optimizer_name] = [train_loss,train_time,train_subopt,test_acc, test_dro]

    file_name2 = loss_name+'_'+ optimizer_name + '_variance.npz'
    loaded_data2 = np.load(dir+file_name2)
    train_loss_var = loaded_data2['array1']
    train_subopt_var = loaded_data2['array2']
    test_acc_var = loaded_data2['array3']
    test_dro_var = loaded_data2['array4']
    optimizer_dict_extremile_var[optimizer_name] = [train_loss_var,train_subopt_var,test_acc_var,test_dro_var]
    train_subopt_std = np.std(train_subopt_var, ddof=1, axis=0)
    optimizer_subopt_std_extremile[optimizer_name] = train_subopt_std



# ----------
fig, axs = plt.subplots(1, 4, figsize=(14, 4))

# --------------
# First plot: CVaR Suboptimality
sgd_loss = optimizer_dict_cvar["sgd"][2]
lsvrg_loss = optimizer_dict_cvar["lsvrg_batch"][2]
prospect_loss = optimizer_dict_cvar["prospect_batch"][2]
primaldual_loss = optimizer_dict_cvar["primaldual_batch"][2]
point = 500

x_sgd = optimizer_dict_cvar["sgd"][1]
x_lsvrg = optimizer_dict_cvar["lsvrg_batch"][1]
x_prospect = optimizer_dict_cvar["prospect_batch"][1]
x_primaldual = optimizer_dict_cvar["primaldual_batch"][1]

sgd_std = optimizer_subopt_std_cvar["sgd"]
lsvrg_std = optimizer_subopt_std_cvar["lsvrg_batch"]
prospect_std = optimizer_subopt_std_cvar["prospect_batch"]
primaldual_std = optimizer_subopt_std_cvar["primaldual_batch"]

axs[0].plot(x_sgd[::50], sgd_loss[:len(x_sgd)][::50], label='SGD', linewidth=3, color='blue',zorder=1)
axs[0].plot(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], label='LSVRG', linewidth=3, color='red',zorder=1)
axs[0].plot(x_prospect[::50], prospect_loss[:len(x_prospect)][::50], label='Prospect', linewidth=3, color='lightblue',zorder=1)
axs[0].plot(x_primaldual, primaldual_loss[:len(x_primaldual)], label='SOREL', linewidth=3, color='orange',zorder=1)

axs[0].fill_between(x_sgd[::50], sgd_loss[:len(x_sgd)][::50], sgd_loss[:len(x_sgd)][::50] + sgd_std[:len(x_sgd)][::50], color="blue", alpha=0.2)
axs[0].fill_between(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], lsvrg_loss[:len(x_lsvrg)] + lsvrg_std[:len(x_lsvrg)], color="red", alpha=0.2)
axs[0].fill_between(x_prospect[::50], prospect_loss[:len(x_prospect)][::50], prospect_loss[:len(x_prospect)][::50] + prospect_std[:len(x_prospect)][::50], color="lightblue", alpha=0.5)
axs[0].fill_between(x_primaldual, primaldual_loss[:len(x_primaldual)], primaldual_loss[:len(x_primaldual)]+ primaldual_std[:len(x_primaldual)], color="orange", alpha=0.4)

axs[0].scatter(x_sgd[point], sgd_loss[point], color='blue',s=100, marker='^',zorder=2)
axs[0].scatter(x_lsvrg[point//2], lsvrg_loss[point//2], color='red',s=80,marker='s',zorder=2)
axs[0].scatter(x_prospect[point], prospect_loss[point], color='lightblue',s=100,marker='*',zorder=2)
axs[0].scatter(x_primaldual[point//2], primaldual_loss[point//2], color='orange',s=100,marker='o',zorder=2)

axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_xlabel('Time (s)',fontsize=12)
axs[0].set_ylabel('Suboptimality',fontsize=12)
axs[0].set_title("CVaR")

# -------------------
# Second plot: CVaR Median Group Error
methods = ['SGD', 'LSVRG', 'Prospect', 'SOREL']

sgd_dro = optimizer_dict_cvar["sgd"][4][:501]
lsvrg_dro = optimizer_dict_cvar["lsvrg_batch"][4][:251]
prospect_dro = optimizer_dict_cvar["prospect_batch"][4][:501]
primaldual_dro = optimizer_dict_cvar["primaldual_batch"][4][:251]

error_sgd = np.mean(sgd_dro[-10:])
error_lsvrg = np.mean(lsvrg_dro[-10:])
error_prospect = np.mean(prospect_dro[-10:])
error_primaldual = np.mean(primaldual_dro[-10:])

errors = [error_sgd, error_lsvrg, error_prospect, error_primaldual]

axs[1].bar(methods, errors, color=['blue', 'red', 'lightblue', 'orange'],alpha=0.8, width=0.5)
axs[1].set_ylabel('Worst Group Error',fontsize=12)
axs[1].set_ylim(0.8, 0.812)
axs[1].yaxis.set_major_locator(MaxNLocator(3)) 
axs[1].set_title('CVaR')

# ------------------
sgd_std = optimizer_subopt_std_extremile["sgd"]
lsvrg_std = optimizer_subopt_std_extremile["lsvrg_batch"]
prospect_std = optimizer_subopt_std_extremile["prospect_batch"]
primaldual_std = optimizer_subopt_std_extremile["primaldual_batch"]

sgd_loss = optimizer_dict_extremile["sgd"][2]
lsvrg_loss = optimizer_dict_extremile["lsvrg_batch"][2]
prospect_loss = optimizer_dict_extremile["prospect_batch"][2]
primaldual_loss = optimizer_dict_extremile["primaldual_batch"][2]
point = 500

x_sgd = optimizer_dict_extremile["sgd"][1]
x_lsvrg = optimizer_dict_extremile["lsvrg_batch"][1]
x_prospect = optimizer_dict_extremile["prospect_batch"][1]
x_primaldual = optimizer_dict_extremile["primaldual_batch"][1]

axs[2].plot(x_sgd[::50], sgd_loss[:len(x_sgd)][::50], label='SGD', linewidth=3, color='blue',zorder=1)  # 可以根据需要调整 marker
axs[2].plot(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], label='LSVRG', linewidth=3, color='red',zorder=1)
axs[2].plot(x_prospect[::50], prospect_loss[:len(x_prospect)][::50], label='Prospect', linewidth=3, color='lightblue',zorder=1)
axs[2].plot(x_primaldual, primaldual_loss[:len(x_primaldual)], label='SOREL', linewidth=3, color='orange',zorder=1)

axs[2].fill_between(x_sgd[::50], sgd_loss[:len(x_sgd)][::50], sgd_loss[:len(x_sgd)][::50] + sgd_std[:len(x_sgd)][::50], color="blue", alpha=0.2)
axs[2].fill_between(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], lsvrg_loss[:len(x_lsvrg)] + lsvrg_std[:len(x_lsvrg)], color="red", alpha=0.2)
axs[2].fill_between(x_prospect[::50], prospect_loss[:len(x_prospect)][::50], prospect_loss[:len(x_prospect)][::50] + prospect_std[:len(x_prospect)][::50], color="lightblue", alpha=0.5)
axs[2].fill_between(x_primaldual, primaldual_loss[:len(x_primaldual)], primaldual_loss[:len(x_primaldual)]+ primaldual_std[:len(x_primaldual)], color="orange", alpha=0.4)


axs[2].scatter(x_sgd[point], sgd_loss[point], color='blue',s=100, marker='^',zorder=2)
axs[2].scatter(x_lsvrg[point//2], lsvrg_loss[point//2], color='red',s=80,marker='s',zorder=2)
axs[2].scatter(x_prospect[point], prospect_loss[point], color='lightblue',s=100,marker='*',zorder=2)
axs[2].scatter(x_primaldual[point//2], primaldual_loss[point//2], color='orange',s=100,marker='o',zorder=2) 
axs[2].set_yscale('log')
axs[2].set_xscale('log')

axs[2].set_xlabel('Time (s)',fontsize=12)
axs[2].set_ylabel('Suboptimality',fontsize=12)
# axs[2].yaxis.set_major_locator(MaxNLocator(8)) 
axs[2].set_title("Extremile")

# -----------
# Fourth plot: Extremile Median Group Error
sgd_dro = optimizer_dict_extremile["sgd"][4][:501]
lsvrg_dro = optimizer_dict_extremile["lsvrg_batch"][4][:251]
prospect_dro = optimizer_dict_extremile["prospect_batch"][4][:501]
primaldual_dro = optimizer_dict_extremile["primaldual_batch"][4][:251]

error_sgd = np.mean(sgd_dro[-10:])
error_lsvrg = np.mean(lsvrg_dro[-10:])
error_prospect = np.mean(prospect_dro[-10:])
error_primaldual = np.mean(primaldual_dro[-10:])

errors = [error_sgd, error_lsvrg, error_prospect, error_primaldual]

axs[3].bar(methods, errors, color=['blue', 'red', 'lightblue', 'orange'], alpha=0.8,width=0.5)
axs[3].set_ylabel('Worst Group Error',fontsize=12)
axs[3].set_ylim(0.75, 0.770)
axs[3].yaxis.set_major_locator(MaxNLocator(5)) 
axs[3].set_title('Extremile')

# Adjust layout
plt.tight_layout()
fig.subplots_adjust(bottom=0.25) 
handles, labels = axs[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), prop={'size': 12})
plt.savefig('./figure/dro.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()
