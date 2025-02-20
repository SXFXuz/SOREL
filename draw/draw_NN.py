import os
import time
import matplotlib.pyplot as plt
import numpy as np

n_passes = 600
optimizer_list = ['sgdNN','lsvrgNN','prospectNN','sorelNN'] 

def get_loss(dataset, loss_name):
    dir = './result/NN/'+dataset +'/'
    optimizer_dict = {}
    optimizer_dict_full = {}
    optimizer_dict_std = {}
    for optimizer_name in optimizer_list:
        file_name = loss_name+'_'+ optimizer_name + '.npz'
        loaded_data = np.load(dir+file_name)
        train_loss = loaded_data['array1']
        train_subgrad = loaded_data['array2']
        train_time = loaded_data['array3']
        optimizer_dict[optimizer_name] = [train_loss,train_subgrad,train_time]

        file_name2 = loss_name+'_'+ optimizer_name + '_variance.npz'
        loaded_data2 = np.load(dir+file_name2)
        train_loss_full = loaded_data2['array1']
        train_subgrad_full = loaded_data2['array2']
        train_time_full = loaded_data2['array3']
        optimizer_dict_full[optimizer_name] = np.array([train_loss_full,train_subgrad_full,train_time_full])
        optimizer_dict_std[optimizer_name] = np.array([np.std(train_loss_full, axis=0), np.std(train_subgrad_full, axis=0), np.std(train_time_full, axis=0)])
    return optimizer_dict['sgdNN'][0], optimizer_dict['lsvrgNN'][0], optimizer_dict['prospectNN'][0], optimizer_dict['sorelNN'][0]


loss_dir = {'CVaR':'superquantile','ESRM':'esrm','Extremile':'extremile'}
# x_lim_dir = {'CVaR':{'energy':1,'concrete':1,'kin8nm':5,'power':10}, 'ESRM':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}, 'Extremile':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}} 
x_lim_dir = {'superquantile':{'energy':(0.035,0.18),'concrete':(0.083,0.145)}, 'esrm':{'energy':(0.025, 0.2),'concrete':(0.076,0.15)},'extremile':{'energy':(0.035,0.18),'concrete':(0.085,0.14)}}
n_rows = 2
n_cols = 3
titles = ['CVaR', 'ESRM', 'Extremile']
fig, axes = plt.subplots(n_rows, n_cols, figsize=(13.5, 7.5))

y_labels = ['energy', 'concrete']
# Plot the data
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]

        loss_name = loss_dir[titles[j]]
        dataset = y_labels[i]

        sgd_loss, lsvrg_loss, prospect_loss, primaldual_loss = get_loss(dataset, loss_name)
        x_sgd = np.arange(0,n_passes+1)
        x_lsvrg = np.arange(0, n_passes+1, 2)
        x_prospect = np.arange(0,n_passes+1)
        x_primaldual = np.arange(0, n_passes+1, 2)

        downsample = 10
        ax.plot(x_sgd[::10], sgd_loss[:len(x_sgd)][::10], label='SGD', linewidth=3, color='blue')
        ax.plot(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], label='LSVRG', linewidth=3, color='red')
        ax.plot(x_prospect[::5], prospect_loss[:len(x_prospect)][::5], label='Prospect', linewidth=3, color='lightblue')
        ax.plot(x_primaldual, primaldual_loss[:len(x_primaldual)], label='SOREL', linewidth=3, color='orange')

        x_lim_d, x_lim_u = x_lim_dir[loss_name][dataset]

        ax.set_ylim([x_lim_d, x_lim_u])
        if i == 0:
            ax.set_title(titles[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(y_labels[i], fontsize=12)
        if j == 1:
            ax.set_xlabel('Passes', fontsize=12)

fig.tight_layout()
fig.subplots_adjust(bottom=0.14) 

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), prop={'size': 12})

plt.savefig('./figure/NN_subopt.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()