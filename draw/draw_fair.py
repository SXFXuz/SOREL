import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

def get_loss_ACS(loss_name):
    optimizer_list = ['sgd','lsvrg_batch','prospect_batch','primaldual_batch'] 
    dir = './result/fair/NY/'
    optimizer_dict = {}
    for optimizer_name in optimizer_list:
        file_name = loss_name+'_'+ optimizer_name + '.npz'
        loaded_data = np.load(dir+file_name)
        train_loss = loaded_data['array1']
        train_time = loaded_data['array2']
        test_acc = loaded_data['array3']
        train_eo = loaded_data['array4']
        test_eo = loaded_data['array5']
        train_dp = loaded_data['array6']
        test_dp = loaded_data['array7']
        train_subopt = loaded_data['array8']
        optimizer_dict[optimizer_name] = [train_loss,train_time,train_subopt,test_acc,train_eo,test_eo,train_dp,test_dp]

    # subopt v.s. Time
    sgd_loss = optimizer_dict["sgd"][2]
    lsvrg_loss = optimizer_dict["lsvrg_batch"][2]
    prospect_loss = optimizer_dict["prospect_batch"][2]
    primaldual_loss = optimizer_dict["primaldual_batch"][2]

    x_sgd = optimizer_dict["sgd"][1]
    x_lsvrg = optimizer_dict["lsvrg_batch"][1]
    x_prospect = optimizer_dict["prospect_batch"][1]
    x_primaldual = optimizer_dict["primaldual_batch"][1]
    return sgd_loss, x_sgd, lsvrg_loss, x_lsvrg, prospect_loss, x_prospect, primaldual_loss, x_primaldual


def get_loss_Law(loss_name):
    dir = './result/fair/School/'
    optimizer_dict = {}
    optimizer_dict_var = {}
    optimizer_list = ['sgd','lsvrg_batch','prospect_batch','primaldual_batch'] 
    for optimizer_name in optimizer_list:
        file_name = loss_name+'_'+ optimizer_name + '.npz'
        loaded_data = np.load(dir+file_name)
        train_loss = loaded_data['array1']
        train_time = loaded_data['array2']
        train_mse = loaded_data['array3']
        test_mse = loaded_data['array4']
        train_smd = loaded_data['array5']
        test_smd = loaded_data['array6']
        train_subopt = loaded_data['array7']
        optimizer_dict[optimizer_name] = [train_loss,train_time,train_subopt,train_mse,test_mse,train_smd,test_smd]

    # subopt v.s. Time
    sgd_loss = optimizer_dict["sgd"][2]
    lsvrg_loss = optimizer_dict["lsvrg_batch"][2]
    prospect_loss = optimizer_dict["prospect_batch"][2]
    primaldual_loss = optimizer_dict["primaldual_batch"][2]

    x_sgd = optimizer_dict["sgd"][1]
    x_lsvrg = optimizer_dict["lsvrg_batch"][1]
    x_prospect = optimizer_dict["prospect_batch"][1]
    x_primaldual = optimizer_dict["primaldual_batch"][1]
    return sgd_loss, x_sgd, lsvrg_loss, x_lsvrg, prospect_loss, x_prospect, primaldual_loss, x_primaldual

loss_dir = {'CVaR':'superquantile','ESRM':'esrm','Extremile':'extremile'}
x_lim_dir = {'CVaR':{'energy':1,'concrete':1,'kin8nm':5,'power':10}, 'ESRM':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}, 'Extremile':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}} 
n_rows = 2
n_cols = 3
titles = ['CVaR', 'ESRM', 'Extremile']
fig, axes = plt.subplots(n_rows, n_cols, figsize=(9, 6))

y_labels = ['ACS Employment', 'Law School']
# Plot the data
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]

        loss_name = loss_dir[titles[j]]
        dataset = y_labels[i]

        if dataset == 'ACS Employment':
            sgd_loss, x_sgd, lsvrg_loss, x_lsvrg, prospect_loss, x_prospect, primaldual_loss, x_primaldual = get_loss_ACS(loss_name)
        else:
            sgd_loss, x_sgd, lsvrg_loss, x_lsvrg, prospect_loss, x_prospect, primaldual_loss, x_primaldual = get_loss_Law(loss_name)

        # 绘制数据
        ax.plot(x_sgd, sgd_loss[:len(x_sgd)], label='SGD', linewidth=3, color='blue')  # 可以根据需要调整 marker
        ax.plot(x_lsvrg, lsvrg_loss[:len(x_lsvrg)], label='LSVRG', linewidth=3, color='red')
        ax.plot(x_prospect, prospect_loss[:len(x_prospect)], label='Prospect', linewidth=3, color='lightblue')
        ax.plot(x_primaldual, primaldual_loss[:len(x_primaldual)], label='SOREL', linewidth=3, color='orange')

        # 设置 Y 轴的对数刻度
        ax.set_yscale('log')
        ax.set_xscale('log')

        if i == 0:
            ax.set_title(titles[j], fontsize=12)
        if j == 0:
            ax.set_ylabel(y_labels[i], fontsize=12)
        if j == 1:
            ax.set_xlabel('Time (s)', fontsize=12)

# Adjust layout
fig.tight_layout()
fig.subplots_adjust(bottom=0.18) 

# Add a legend
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), prop={'size': 12})

plt.savefig('./figure/fair_subopt.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()


