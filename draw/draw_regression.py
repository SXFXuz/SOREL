import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dir_name = './result/regression/'
dir_name_baseline = './result/regression/'
save_dir_0 = './figure/'

svg = True

#------------------------------
# single sample
titles = ['CVaR', 'CVaR', 'ESRM', 'ESRM', 'Extremile', 'Extremile']
# Labels for the y-axis
y_labels = ['yacht', 'energy', 'concrete', 'kin8nm', 'power']
loss_dir = {'CVaR':'superquantile','ESRM':'esrm_hard','Extremile':'extremile_hard'}
n_rows = 5
n_cols = 6

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12.5))


x_lim_dir = {'CVaR':{'yacht':4,'energy':25,'concrete':20}, 'ESRM':{'yacht':3,'energy':15,'concrete':8}, 'Extremile':{'yacht':3,'energy':10,'concrete':10}} 
# Plot the data
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]
        loss_name = loss_dir[titles[j]]
        dataset = y_labels[i]

        optimizer_name = "sgd"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        sgd_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        sgd_time = df.values

        optimizer_name = "lsvrg"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        lsvrg_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        lsvrg_time = df.values

        optimizer_name = "prospect"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        prospect_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        prospect_time = df.values

        optimizer_name = "primaldual"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name+'suboptimality/'+file_name, engine='openpyxl')
        primaldual_loss = df.values
        df = pd.read_excel(dir_name+'time/'+file_name, engine='openpyxl')
        primaldual_time = df.values

        if j % 2 ==0:
            if i < 3:
                x_lim = 101
            else:
                x_lim = 51
            # passes
            x1 = np.arange(0, x_lim)
            x2 = np.arange(0, x_lim, 2)
            x3 = np.arange(0, x_lim)
            x4 = np.arange(0, x_lim, 2)
            
        else:
            x1 = sgd_time
            x2 = lsvrg_time
            x3 = prospect_time
            x4 = primaldual_time
        y1 = sgd_loss[:len(x1)]
        y2 = lsvrg_loss[:len(x2)]
        y3 = prospect_loss[:len(x3)]
        y4 = primaldual_loss[:len(x4)]

        if j % 2 == 1:
            if i<3:
                x1 = x1[::100]
                y1 = y1[::100]
            elif i == 3:
                x1 = x1[::10]
                y1 = y1[::10]
            else:
                x1 = x1[::5]
                y1 = y1[::5]


        ax.plot(x1, y1, label='SGD', linewidth=3, color='blue')
        ax.plot(x2, y2, label='LSVRG', linewidth=3, color='red')
        ax.plot(x3, y3, label='Prospect', linewidth=3, color='lightblue')
        ax.plot(x4, y4, label='SOREL', linewidth=3, color='orange')
        ax.set_yscale('log')
        if i >= 3 and j%2 == 1:
            ax.set_xscale('log')

        if i < 3 and j % 2 == 1:
            x_lim = x_lim_dir[titles[j]][y_labels[i]]
            ax.set_xlim([0, x_lim])
        if i == 0:
            ax.set_title(titles[j], fontsize=14)
        if j == 0:
            ax.set_ylabel(y_labels[i], fontsize=14)
        if i == n_rows - 1 and j%2==0:
            ax.set_xlabel('Passes', fontsize=14)
        if i== n_rows - 1 and j%2 == 1:
            ax.set_xlabel('Time (s)', fontsize=14)

fig.tight_layout()
fig.subplots_adjust(bottom=0.11) 

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), prop={'size': 15})

# Save the plot
# plt.savefig('plot_with_final.png')
plt.savefig('./figure/regression.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()

# -----------------------
#batch sample
loss_dir = {'CVaR':'superquantile','ESRM':'esrm_hard','Extremile':'extremile_hard'}
x_lim_dir = {'CVaR':{'energy':1,'concrete':1,'kin8nm':5,'power':10}, 'ESRM':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}, 'Extremile':{'energy':1,'concrete':1,'kin8nm':2.5,'power':2.5}} 
n_rows = 4
n_cols = 6
titles = ['CVaR', 'CVaR', 'ESRM', 'ESRM', 'Extremile', 'Extremile']
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

y_labels = ['energy', 'concrete', 'kin8nm', 'power']
# Plot the data
for i in range(n_rows):
    for j in range(n_cols):
        ax = axes[i, j]

        loss_name = loss_dir[titles[j]]
        dataset = y_labels[i]

        # Get the data
        optimizer_name = "sgd"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        sgd_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        sgd_time = df.values

        optimizer_name = "lsvrg_batch"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        lsvrg_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        lsvrg_time = df.values

        optimizer_name = "prospect_batch"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name_baseline+'suboptimality/'+file_name, engine='openpyxl')
        prospect_loss = df.values
        df = pd.read_excel(dir_name_baseline+'time/'+file_name, engine='openpyxl')
        prospect_time = df.values

        optimizer_name = "primaldual_batch"
        file_name = 'regression_'+dataset+'_'+loss_name+'_'+optimizer_name+"_"+'best_result.xlsx'
        df = pd.read_excel(dir_name+'suboptimality/'+file_name, engine='openpyxl')
        primaldual_loss = df.values
        df = pd.read_excel(dir_name+'time/'+file_name, engine='openpyxl')
        primaldual_time = df.values

        if j % 2 == 0:
            if i < 2:
                x_lim = 401
            else:
                x_lim = 101
            # passes
            x1 = np.arange(0, x_lim)
            x2 = np.arange(0, x_lim, 2)
            x3 = np.arange(0, x_lim)
            x4 = np.arange(0, x_lim, 2)
            
        else:
            x1 = sgd_time
            x2 = lsvrg_time
            x3 = prospect_time
            x4 = primaldual_time
        y1 = sgd_loss[:len(x1)]
        y2 = lsvrg_loss[:len(x2)]
        y3 = prospect_loss[:len(x3)]
        y4 = primaldual_loss[:len(x4)]

        if i == 0:
            if j % 2 ==0:
                x1 = x1[::20]
                y1 = y1[::20]
                x2 = x2[::10]
                y2 = y2[::10]
                x3 = x3[::10]
                y3 = y3[::10]
            else:
                x1 = x1[::50]
                y1 = y1[::50]
                x2 = x2[::30]
                y2 = y2[::30]
                x3 = x3[::50]
                y3 = y3[::50]
        elif i == 1:
            x1 = x1[::20]
            y1 = y1[::20]
            x2 = x2[::10]
            y2 = y2[::10]
            x3 = x3[::10]
            y3 = y3[::10]
        elif i == 2:
            x1 = x1[::5]
            y1 = y1[::5]
            x2 = x2[::5]
            y2 = y2[::5]
        else:
            x1 = x1[::5]
            y1 = y1[::5]


        ax.plot(x1, y1, label='SGD', linewidth=3, color='blue')
        ax.plot(x2, y2, label='LSVRG', linewidth=3, color='red')
        ax.plot(x3, y3, label='Prospect', linewidth=3, color='lightblue')
        ax.plot(x4, y4, label='SOREL', linewidth=3, color='orange')
        ax.set_yscale('log')

        if j % 2 == 1:
            x_lim = x_lim_dir[titles[j]][y_labels[i]]
            ax.set_xlim([0, x_lim])
        ax.set_xlim([0, x_lim])
        if i == 0:
            ax.set_title(titles[j], fontsize=14)
        if j == 0:
            ax.set_ylabel(y_labels[i], fontsize=14)
        if i == n_rows - 1 and j%2==0:
            ax.set_xlabel('Passes', fontsize=14)
        if i== n_rows - 1 and j%2 == 1:
            ax.set_xlabel('Time (s)', fontsize=14)

fig.tight_layout()
fig.subplots_adjust(bottom=0.11) 

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0), prop={'size': 15})

plt.savefig('./figure/regression_batch.pdf', format='pdf',bbox_inches='tight', pad_inches=0)
plt.show()
