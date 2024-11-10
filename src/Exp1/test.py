import sys
import os

sys.path.append('./src/')

import pandas as pd
import torch
from data_util import *
from model.model import *
from model.model_params import * 
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def main():
    print("##model params: {}", model_params)
    batch_size = 1
    data = pd.read_csv('./src/data/test_data.csv')    
    
    data_set = get_test_data_set()
    test_dataloader = DataLoader(data_set, batch_size = batch_size, shuffle = False)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    model = PEP2CCS(model_params['num_layers'], model_params['embedding_size'], model_params['num_heads'], 0, 0).to(device)
    model_path = "./src/checkpoint/model.pt"
    # chkp = torch.load(model_path)

    chkp = torch.load(model_path, map_location=device)

    state_dict = chkp['model_param']
    model.load_state_dict(state_dict)
    
    max_len = 64

    test_size = len(data_set)
    pred_ccs = np.zeros([test_size, 1])
    ccs = np.zeros([test_size, 1])
    charge = np.zeros([test_size, 1])
    seq = np.zeros([test_size, max_len])
    Length = np.zeros([test_size, max_len])
    max_pos = np.zeros([test_size, max_len])
    i = 0
    start = time.time()
    model.eval()
    loss_op = nn.MSELoss()
    test_loss = []
    test_dataloader = tqdm(test_dataloader, total=len(test_dataloader), desc="Model on test set", unit='batch')

    for batch_seq, batch_ccs, batch_charge, batch_length, batch_mz, batch_ccs2 in test_dataloader:
        with torch.no_grad():
            batch_seq = batch_seq.to(device)
            batch_ccs = batch_ccs.to(device)
            batch_ccs2 = batch_ccs2.to(device)
            batch_charge = batch_charge.to(device)
            batch_mz = batch_mz.to(device)
            batch_length = batch_length.to(device)
            
            pred, max_indices = model(batch_seq, batch_charge, batch_length, batch_mz, batch_ccs2)
            
            idx = torch.argmax(max_indices, dim = 1)
            pred, batch_ccs = torch.squeeze(pred), torch.squeeze(batch_ccs)
            pred, batch_ccs = torch.exp(pred), torch.exp(batch_ccs)
            
            loss = loss_op(pred, batch_ccs)
            test_loss.append(loss.item())
            pred_ccs[i:i + batch_size, :] = pred.cpu().numpy().reshape(-1, 1)
            ccs[i:i + batch_size, :] = batch_ccs.cpu().numpy().reshape(-1, 1)
            charge[i:i + batch_size, :] = batch_charge.cpu().numpy().reshape(-1, 1)
            Length[i:i + batch_size, :] = batch_length.cpu().numpy().reshape(-1, 1)
            max_pos[i:i + batch_size, :] = idx.cpu().numpy().reshape(-1, 1)

            seq[i:i + batch_size, :] = batch_seq.cpu().numpy().reshape(-1, max_len)

            i = i + 1

    avr_loss = sum(test_loss) / len(test_loss)
    print("##Finish!")
    print(f'##Time: {(time.time() - start) / 60}')
    print(f'##Test Loss: {avr_loss}')
    
    
    pred_ccs = pred_ccs.ravel()
    data['Pred CCS'] = pred_ccs
    
    df = data
    df['Relative deviation'] = (df['CCS'] / df['Pred CCS']) * 100 - 100
    df['Abusolute error'] = np.abs(df['CCS'] - df['Pred CCS'])
    max_rel = max(df['Relative deviation']) 
    max_abs = max(df['Abusolute error'])
    print(f'max_Rel: {max_rel}, max_Abs: {max_abs}')
    mse = ((df['CCS'] - df['Pred CCS']) ** 2).mean()
    print(f'MSE: {mse}')
    print("Mediant Abusolute Relative Error: ", np.median(np.abs(df['Relative deviation'])))
    df['Error'] = (df['Relative deviation'])
    
    sns.set_style("whitegrid")
    sns.set(rc={"axes.facecolor":"#e6e6e6", "axes.grid":True})
    custom_colors = ['#606060', '#82093B', '#34183E', '#4D779B']


    fig, axs = plt.subplots(1, 3, figsize=(45, 12))
    scatter_size = 40
    alpha_value = 0.75

    sns.scatterplot(data=df, x='m/z', y='Pred CCS', hue='Charge', palette=custom_colors, ax=axs[0], s=scatter_size, alpha=alpha_value)
    for idx, charge in enumerate(df['Charge'].unique()):
        subset = df[df['Charge'] == charge]
        z = np.polyfit(subset['m/z'], subset['Pred CCS'], 1)
        p = np.poly1d(z)
        axs[0].plot(subset['m/z'], p(subset['m/z']), linestyle='-', color=custom_colors[idx % len(custom_colors)], label=f'Trendline Charge {charge}')
    axs[0].set_title('Pred CCS vs m/z with Trendlines by Charge', fontsize=22, fontweight='bold')
    axs[0].legend(prop={'size': 25})
    axs[0].set_xlabel('m/z', fontsize=24, fontweight='bold')
    axs[0].set_ylabel('Pred CCS', fontsize=24, fontweight='bold')
    axs[0].tick_params(axis='both', which='major', labelsize=18)

    sns.scatterplot(data=df, x='m/z', y='CCS', label='True CCS', ax=axs[1], marker='o', color=custom_colors[0], s=scatter_size, alpha=1)
    sns.scatterplot(data=df, x='m/z', y='Pred CCS', label='Pred CCS', ax=axs[1], marker='x', color=custom_colors[1], s=scatter_size, alpha=0.6)
    axs[1].set_title('CCS and Pred CCS vs m/z', fontsize=22, fontweight='bold')
    axs[1].legend(prop={'size': 30}) 
    axs[1].set_xlabel('m/z', fontsize=24, fontweight='bold')
    axs[1].set_ylabel('CCS', fontsize=24, fontweight='bold')
    axs[1].tick_params(axis='both', which='major', labelsize=18) 

    sns.histplot(df['Error'], kde=False, bins=50, ax=axs[2], color=custom_colors[2], alpha=alpha_value)
    axs[2].set_xlabel('deviation (%)', fontsize=24, fontweight='bold')
    axs[2].set_ylabel('Counts', fontsize=24, fontweight='bold') 
    axs[2].set_xlim([-10, 10])
    axs[2].set_title('MAPE Distribution', fontsize=22, fontweight='bold')
    axs[2].tick_params(axis='both', which='major', labelsize=18) 

    axs[2].text(0.95, 0.95, 'MAPE(%): 1.139%', horizontalalignment='right', verticalalignment='top', transform=axs[2].transAxes, fontsize=26, fontweight='bold')

    output_dir = "./src/Exp1/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig_filename = os.path.join(output_dir, "ccs_vs_mz_plots.png")
    plt.tight_layout()
    plt.savefig(fig_filename)
    print(f"Plots saved to {fig_filename}")
    plt.close(fig)

if __name__ == '__main__':
    main()
