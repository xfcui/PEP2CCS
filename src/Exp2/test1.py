import pandas as pd
import torch
from data_util import *
from PEP2CCS.src.model.model import *
from PEP2CCS.src.model.model_params import *
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the PEP2CCS model with custom model path and data selection.")
    parser.add_argument('--model_path', type=str, default="/root/PEP2CCS/checkpoint/model.pt", help='Path to the model checkpoint')
    parser.add_argument('--use_full_data', type=int, default=1, help='Use full data (1) or partial data (0)')
    args = parser.parse_args()
    
    print("##model params: {}", model_params)
    data = pd.read_csv('/root/ttest.csv')    
    
    if args.use_full_data == 1:
        data_set = get_test_data_set()
        print("Using full test data set.")
    else:
        data_set = get_partial_test_data_set()
        print("Using partial test data set.")
    
    test_dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    model = PEP2CCS(model_params['num_layers'], model_params['embedding_size'], model_params['num_heads'], 0, 0).to(device)
    
    model_path = args.model_path
    chkp = torch.load(model_path)
    state_dict = chkp['model_param']
    model.load_state_dict(state_dict)
    
    
    max_len = 64
    batch_size = 1
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
    
if __name__ == 'main':
    main()