import sys
import os
import pandas as pd
import torch
from data_util import *
from model_params import * 
from model import *
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
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    model = PEP2CCS(model_params['num_layers'], model_params['embedding_size'], model_params['num_heads'], 0, 0).to(device)
    model_path = "./src/Exp4/model.pt" 
    chkp = torch.load(model_path, map_location = device)
    state_dict = chkp['model_param']
    model.load_state_dict(state_dict)
    
    max_len = 64

    test_size = len(data_set)
    pred_ccs = np.zeros([test_size, 1])
    ccs = np.zeros([test_size, 1])
    charge = np.zeros([test_size, 1])
    seq = np.zeros([test_size, max_len])
    Length = np.zeros([test_size, max_len])
    i = 0
    start = time.time()
    model.eval()
    loss_op = nn.MSELoss()
    test_loss = []
    test_dataloader = tqdm(test_dataloader, total=len(test_dataloader), desc="Model on test set", unit='batch')

    for batch_seq, batch_ccs, batch_charge, batch_length, batch_mz, batch_ccs2, t_vector in test_dataloader:
        with torch.no_grad():
            batch_seq = batch_seq.to(device)
            batch_ccs = batch_ccs.to(device)
            batch_ccs2 = batch_ccs2.to(device)
            batch_charge = batch_charge.to(device)
            batch_mz = batch_mz.to(device)
            batch_length = batch_length.to(device)
            t_vector = t_vector.to(device)
            
            pred = model(batch_seq, batch_charge, batch_length, batch_mz, batch_ccs2, t_vector)
            
            pred, batch_ccs = torch.squeeze(pred), torch.squeeze(batch_ccs)
            pred, batch_ccs = torch.exp(pred), torch.exp(batch_ccs)
            
            loss = loss_op(pred, batch_ccs)
            test_loss.append(loss.item())
            pred_ccs[i:i + batch_size, :] = pred.cpu().numpy().reshape(-1, 1)
            ccs[i:i + batch_size, :] = batch_ccs.cpu().numpy().reshape(-1, 1)
            charge[i:i + batch_size, :] = batch_charge.cpu().numpy().reshape(-1, 1)
            Length[i:i + batch_size, :] = batch_length.cpu().numpy().reshape(-1, 1)

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

if __name__ == '__main__':
    main()
