import argparse
import pandas as pd
import torch
from data_util import *
from model import *
from model_params import model_params
from torch.utils.data import DataLoader
import os
import time
from tqdm import tqdm
import numpy as np

def main(data_path):
    print("##model params:", model_params)
    
    data = pd.read_csv(data_path)
    
    try:
        data_set = get_test_data_set()
    except Exception as e:
        print(f"Error loading data set: {e}")
        return
    
    batch_size = model_params.get('batch_size', 1)
    test_dataloader = tqdm(DataLoader(data_set, batch_size=batch_size, shuffle=False), total=len(data_set) // batch_size + 1, desc="Model on test set", unit='batch')
    
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    
    model = PEP2CCS(model_params['num_layers'], model_params['embedding_size'], model_params['num_heads'], 0, 0).to(device)
    model_path = "./src/checkpoint/model.pt"
    
    if os.path.exists(model_path):
        try:
            chkp = torch.load(model_path, map_location=device)
            model.load_state_dict(chkp['model_param'])

        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            return
    else:
        print("Model path does not exist.")
        return
    
    max_len = 64
    test_size = len(data_set)
    pred_ccs, ccs, charge, seq, Length, max_pos = [], [], [], [], [], []

    start = time.time()
    model.eval()
    loss_op = torch.nn.MSELoss()
    test_loss = []

    for batch_seq, batch_ccs, batch_charge, batch_length, batch_mz, batch_ccs2 in test_dataloader:
        with torch.no_grad():
            batch_seq = batch_seq.to(device)
            batch_ccs = batch_ccs.to(device)
            batch_ccs2 = batch_ccs2.to(device)
            batch_charge = batch_charge.to(device)
            batch_mz = batch_mz.to(device)
            batch_length = batch_length.to(device)
            
            pred, max_indices = model(batch_seq, batch_charge, batch_length, batch_mz, batch_ccs2)
            
            idx = torch.argmax(max_indices, dim=1)
            pred, batch_ccs = torch.squeeze(pred), torch.squeeze(batch_ccs)
            
            pred, batch_ccs = torch.exp(pred.clamp(-10, 10)), torch.exp(batch_ccs.clamp(-10, 10))
            
            loss = loss_op(pred, batch_ccs)
            test_loss.append(loss.item())
            pred_ccs.append(pred.cpu().numpy().reshape(-1, 1))
            ccs.append(batch_ccs.cpu().numpy().reshape(-1, 1))
            charge.append(batch_charge.cpu().numpy().reshape(-1, 1))
            Length.append(batch_length.cpu().numpy().reshape(-1, 1))
            max_pos.append(idx.cpu().numpy().reshape(-1, 1))
            seq.append(batch_seq.cpu().numpy().reshape(-1, max_len))

    avr_loss = sum(test_loss) / len(test_loss)
    print("##Finish!")
    print(f'##Time: {(time.time() - start) / 60}')
    
    data['Pred CCS'] = np.vstack(pred_ccs).ravel()
    data.to_csv('./predict_data.csv')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PEP2CCS Model Inference')
    parser.add_argument('--data_path', default = '../data/test_data.csv', type=str, required=True, help='Path to the test CSV file')
    args = parser.parse_args()
    
    main(args.data_path)
