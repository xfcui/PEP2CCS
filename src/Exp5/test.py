import pandas as pd
import torch
from PEP2CCS.src.Exp5.data_util import get_test_data_set
from PEP2CCS.src.model.model_params import *
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
import numpy as np
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="Run different models based on input arguments.")
    parser.add_argument('--model_choice', type=int, default=1, help='Choose which model to import: 1 for model1, 2 for model2, etc.')
    parser.add_argument('--model_path', type=str, default="/root/PEP2CCS/checkpoint/model.pt", help='Path to the model checkpoint')
    
    args = parser.parse_args()
    
    if args.model_choice == 1:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model1")
    elif args.model_choice == 2:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model2")
    elif args.model_choice == 3:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model3")
    elif args.model_choice == 4:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model4")   
    elif args.model_choice == 5:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model5")   
    elif args.model_choice == 6:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model6")   
    elif args.model_choice == 7:
        model_module = importlib.import_module("PEP2CCS.src.Exp5.model7")   
        

    model_class = getattr(model_module, "PEP2CCS")

    print("##model params: {}", model_params)
    data = pd.read_csv('/root/ttest.csv')    
    
    data_set = get_test_data_set()
    test_dataloader = DataLoader(data_set, batch_size=batch_size, shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)
    
    # 加载模型
    model = model_class(model_params['num_layers'], model_params['embedding_size'], model_params['num_heads'], 0, 0).to(device)
    
    # 使用传入的 model_path
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