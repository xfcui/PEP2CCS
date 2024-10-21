import torch
import pandas as pd
import numpy as np
import string
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

fit_params = {
    2: (7.501715950104645, 0.572551240513163),
    3: (5.754739271491943, 93.65166570554298),
    4: (5.362181949494291, 178.73137126196363)
}

def trendline_func(m_z, a, b, charge):
    return a * charge * np.sqrt(m_z) + b

def predict_ccs(m_z, charge):
    m_z = np.array(m_z).reshape(-1)
    charge = np.array(charge).reshape(-1)
    
    predicted_ccs = np.zeros_like(m_z)
    
    for chg, (a, b) in fit_params.items():
        mask = (charge == chg)
        if mask.any():
            predicted_ccs[mask] = trendline_func(m_z[mask], a, b, chg)
    
    return predicted_ccs

class polypeptide(Dataset):
    def __init__(self, seq, ccs, charge, length, mz, ccs2):
        self.seq = torch.LongTensor(seq.tolist())
        self.ccs = torch.FloatTensor(ccs)
        self.charge = torch.LongTensor(charge)
        self.length = torch.LongTensor(length)
        self.mz = torch.FloatTensor(mz)
        self.ccs2 = torch.FloatTensor(ccs2)
    
    def __getitem__(self, idx):
        return self.seq[idx], self.ccs[idx], self.charge[idx], self.length[idx], self.mz[idx], self.ccs2[idx]
    
    def __len__(self):
        return len(self.seq)

def process_seq(data, max_length=64):
    
    valid_amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    char_to_int = {char: idx + 6 for idx, char in enumerate(valid_amino_acids)}
    char_to_int['<PAD>'] = 0
    char_to_int['<SOS>'] = 1
    char_to_int['<EOS>'] = 2
    char_to_int['Charge2'] = 3
    char_to_int['Charge3'] = 4
    char_to_int['Charge4'] = 5
    
    with open('char_to_int.json', 'w') as json_file:
        json.dump(char_to_int, json_file)
    print("##char_to_int is saved!")

    def encode_and_pad(row):
        seq, charge = row['Sequence'], row['Charge']
        charge_token = char_to_int[f'Charge{charge}']
        sos = char_to_int['<SOS>']
        eos = char_to_int['<EOS>']
        encoded = [sos] + [charge_token] + [char_to_int[char] for char in seq if char in char_to_int] + [eos]
        padded = np.pad(encoded, (0, max_length - len(encoded)), mode='constant', constant_values=char_to_int['<PAD>'])
        return padded

    seq = data.apply(encode_and_pad, axis=1)
    return np.array(seq.tolist(), dtype=np.int64)

def get_data_set():
    data = pd.read_csv('/root/PEP2CCS/data/train_data.csv')
    train_data, valid_data = train_test_split(data, test_size = 0.01, random_state = 1115)

    train_data = train_data.reset_index(drop=True)
    valid_data = valid_data.reset_index(drop=True)
    train_seq = process_seq(train_data)
    valid_seq = process_seq(valid_data)

    train_ccs, train_charge = np.array(train_data['CCS']).reshape(-1, 1), np.array(train_data['Charge']).reshape(-1, 1)
    valid_ccs, valid_charge = np.array(valid_data['CCS']).reshape(-1, 1), np.array(valid_data['Charge']).reshape(-1, 1)
    train_ccs, valid_ccs = np.log(train_ccs), np.log(valid_ccs)
    
    train_length = np.array(train_data['Length'])
    valid_length = np.array(valid_data['Length'])
    train_mz = np.array(train_data['m/z']).reshape(-1, 1)
    valid_mz = np.array(valid_data['m/z']).reshape(-1, 1)
    
    train_ccs2 = predict_ccs(train_mz, train_charge).reshape(-1, 1)
    valid_ccs2 = predict_ccs(valid_mz, valid_charge).reshape(-1, 1)
    train_ccs2, valid_ccs2 = np.log(train_ccs2).reshape(-1, 1), np.log(valid_ccs2).reshape(-1, 1)
    
    train_dataset = polypeptide(train_seq, train_ccs, train_charge, train_length, train_mz, train_ccs2)
    valid_dataset = polypeptide(valid_seq, valid_ccs, valid_charge, valid_length, valid_mz, valid_ccs2)
    print("Data has been processed!!!")

    return train_dataset, valid_dataset

def get_test_data_set():
    df = pd.read_csv('/root/PEP2CCS/data/test_data.csv')

    test_data = df
    
    test_seq = process_seq(test_data)
    test_ccs, test_charge = np.array(test_data['CCS']).reshape(-1, 1), np.array(test_data['Charge']).reshape(-1, 1)
    test_ccs = np.log(test_ccs)
    
    test_length = np.array(test_data['Length'])
    test_mz = np.array(test_data['m/z']).reshape(-1, 1)
    
    test_ccs2 = predict_ccs(test_mz, test_charge).reshape(-1, 1)
    test_ccs2 = np.log(test_ccs2).reshape(-1, 1)
    
    test_dataset = polypeptide(test_seq, test_ccs, test_charge, test_length, test_mz, test_ccs2)
    print("Data has been processed!!!")

    return test_dataset
