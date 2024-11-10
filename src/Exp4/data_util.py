import torch
import pandas as pd
from collections import Counter
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

fit_params = {
    2: (7.475148902379004, 1.3884478861522933),
    3: (5.686819960670556, 99.35526026933732),
    4: (5.419747913747912, 175.14669422777405)
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
    def __init__(self, seq, ccs, charge, length, mz, ccs2, vector):
        self.seq = torch.LongTensor(seq.tolist())
        self.ccs = torch.FloatTensor(ccs)
        self.ccs2 = torch.FloatTensor(ccs2)
        self.charge = torch.LongTensor(charge)
        self.length = torch.LongTensor(length)
        self.mz = torch.FloatTensor(mz)
        self.vector = torch.FloatTensor(vector.tolist())
        
    def __getitem__(self, idx):
        return self.seq[idx], self.ccs[idx], self.charge[idx], self.length[idx], self.mz[idx], self.ccs2[idx], self.vector[idx]
    
    def __len__(self):
        return len(self.seq)

def process_seq(data, max_length = 64):
    char_to_int = {char: idx + 6 for idx, char in enumerate(sorted(set(''.join(data['Sequence']))))}
    char_to_int['<PAD>'] = 0
    char_to_int['<SOS>'] = 1
    char_to_int['<EOS>'] = 2
    char_to_int['Charge2'] = 3
    char_to_int['Charge3'] = 4
    char_to_int['Charge4'] = 5

    def encode_and_pad(row):
        seq, charge = row['Sequence'], row['Charge']
        charge_token = char_to_int[f'Charge{charge}']
        sos = char_to_int['<SOS>']
        eos = char_to_int['<EOS>']
        encoded = [sos] + [charge_token] + [char_to_int[char] for char in seq] + [eos]
        padded = np.pad(encoded, (0, max_length - len(encoded)), mode='constant', constant_values=char_to_int['<PAD>'])
        return padded

    seq = data.apply(encode_and_pad, axis = 1)
    return np.array(seq.tolist(), dtype = np.int64)

def max_min_transformer(arr):
    max_val, min_val = np.max(arr), np.min(arr)
    normalized_arr = (arr - min_val) / (max_val - min_val)
    return normalized_arr, max_val, min_val

def denormalize_array(normalized_arr, max_val, min_val):
    denormalized_arr = normalized_arr * (max_val - min_val) + min_val
    return denormalized_arr

def sequence_to_vector(sequence):
    valid_amino_acids = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 
                         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    vector = np.zeros(len(valid_amino_acids), dtype = int)
    
    counter = Counter(sequence)
    
    for char, count in counter.items():
        if char in valid_amino_acids:
            index = valid_amino_acids.index(char)
            vector[index] = count
    
    return vector

def charge_to_onehot(charge, num_classes=3):
    one_hot = np.zeros(num_classes, dtype=int)
    one_hot[charge - 2] = 1  
    return one_hot

def get_data_set():
    data = pd.read_csv('../../ttrain.csv')
    train_data, valid_data = train_test_split(data, test_size = 0.01, random_state = 1115)
    train_data = train_data.reset_index(drop = True)
    valid_data = valid_data.reset_index(drop = True)
    train_seq = process_seq(train_data)
    valid_seq = process_seq(valid_data)
    
    train_vector = np.array([sequence_to_vector(seq) for seq in train_data['Sequence']])
    valid_vector = np.array([sequence_to_vector(seq) for seq in valid_data['Sequence']])
    
    train_onehot = np.array([charge_to_onehot(charge) for charge in train_data['Charge']])
    valid_onehot = np.array([charge_to_onehot(charge) for charge in valid_data['Charge']])
    
    train_vector = np.hstack((train_vector, train_onehot))
    valid_vector = np.hstack((valid_vector, valid_onehot))

    train_ccs, train_charge = np.array(train_data['CCS']).reshape(-1, 1), np.array(train_data['Charge']).reshape(-1, 1)
    valid_ccs, valid_charge = np.array(valid_data['CCS']).reshape(-1, 1), np.array(valid_data['Charge']).reshape(-1, 1)
    train_ccs, valid_ccs = np.log(train_ccs), np.log(valid_ccs)

    train_length = np.array(train_data['Length']) + 3
    valid_length = np.array(valid_data['Length']) + 3
    train_mz = np.array(train_data['m/z']).reshape(-1, 1)
    valid_mz = np.array(valid_data['m/z']).reshape(-1, 1)
    
    train_ccs2 = predict_ccs(train_mz, train_charge).reshape(-1, 1)
    valid_ccs2 = predict_ccs(valid_mz, valid_charge).reshape(-1, 1)
    train_ccs2, valid_ccs2 = np.log(train_ccs2), np.log(valid_ccs2)
    
    train_dataset = polypeptide(train_seq, train_ccs, train_charge, train_length, train_mz, train_ccs2, train_vector)
    valid_dataset = polypeptide(valid_seq, valid_ccs, valid_charge, valid_length, valid_mz, valid_ccs2, valid_vector)
    print("Data has been processed!!!")

    return train_dataset, valid_dataset

def get_test_data_set():
    test_data = pd.read_csv('./src/data/test_data.csv')
    test_seq = process_seq(test_data)
    test_vector = np.array([sequence_to_vector(seq) for seq in test_data['Sequence']])
    
    test_onehot = np.array([charge_to_onehot(charge) for charge in test_data['Charge']])
    test_vector = np.hstack((test_vector, test_onehot))

    test_ccs, test_charge = np.array(test_data['CCS']).reshape(-1, 1), np.array(test_data['Charge']).reshape(-1, 1)
    test_ccs = np.log(test_ccs)
    test_length = np.array(test_data['Length']) + 3
    test_mz = np.array(test_data['m/z']).reshape(-1, 1)
    
    test_ccs2 = predict_ccs(test_mz, test_charge).reshape(-1, 1)
    test_ccs2 = np.log(test_ccs2).reshape(-1, 1)
    test_dataset = polypeptide(test_seq, test_ccs, test_charge, test_length, test_mz, test_ccs2, test_vector)
    print("Data has been processed!!!")

    return test_dataset
