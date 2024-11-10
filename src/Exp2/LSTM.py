import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

class Embedding(nn.Module):
    def __init__(self, embedding_size):
        super(Embedding, self).__init__()
        self.embedding_size = embedding_size
        self.seq_embedding = nn.Embedding(32, embedding_size)
    def forward(self, seq):
        seq = self.seq_embedding(seq)
        return seq

class EncoderLayer(nn.Module):
    def __init__(self, num_layers, embedding_size, dropout):
        super(EncoderLayer, self).__init__()
        self.num_layers = num_layers
        self.LSTM = nn.LSTM(input_size = embedding_size, hidden_size = embedding_size, num_layers = num_layers, bias = False, batch_first = True, dropout = dropout, bidirectional = True)        
    
    def forward(self, seq, length):
        seq = pack_padded_sequence(seq, length, batch_first = True, enforce_sorted = False)
        out, (hn, _) = self.LSTM(seq)
        out, _ = pad_packed_sequence(out, batch_first=True)
        final_state = torch.cat([hn[-1, :, :], hn[-2, :, :]], dim = 1)
        return final_state
    
class Regression(nn.Module):
    def __init__(self, embedding_size, dropout):
        super(Regression, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size + 1, embedding_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_size, 1)
        )
    def forward(self, final_state, charge):
        final_state = torch.concat([final_state, charge], dim = 1)
        return self.mlp(final_state)

        
class PEP2CCS(nn.Module):
    def __init__(self, num_layers, embedding_size, dropout, max_len = 64):
        super(PEP2CCS, self).__init__()
        self.max_len = max_len
        self.Embedding = Embedding(embedding_size)
        self.Encoder = EncoderLayer(num_layers, embedding_size, dropout)
        self.Regression = Regression(embedding_size, dropout)
    
    def forward(self, seq, charge, length):
        seq = self.Embedding(seq)
        seq = self.Encoder(seq, length)
        out = self.Regression(seq, charge)
        return out
    
