import torch
import torch.nn as nn
import torch.nn.functional as F

class Mix_Pooling(nn.Module):
    def __init__(self, embedding_size):
        super(Mix_Pooling, self).__init__()
        self.local_agg1 = self._make_cnnblock(embedding_size, embedding_size, 3, 1, 1)
        self.local_agg2 = self._make_cnnblock(embedding_size, embedding_size, 5, 1, 1)
        self.local_agg3 = self._make_cnnblock(embedding_size, embedding_size, 7, 1, 1)

        self.global_linear = nn.Linear(embedding_size, 1)
        
    def _make_cnnblock(self, in_channels, out_channels, kernel_size, dilation, nlayers):
        layers = []
        layers.append(nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation*int((kernel_size-1)/2), dilation=dilation))
        layers.append(nn.SELU())

        for _ in range(nlayers - 1):
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation*int((kernel_size-1)/2), dilation=dilation))
            layers.append(nn.SELU())
            in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, seq, mask):
        mask = (~mask).unsqueeze(-1).float()
        cls = seq[:, 0, :]
        seq = seq * mask
        
        res = self.global_linear(seq)
        res = torch.where(mask == 0.0, torch.tensor(float('-inf'), device=res.device), res)
        pooling_weight = F.softmax(res, dim=1)
        res = torch.sum(pooling_weight * seq, dim=1)
        
        seq = seq.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        seq = self.local_agg1(seq) * mask
        seq = self.local_agg2(seq) * mask
        seq = self.local_agg3(seq) * mask
        seq = seq.permute(0, 2, 1)
        mask = mask.permute(0, 2, 1)
        
        seq, max_indices = torch.max(seq, dim=1)
        
        return torch.cat([res, seq], dim=1)

class Embedding(nn.Module):
    def __init__(self, embedding_size, max_len):
        super(Embedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.seq_embedding = nn.Embedding(32, embedding_dim=embedding_size)
        self.position_embedding = nn.Embedding(max_len, embedding_size)
            
    def forward(self, seq):
        batch_size, _  = seq.shape
        position = torch.arange(0, self.max_len).expand(batch_size, self.max_len).to(seq.device)
        seq = self.seq_embedding(seq)
        seq = seq + self.position_embedding(position)
        return seq

class EncoderLayer(nn.Module):
    def __init__(self, num_layers, embedding_size, num_heads, dropout):
        super(EncoderLayer, self).__init__()
        self.num_layers = num_layers 
        d_ff = 4 * embedding_size
        self.attn = nn.ModuleList([nn.MultiheadAttention(embedding_size, num_heads, dropout, batch_first=True)
                                   for _ in range(num_layers)])

        self.mlp = nn.ModuleList([nn.Sequential(nn.Linear(embedding_size, d_ff), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_ff, embedding_size))
                                  for _ in range(num_layers)])

        self.norms1 = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        self.norms2 = nn.ModuleList([nn.LayerNorm(embedding_size) for _ in range(num_layers)])
        self.norm3= nn.LayerNorm(embedding_size)
        
    def forward(self, seq, mask):
        for i in range(self.num_layers):
            seq2 = self.norms1[i](seq)
            seq2, _ = self.attn[i](seq2, seq2, seq2, key_padding_mask=mask)
            seq = seq2 + seq
            
            seq2 = self.norms2[i](seq)
            seq2 = self.mlp[i](seq2)
            seq = seq2 + seq
            
        seq = self.norm3(seq)
        return seq

class Regression(nn.Module):
    def __init__(self, embedding_size, p):
        super(Regression, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(2 * embedding_size + 25, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, 1),
        )
    def forward(self, vector, length, seq, mz):
        length = length.unsqueeze(-1).to(vector.device)
        final_state = torch.cat([vector, length, seq, mz], dim=1)
        return self.mlp(final_state)

class PEP2CCS(nn.Module):
    def __init__(self, num_layers, embedding_size, num_heads, dropout, p, max_len=64):
        super(PEP2CCS, self).__init__()
        self.max_len = max_len
        self.Embedding = Embedding(embedding_size, max_len)
        self.Encoder = EncoderLayer(num_layers, embedding_size, num_heads, dropout)
        self.Pooling = Mix_Pooling(embedding_size)
        self.Regression = Regression(embedding_size, p)
    
    def forward(self, seq, charge, length, mz, ccs2, vector):
        mask = (seq == 0).clone().detach().to(seq.device)
        seq = self.Embedding(seq)
        seq = self.Encoder(seq, mask)

        seq = self.Pooling(seq, mask)
        out = self.Regression(vector, length, seq, torch.sqrt(mz)) + ccs2
        return out
