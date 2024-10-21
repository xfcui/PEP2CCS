import torch
import torch.nn as nn
import torch.nn.functional as F

class Mix_Pooling(nn.Module):
    def __init__(self, embedding_size):
        super(Mix_Pooling, self).__init__()

    def forward(self, seq, mask):
        mask = (~mask).unsqueeze(-1).float()  # 反转mask并扩展最后一维
        cls = seq[:, 1, :]  # 获取第一个位置的CLS token

        # 将mask位置置为 -1e9
        seq = seq * mask + (1.0 - mask) * -1e9

        # 在维度1 (max_len) 上进行 max pooling，保持输入的形状
        seq, max_indices = torch.max(seq, dim=1)

        if self.training:
            return torch.cat([cls, seq], dim=1)
        else:
            return torch.cat([cls, seq], dim=1), max_indices

class Embedding(nn.Module):
    def __init__(self, embedding_size, max_len):
        super(Embedding, self).__init__()
        self.max_len = max_len
        self.embedding_size = embedding_size
        self.seq_embedding = nn.Embedding(32, embedding_dim = embedding_size)
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
            nn.Linear(2 * embedding_size + 2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(embedding_size, embedding_size // 2),
            nn.LayerNorm(embedding_size // 2),
            nn.ReLU(),
            nn.Linear(embedding_size // 2, 1)
        )
    def forward(self, seq, length, mz):
        length = length.unsqueeze(-1)
        final_state = torch.cat([seq, length, mz], dim=1)
        return self.mlp(final_state)

class Seq2CCS(nn.Module):
    def __init__(self, num_layers, embedding_size, num_heads, dropout, p, max_len=64):
        super(Seq2CCS, self).__init__()
        self.max_len = max_len
        self.Embedding = Embedding(embedding_size, max_len)
        self.Encoder = EncoderLayer(num_layers, embedding_size, num_heads, dropout)
        self.Pooling = Mix_Pooling(embedding_size)
        self.Regression = Regression(embedding_size, p)
    
    def forward(self, seq, charge, length, mz, ccs2):
        mask = (seq == 0).clone().detach().to(seq.device)
        seq = self.Embedding(seq)
        seq = self.Encoder(seq, mask)
        
        if self.training:
            seq = self.Pooling(seq, mask)
            out = self.Regression(seq, length, torch.sqrt(mz)) + ccs2
            return out
        else:
            seq, max_indices = self.Pooling(seq, mask)
            out = self.Regression(seq, length, torch.sqrt(mz)) + ccs2
            return out, max_indices
