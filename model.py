import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F



class Encoder(nn.Module):
    def __init__(self, embed_dim, enc_hid_dim, num_layers, dropout):
        super().__init__()
        
        self.embedding = nn.Linear(2, embed_dim)
        
        self.rnn = nn.GRU(embed_dim, enc_hid_dim, num_layers=num_layers, bidirectional = True)
                
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_lens):
        embedded = self.dropout(self.embedding(src))
        
        packed_embedded = pack_padded_sequence(embedded, src_lens, batch_first=True)
        
        packed_outputs, hidden = self.rnn(packed_embedded)
        
        outputs, _ = pad_packed_sequence(packed_outputs, batch_first=True)
        
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1)
        
        return outputs, hidden
    
    
class Quantization(nn.Module):
    def __init__(self, hidden_dim, n_codebooks, K, tau = 1, alpha = 1, beta=.2):
        super().__init__()
        self.n_codebooks = n_codebooks
        self.code_size = int(hidden_dim/n_codebooks)
        self.tau = tau
        self.K = K
        self.alpha = alpha
        self.beta = .2
        
        self.codes = []
        for _ in range(self.n_codebooks):
            self.codes.append(torch.randn((self.code_size, self.K), requires_grad=True))
        
        
    def forward(self, hidden_state):
        # hidden state is [batch_size, n_features]
        next_hidden_state = torch.zeros_like(hidden_state)
        loss = 0
        for code_idx in range(self.n_codebooks):
            left = code_idx * self.code_size
            right = left + self.code_size
            
            current_code = self.codes[code_idx]
            
            hidden_features = hidden_state[:, left:right]
            
            logits = torch.mm(hidden_features, current_code)
            
            feature_norms = torch.linalg.norm(hidden_features, dim=1, keepdim=True)
            code_norms = torch.linalg.norm(current_code, dim=0, keepdim=True)
            logits /= feature_norms
            logits /= code_norms
            
            # attention is of size [batch_size, K]
            attention = F.gumbel_softmax(logits, tau=self.tau, hard=True)

            next_hidden_state[:, left:right] = torch.mm(attention, current_code.T)
            
            detached_codes = next_hidden_state[:, left:right].detach()
            detached_projections = hidden_state[:, left:right].detach()
            
            loss += self.beta * torch.sum(torch.linalg.norm(detached_codes - hidden_features, dim=1))
            loss += self.alpha * torch.sum(torch.linalg.norm(next_hidden_state[:, left:right] - detached_projections, dim=1))
            
        return next_hidden_state, loss
    
    
class Decoder(nn.Module):
    def __init__(self, z_dim, embed, dec_hid):
        super().__init__()
        
        self.embedding = nn.Linear(2, embed)
        
        