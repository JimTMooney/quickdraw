import torch
from torch.utils.data import DataLoader
import numpy as np

def create_loader(dataset, collate_fn, batch_size = 64):
    return DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size) 
                   
                   
def padding_collate_fn(data, max_pad=False, longest_seq=63):
    seqs, labels = zip(*data)
    batch_size = len(seqs)
    
    pad_length = 0
    seq_lengths = [seq.shape[0] for seq in seqs]
    if max_pad:
        pad_length = longest_seq
    else:
        pad_length = np.max(seq_lengths)
    
    desc_seqs = np.flip(np.argsort(seq_lengths))
    
    seq_batch = torch.zeros((batch_size, pad_length, 2))
    for idx, original_idx in enumerate(desc_seqs):
        seq_batch[idx, :seq_lengths[original_idx], :] = seqs[original_idx]
    
    seq_lengths = np.flip(np.sort(seq_lengths))
    
    return seq_batch, labels, torch.tensor(seq_lengths.copy())