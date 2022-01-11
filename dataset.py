import torch
from torch.utils.data import Dataset
import os
from itertools import repeat
import numpy as np

class StrokeDataset(Dataset):
    def __init__(self, stroke_dir, split):
        all_strokes_path = os.path.join(stroke_dir, split)
        all_classes = [os.path.join(all_strokes_path, sketch_class) for sketch_class in os.listdir(all_strokes_path)]
        
        all_tuples = set()
        self.class_list = []
        
        for class_idx, sketch_class in enumerate(all_classes):
            file_list = [os.path.join(sketch_class, file) for file in os.listdir(sketch_class)]
            self.class_list.append(os.path.basename(sketch_class))
            file_tuples = set(zip(file_list, repeat(class_idx)))
            all_tuples.update(file_tuples)
            
        full_data = list(all_tuples)
        
        permutation = np.random.permutation(len(full_data))
        
        permute_data = []

        for idx in permutation:
            permute_data.append(full_data[idx])
            
        self.full_data = permute_data
        
        
    def __len__(self):
        return len(self.full_data)
        
    def __getitem__(self, idx):
        seq_path, label = self.full_data[idx]
        sketch_seq = torch.load(seq_path)
        return sketch_seq, label
    
    def get_class_by_idx(self, idx):
        return self.class_list[idx]