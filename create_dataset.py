import numpy as np
import os
import torch

def normalize_drawing(drawing_data, is_reversed=True):
    if not is_reversed:
        drawing_data = np.flip(drawing_data, axis=0)
    
    key_pts = drawing_data[:, :2]
    abs_pts = np.cumsum(key_pts, axis=0)
    min_bounds = np.min(abs_pts, axis=0)
    max_bounds = np.max(abs_pts, axis=0)
    full_sz = (max_bounds - min_bounds) + np.finfo(np.float32).eps
    
    normal_pts = ((abs_pts - min_bounds) / full_sz) - .5
    pt_diffs = np.diff(normal_pts, axis=0)
    starting_pt = np.expand_dims(normal_pts[0], axis=0)
    start_stroke = np.expand_dims(drawing_data[:, 2], axis=1)
    
    all_pts = np.concatenate((starting_pt, pt_diffs), axis=0)
    full_drawing = np.concatenate((all_pts, start_stroke), axis=1)
    
    return full_drawing

def create_directory(dir_name):
    if not os.path.isdir(dir_name):
        os.mkdir(dir_name)
        
def create_directories(dir_name, split_list=['train', 'valid', 'test']):
    create_directory(dir_name)
    for split in split_list:
        create_directory(os.path.join(dir_name, split))

def get_class_list():
    file_list = os.listdir(os.path.join(os.getcwd(), "raw_data"))
    class_list = [filename[:-4] for filename in file_list]
    
    return class_list

def write_full_samples(file_data, write_dir, split, class_name, n_samples):
    split_dir = os.path.join(write_dir, split)
    class_dir = os.path.join(split_dir, class_name)
    create_directory(class_dir)
    
    random_set = np.random.permutation(len(file_data))[:n_samples]
    subset_data = file_data[random_set]
    
    for idx, drawing in enumerate(subset_data):
        normal_drawing = normalize_drawing(drawing, is_reversed=False)
        filename = os.path.join(class_dir, str(idx) + ".pt")
        torch.save(torch.tensor(normal_drawing), filename)


def full_data_constructor(n_samples = [10, 1, 1]):
    class_list = get_class_list()
    
    write_dir = os.path.join(os.getcwd(), "data_full")
    split_list=['train', 'valid', 'test'] 
    create_directories(write_dir, split_list)
    
    read_dir = os.path.join(os.getcwd(), "raw_data")
    for class_name in class_list:
        class_file = class_name + ".npz"
        filename = os.path.join(read_dir, class_file)

        file_data = np.load(filename, encoding='latin1', allow_pickle=True)

        for idx, split in enumerate(split_list):
            split_data = file_data[split]
            n_split = n_samples[idx]
            write_full_samples(split_data, write_dir, split, class_name, n_split)
            
            
def add_strokes(drawing, class_dir, file_idx):
    stroke_mask = drawing[:, 2] == 1
    start_indices = np.arange(len(stroke_mask))[stroke_mask]
    start_indices = np.append(start_indices, len(stroke_mask))
    
    stroke_idx = 0
    for pt_idx in range(1, len(start_indices)):
        start = start_indices[pt_idx-1]
        end = start_indices[pt_idx]
        center_stroke = drawing[start:end, :2]
        center_stroke[0][0] = 0.0
        center_stroke[0][1] = 0.0
        center_stroke = center_stroke.clone().detach()
        filename = os.path.join(class_dir, str(file_idx) + '-' + str(stroke_idx) + '.pt')
        torch.save(torch.tensor(center_stroke), filename)
        stroke_idx += 1
        
def stroke_data_constructor():
    class_list = get_class_list()
    
    write_dir = os.path.join(os.getcwd(), "data_strokes")
    split_list=["train", "valid", "test"]
    create_directories(write_dir, split_list)
    
    read_dir = os.path.join(os.getcwd(), "data_full")
    for split in split_list:
        split_read= os.path.join(read_dir, split)
        split_write = os.path.join(write_dir, split)
        for class_name in class_list:
            class_read = os.path.join(split_read, class_name)
            class_write = os.path.join(split_write, class_name)
            create_directory(class_write)
            all_files = [os.path.join(class_read, file) for file in os.listdir(class_read) if file.endswith(".pt")]
            for file_idx, file in enumerate(all_files):
                drawing = torch.load(file)
                add_strokes(drawing, class_write, file_idx)
    

if __name__ == '__main__':
    full_data_constructor()
    stroke_data_constructor()