import numpy as np
import matplotlib.pyplot as plt
import os
import torch

def render_drawing(drawing_data, class_name, ax=None, is_reversed=True):
    if not is_reversed:
        drawing_data = np.flip(drawing_data, axis=0)
        
    abs_data = np.cumsum(drawing_data, axis=0)
    if drawing_data.shape[1] == 3:
        start_mask = drawing_data[:, 2] == 1
        start_indices = np.arange(len(drawing_data))[start_mask]
        start_indices = np.append(start_indices, len(drawing_data))
    else:
        start_indices = [0, len(drawing_data)]
    
    if ax == None:
        plt.title('Sketch Drawing of ' + class_name)
        ax = plt
        
    for idx in range(1, len(start_indices)):
        start = start_indices[idx-1]
        end = start_indices[idx]
        ax.plot(abs_data[start:end, 0], abs_data[start:end, 1], c = 'blue')
        
def render_strokes(file_idx, class_name, split):
    class_path = os.path.join(os.getcwd(), "data_strokes")
    class_path = os.path.join(class_path, split)
    class_path = os.path.join(class_path, class_name)
    strokes = [os.path.join(class_path, stroke) for stroke \
                            in os.listdir(class_path) if str(file_idx) + '-' in stroke]
    n_strokes = len(strokes)
    width = 20.0
    height = width / float(n_strokes)
    fig, ax = plt.subplots(1, n_strokes, figsize=(width, height))
    for idx, stroke in enumerate(strokes):
        drawing = torch.load(stroke)
        render_drawing(drawing, class_name, ax[idx])